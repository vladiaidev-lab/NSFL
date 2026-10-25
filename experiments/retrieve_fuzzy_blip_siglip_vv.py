### RETRIEVE_FUZZY_BLIP_SIGLIP_VV.PY

import pickle
from html import parser
import os
import ssl
from tqdm import tqdm
import argparse
import json
import numpy as np
import faiss
import torch
import datetime
#from dataset import get_boolquestions_evalset, get_fuzzy_evalset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from operator import itemgetter
from fuzzy_logic import FuzzyLogic, NumpyEncoder, create_geometric_logic_vector
from objective_hnsw import ObjectiveHNSW

from models.models import ModelWrapper
from PIL import Image
import requests


if torch.xpu.is_available():
    device = torch.device("xpu")
    print(f"Using device: {torch.xpu.get_device_name(0)}")
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MODEL = None  #name
model = None  #ModelWrapper("blip", device)
MODEL_NAME = None

def loadmodel(model_name):
    global model, MODEL_NAME
    model = ModelWrapper(model_name, device)
    MODEL_NAME = model_name



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="coco", help="Coco / Quest / ?")
    parser.add_argument("--subdataset", type=str, default="val", help="val / test / train")
    parser.add_argument("--default_path", type=str, default="logic_demo\\COCO-LOGIC\\", help="Folder of queries, data, index")
    parser.add_argument("--images_path", type=str, default="logic_demo/COCO2017/", help="Images path")
    parser.add_argument("--model_name", type=str, default='blip-large', help="Blip or Siglip.")
    parser.add_argument("--augment_type", type=str, default="QUEST", help="QUEST (that are also...) / SIMPLE (only 'AND')")
    parser.add_argument("--input_path", type=str, default=None, help="Path to images list as documents. If None, constructed from dataset/subdataset.")
    parser.add_argument("--queries_path", type=str, default=None, help="Path to the queries file. If None, defaults to data/coco_logic_600_<subdataset>_{quest_style|simple}.json, chosen by --augment_type.")
    
    parser.add_argument("--topk", type=int, default=100, help="Number of documents to retrieve per query.")
    parser.add_argument("--topk_max", type=int, default=1000, help="Number of documents to retrieve per query.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for retrieval.")
    parser.add_argument("--index_batch_size", type=int, default=1000, help="Batch size for building the index.")
    parser.add_argument("--rebuild_index", action="store_true", help="Whether to rebuild the index.")
    parser.add_argument("--is_faiss", type=int, default=1, help="is faiss db or in memory search")
    parser.add_argument("--is_debug", type=int, default=0, help="is debug")
    parser.add_argument("--is_queries", type=int, default=0, help="is data in queries")
    parser.add_argument("--is_decompose", type=int, default=1, help="decompose to boolean queries or not")
    parser.add_argument("--name", type=str, default='', help="special name of run")
    
    parser.add_argument("--output_path", type=str, default='debug_out/retrieval/', help="Path to save the retrieval results.")
    parser.add_argument('--query_encoder', type=str, default='', help="Name or path of the query encoder model, not in use for images.")

    parser.add_argument("--is_geometric", type=int, default=0, help="Activate Geometric Vector Logic")
    parser.add_argument("--is_opt_onsphere", type=int, default=0, help="Activate custom QueryVector On-Sphere Optimization to fit Logic-Formulas ")
    parser.add_argument("--fuzzy_type", type=str, default="NSFL", help="NSFL / NSFL-DEBUG / GEO")
    parser.add_argument('--sqo_stats_path', type=str, default=None,
                        help="If set (and is_opt_onsphere=1), dump per-query SQO "
                             "convergence stats to this JSON file.")
    parser.add_argument('--sqo_steps', type=int, default=100, help="SQO: max gradient steps.")
    parser.add_argument('--sqo_lr', type=float, default=0.2, help="SQO: learning rate (alpha).")
    parser.add_argument('--sqo_patience', type=int, default=5, help="SQO: early-stopping patience.")
    parser.add_argument('--sqo_min_step_before_stop', type=int, default=None,
                        help="SQO: minimum step before early stopping is allowed to "
                             "fire. Default None reproduces steps//3 exactly; pass 0 "
                             "for patience-alone (EXP 2b winning config).")
    parser.add_argument('--sig_gate', type=int, default=0,
                        help="Significance-gated conjunctive correction (see "
                             "fuzzy_logic.py). 0 (default) = existing unconditional "
                             "AND2/AND3 correction, bit-identical to before this flag "
                             "existed. 1 = gate on. Only affects 'atom0 & atom1' and "
                             "'atom0 & atom1 & atom2'.")

    args = parser.parse_args()
    args.images_path = f"{args.images_path}{args.subdataset}\\"
    
    if args.input_path is None:
        args.input_path = f"{args.default_path}{args.dataset}_corpus_{args.subdataset}.json"
    if args.queries_path is None:
        coco_logic_realization = {"QUEST": "quest_style", "SIMPLE": "simple"}[args.augment_type.upper()]
        args.queries_path = f"data/coco_logic_600_{args.subdataset}_{coco_logic_realization}.json"

    if (('blip' in args.model_name) or ('siglip' in args.model_name)):
        if ('blip' in args.model_name):
            if ('large' in args.model_name):
                args.faiss_data_path = f"{args.default_path}faiss_{args.dataset}_{args.subdataset}_blip_large"                
            else:
                args.faiss_data_path = f"{args.default_path}faiss_{args.dataset}_{args.subdataset}_blip_base"
        elif ('siglip' in args.model_name):
            args.faiss_data_path = f"{args.default_path}faiss_{args.dataset}_{args.subdataset}_{args.model_name}"
        
        
    args.saved_queries_path = args.faiss_data_path + '\\'
    return args


def load_document_embeddings(documents_file, is_encode_all):
    print("Inside load_document_embeddings()")
    dict_docids_img = {}
    docids = []
    embed_list = []
    urls_images = []
    with open(args.input_path, 'r') as file:
        data = json.load(file)
        for i, item in enumerate(data):
            urls_images.append(item["image"])      # loads the entire list
            docids.append(str(i))
            dict_docids_img[str(i)] = item["image"]

    if is_encode_all:  
        image_list = [args.images_path + item['title'] for item in data]
        
        batch_size = args.batch_size
        number_of_images = len(image_list)
        embed_list = []
        #loop over number_of_images for each batch
        for i in range(0, number_of_images, batch_size):
            print(f"Processing {i+1}/{number_of_images} images")
            images_per_batch = min(batch_size, number_of_images-i)
            images = image_list[i:i+images_per_batch]
            print(image_list[i:i+images_per_batch])
            if ('\\' in args.images_path):
                images = [Image.open(image) for image in images]
            else:
                images = [Image.open(requests.get(image, stream=True).raw) for image in images]

            embeds = model.encode_images(images)
            embed_list.append(embeds)
            print(f"Processing {i+1}/{number_of_images} images")

        embed_list = torch.cat(embed_list, dim=0)        

    return docids, embed_list, dict_docids_img


def encode_query(query):
    # a_embeds = query_encoder.encode([query], prompt_name=prompt_name)
    # a_embeds = np.array(a_embeds)
    # a_embeds = np.concatenate(a_embeds, axis=0)
    query = f"a photo of a {query}"
    if ("siglip" in args.model_name): 
        a_embeds = model.encode_texts([query]).cpu().numpy()
    elif ('blip' in args.model_name):
        a_embeds = model.encode_texts([query])
    return a_embeds



def get_fuzzy_evalset_queries(eval_data):    
    passage_id = 'passage_id'
    qid2question = {}
    qid2query = {}
    qid2docids = {}
    qid2formulas = {}
    qid2qtypes = {}
    qid2neg_docids = {}
    for d in eval_data:
        qid = d["qid"] 
        qid2query[qid] = d["atoms"]
        qid2formulas[qid] = d["atom_formulas"]
        qid2qtypes[qid] = d["question_type"]
        qid2question[qid] = d["question"]
        qid2docids[qid] = [pos_ctx[passage_id] for pos_ctx in d["positive_ctxs"]]
        qid2neg_docids[qid] = [pos_ctx[passage_id] for pos_ctx in d["negative_ctxs"]]

    return qid2question, qid2query, qid2docids, qid2neg_docids, qid2formulas, qid2qtypes

        

def load_all_data(queries, is_encode_all, saved_queries_path):
    if "Qwen" in args.query_encoder:
        model_kwargs = {"torch_dtype":torch.float16}
    else:
        model_kwargs = None
    
    
    questions, eval_qid2query, eval_qid2docids, qid2neg_docids, eval_qid2formulas, eval_qid2qtypes = get_fuzzy_evalset_queries(queries)
    qids = sorted(list(eval_qid2docids.keys()))
   
    #load or encode queries
    question_embeds, query_embeds, qtype_index, qformula_index, qatoms_names, gt_docs, gt_neg_docs = load_queries_embeddings(
        saved_queries_path, questions, qids, eval_qid2query, eval_qid2docids, qid2neg_docids, eval_qid2formulas, eval_qid2qtypes, args.queries_path)
      
    #load documents
    docids, doc_embeds, dict_docids_img = load_document_embeddings(args.input_path, is_encode_all)

    #return qids, question_embeds, query_embeds, docids, doc_embeds, qtype_index, qformula_index, qatoms_names, gt_docs, gt_neg_docs
    return qids, question_embeds, query_embeds, docids, doc_embeds, qtype_index, qformula_index, qatoms_names, gt_docs, gt_neg_docs, dict_docids_img



def load_queries_embeddings(saved_queries_path, questions, qids, eval_qid2query, eval_qid2docids, qid2neg_docids, eval_qid2formulas, eval_qid2qtypes, queries_path):
    queries_saved_file = os.path.join(saved_queries_path, f"{args.dataset}_{args.subdataset}_queries(augment-{args.augment_type.lower()}).pkl") if saved_queries_path is not None else None
    if 'debug_' in queries_path:
        queries_saved_file = os.path.join(saved_queries_path, f"{args.dataset}_{args.subdataset}_debug_queries(augment-{args.augment_type.lower()}).pkl") if saved_queries_path is not None else None

    
    if queries_saved_file is not None and os.path.exists(queries_saved_file):                  
        #load from saved fom pickle file
        with open(queries_saved_file, 'rb') as f:
           queries = pickle.load(f)
        question_embeds, query_embeds, qtype_index, qformula_index, qatoms_names, gt_docs, gt_neg_docs = queries['question_embeds'], queries['query_embeds'], queries['qtype_index'], queries['qformula_index'], queries['qatoms_names'], queries['gt_docs'], queries['gt_neg_docs']
    else:
        #encode queries
        question_embeds, query_embeds, qtype_index, qformula_index, qatoms_names, gt_docs, gt_neg_docs = encode_queries_embedding(
            questions, qids, eval_qid2query, eval_qid2docids, qid2neg_docids, eval_qid2formulas, eval_qid2qtypes)
        if saved_queries_path is not None:
            queries = {'question_embeds': question_embeds, 'query_embeds': query_embeds, 'qtype_index': qtype_index, 'qformula_index': qformula_index, 'qatoms_names': qatoms_names, 'gt_docs': gt_docs, 'gt_neg_docs': gt_neg_docs, 'qids': qids}
            os.makedirs(saved_queries_path, exist_ok=True)
            with open(queries_saved_file, 'wb') as f:
                pickle.dump(queries, f, protocol=pickle.HIGHEST_PROTOCOL)        

    return  question_embeds, query_embeds, qtype_index, qformula_index, qatoms_names, gt_docs, gt_neg_docs



def encode_queries_embedding(questions, qids, eval_qid2query, eval_qid2docids, qid2neg_docids, eval_qid2formulas, eval_qid2qtypes):
    query_embeds = []
    qtype_index = []
    qformula_index = []
    qatoms_names = []
    gt_docs = []
    gt_neg_docs = []
    question_embeds = []

    if args.augment_type.upper() == "QUEST":
        separator1 = ' that are also ' 
        separator1_1 = ' that are also ' 
        separator1_2 = ' that are also both ' 
        separator2 = ' and ' 
    else:
        separator1 = ' AND '
        separator1_1 = ' AND '
        separator1_2 = ' AND '
        separator2 = ' AND '

    for queryid in tqdm(qids, desc="Encoding queries"):
        eval_qid2atoms =  eval_qid2query[queryid]
        atom_embeds = []
        atom_qtype = eval_qid2qtypes[queryid]
        qformulas = eval_qid2formulas[queryid]
        question_embed = encode_query(questions[queryid])# normalize_embeddings=True)
        for atom in eval_qid2atoms.keys():
            a_embeds = encode_query(eval_qid2atoms[atom])
            atom_embeds.append(a_embeds)       
        formula1 = qformulas['formula1']
        
        # Augment queries by atoms combinations
        if formula1 == "atom0 & !atom1" or formula1 == "atom0 & atom1" or formula1 == "atom0 | atom1":
            atomAB = eval_qid2atoms['atom0'] + separator1 + eval_qid2atoms['atom1']
            ab_embeds = encode_query(atomAB)
            atom_embeds.append(ab_embeds)      
        
        elif formula1 == "atom0 & atom1 & !atom2" or formula1 == "atom0 & atom1 & atom2" or formula1 == "atom0 | atom1 | atom2" or formula1=="atom0 | atom1 & !atom2":
            atomAB = eval_qid2atoms['atom0'] + separator1_1 + eval_qid2atoms['atom1']
            ab_embeds = encode_query(atomAB)
            atom_embeds.append(ab_embeds)      

            atomAC = eval_qid2atoms['atom0'] + separator1_1 + eval_qid2atoms['atom2']
            ac_embeds = encode_query(atomAC)
            atom_embeds.append(ac_embeds)      

            atomBC = eval_qid2atoms['atom1'] + separator1_1 + eval_qid2atoms['atom2']
            bc_embeds = encode_query(atomBC)
            atom_embeds.append(bc_embeds)  

            atomABC = eval_qid2atoms['atom0'] + separator1_2 + eval_qid2atoms['atom1'] + separator2 + eval_qid2atoms['atom2']
            abc_embeds = encode_query(atomABC)
            atom_embeds.append(abc_embeds)  


        qatoms_names.append(list(eval_qid2atoms.keys()))
        query_embeds.append(atom_embeds)
        qtype_index.append(atom_qtype)
        qformula_index.append(qformulas)
        gt_docs.append(eval_qid2docids[queryid])
        gt_neg_docs.append(qid2neg_docids[queryid])
        question_embeds.append(question_embed)

    return  question_embeds, query_embeds, qtype_index, qformula_index, qatoms_names, gt_docs, gt_neg_docs



def get_cosine_similarity(q_embeds, doc_embeds):
    if ('siglip' in MODEL_NAME):
        q_embeds = np.array(q_embeds)
        shapes = q_embeds.shape
        a = q_embeds.reshape(shapes[0] * shapes[1], -1)
        
        # Normalize both
        a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
        doc_normed = doc_embeds / (np.linalg.norm(doc_embeds, axis=1, keepdims=True) + 1e-8)
        
        b = doc_normed.transpose()
        res = a @ b
        scores = res.reshape(shapes[0], shapes[1], -1).transpose([0, 2, 1])
    else:
        q_embeds = np.array(q_embeds)
        shapes = q_embeds.shape
        a = q_embeds.reshape(shapes[0]*shapes[1],-1)
        b = doc_embeds.transpose()
        res = a @ b
        #res = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        scores = res.reshape(shapes[0],shapes[1],-1).transpose([0,2,1])
   
    return scores




def index_documents(input_path, docids, doc_embeds, index_batch_size=10000, rebuild_index=False):
    index_path = os.path.join(input_path, "index")
    os.makedirs(index_path, exist_ok=True)
    index_file = os.path.join(index_path, "index")
    index_meta_file = os.path.join(index_path, "index_meta.json")
    if os.path.exists(index_file) and os.path.exists(index_meta_file) and not rebuild_index:
        print(f"$$$$$$$$$ Loading index from {index_path}. $$$$$$$$$")
        index = faiss.read_index(index_file)
        with open(index_meta_file) as f:
            docids = json.load(f)["docids"]
        print(f"Index loaded. ntotal:", index.ntotal)
    else:
        if 'siglip' in MODEL_NAME:
            doc_embeds = doc_embeds.cpu().numpy()
        
        if 'siglip2' in MODEL_NAME:
            doc_embeds = doc_embeds / (np.linalg.norm(doc_embeds, axis=1, keepdims=True) + 1e-8)

        print("^^^^^^^^^^^ Building index from scratch.^^^^^^^^^^^^^")
        # docids, doc_embeds = load_document_embeddings(input_path, is_encode_all)
        index = faiss.IndexFlatIP(doc_embeds.shape[1])
        doc_embeds_batches = np.array_split(doc_embeds, max(1, len(doc_embeds) // index_batch_size), axis=0)
        for i, doc_embeds_batch in enumerate(tqdm(doc_embeds_batches, desc="Building index")):
            index.add(doc_embeds_batch)
        faiss.write_index(index, index_file)
        with open(index_meta_file, "w") as f:
            json.dump({"docids": docids}, f)
        print("Index built. ntotal:", index.ntotal)

    return index, docids

def is_faiss_exists(faiss_data_path):
    index_path = os.path.join(faiss_data_path, "index")
    index_file = os.path.join(index_path, "index")
    index_meta_file = os.path.join(index_path, "index_meta.json")
    if os.path.exists(index_file) and os.path.exists(index_meta_file):
        return True
    return False






if __name__ == '__main__':
    args = parse_args()
    loadmodel(args.model_name)

    debug = args.is_debug
    is_faiss = args.is_faiss
    query_debug = False
    query_debug_id = '1'
    fl = FuzzyLogic(type=args.fuzzy_type.upper(), sig_gate=bool(args.sig_gate))

    questions = {}
    with open(args.queries_path) as json_data:
        queries_data = json.load(json_data)
        json_data.close()
    if debug:
        for q in queries_data:
            questions[q['qid']] = {
                'question': q['question'],
                'atoms': q['atoms'],
                'positive_ctxs': q['positive_ctxs'],
                'negative_ctxs': q['negative_ctxs']
            }

    is_encode_all = not is_faiss or not is_faiss_exists(args.faiss_data_path)    
    qids, question_embeds, query_embeds, docids, doc_embeds, ops_index, qformula_index, qatoms_names, gt_pos_docs, gt_neg_docs, dict_docids_img = \
        load_all_data(queries_data, is_encode_all, args.saved_queries_path)

    if is_faiss:
        doc_index, docids = index_documents(args.faiss_data_path, docids, doc_embeds, args.index_batch_size, args.rebuild_index)
        # get all vectors from faiss
        doc_embeds = doc_index.reconstruct_n(0, doc_index.ntotal)
        topk_max = min(args.topk_max, doc_index.ntotal)

    # Check if the output path exists, if not create it
    debug_results = []
    if args.output_path is None:
        args.output_path = os.path.join(args.input_path, "retrieval")
    os.makedirs(args.output_path, exist_ok=True)
    
    # Save the results
    date = datetime.datetime.now().date()
    hour = datetime.datetime.now().strftime("%H")
    minute = datetime.datetime.now().strftime("%M")

    if args.is_opt_onsphere:
        ObjHNSW = ObjectiveHNSW()
    _sqo_stats_log = []

    # with open(os.path.join(args.output_path, f"results.decompose_{args.is_decompose}_{args.name}_{date}_val_{hour}:{minute}_{MODEL_NAME}.txt"), "w") as f:
    with open(os.path.join(args.output_path, f"results.decompose{args.is_decompose}_{args.name}.txt"), "w") as f:
        i=0
        # for qid, question_embed, query_atoms_embeds, operator, atoms_list, formulas, gt_pos_doc, gt_neg_doc in zip(qids, question_embeds, query_embeds, qtype_index, qatoms_names, qformula_index, gt_pos_docs, gt_neg_docs):
        for qid, question_embed, query_atoms_embeds, operator, atoms_list, formulas, gt_pos_doc, gt_neg_doc in zip(qids, question_embeds, query_embeds, ops_index, qatoms_names, qformula_index, gt_pos_docs, gt_neg_docs):
            i = i+1
            # # debug spcific query
            # formula1 = formulas['formula1']
            # if formula1 != "atom0 & atom1 & atom2":
            #     continue

            if args.is_geometric:
                new_vector = create_geometric_logic_vector(formulas['formula1'], question_embed, query_atoms_embeds)
                question_embed = new_vector
            
            if args.is_opt_onsphere:
                atoms_embeds = []
                for a in query_atoms_embeds:
                    atoms_embeds.append(a if isinstance(a, torch.Tensor) else torch.from_numpy(a))

                full_embed = question_embed if isinstance(question_embed, torch.Tensor) else torch.from_numpy(question_embed)
                obj_fn = fl.get_formula_objective(formulas['formula1'], full_embed, atoms_embeds)
                
                new_query_embed, _sqo_stats = ObjHNSW.optimize_on_sphere(
                    obj_fn, dim=question_embed.shape[1], steps=args.sqo_steps, lr=args.sqo_lr,
                    early_stopping_patience=args.sqo_patience, init_value=question_embed[0],
                    return_stats=True, min_step_before_stop=args.sqo_min_step_before_stop,
                )
                if args.sqo_stats_path is not None:
                    _sqo_stats_log.append(_sqo_stats)
                question_embed = np.expand_dims(new_query_embed, 0)
             

            if is_faiss:     
                monolitic_scores, monolitic_ids = doc_index.search(question_embed, topk_max)                
                target_doc_embeds = doc_embeds[monolitic_ids].squeeze()
                query_documents_scores = get_cosine_similarity([query_atoms_embeds], target_doc_embeds)[0] 
                query_documents_ids = monolitic_ids.transpose()
                results = fl.run_query_rescoring_docids(qid, monolitic_scores.transpose(), monolitic_ids.transpose(), query_documents_scores, query_documents_ids, atoms_list, formulas, gt_pos_doc, gt_neg_doc, question_embed, query_atoms_embeds, query_debug_id = query_debug_id, topk=args.topk, debug=debug, target_doc_embeds=target_doc_embeds) 
            else:
                monolitic_scores = get_cosine_similarity([question_embed], doc_embeds)[0]
                if query_debug:
                    if qid == query_debug_id:
                        query_documents_scores = get_cosine_similarity([query_atoms_embeds, query_atoms_embeds], doc_embeds)[0]
                        fl.run_query_rescoring(qid, monolitic_scores, query_documents_scores, atoms_list, formulas, gt_pos_doc, gt_neg_doc, query_debug_id = query_debug_id, topk=args.topk, debug=debug)
                        break          
                else:
                    query_documents_scores = get_cosine_similarity([query_atoms_embeds], doc_embeds)[0]
                    results = fl.run_query_rescoring(qid, monolitic_scores, query_documents_scores, atoms_list, formulas, gt_pos_doc, gt_neg_doc, query_debug_id = query_debug_id, topk=args.topk, debug=debug) 

            if args.is_decompose:
                returned_docids = results[qid]['new_doc_idx']
                returned_scores = results[qid]['new_scores']
            else:
                returned_docids = results[qid]['orig_doc_idx']
                returned_scores = results[qid]['orig_scores']

            for rank, (docid, score) in enumerate(zip(returned_docids, returned_scores), start=1):
                f.write("%s\t%s\t%d\t%f\n" % (qid, docid, rank, score))

            if debug:            
                results['question'] = questions[qid]['question']
                results['query_atoms'] = questions[qid]['atoms']
                results['positive_docs'] = questions[qid]['positive_ctxs']
                results['negative_docs'] = questions[qid]['negative_ctxs']
                debug_results.append(results)

        if debug:
            #debug_results = map_query_to_data(qids, queries, debug_results)
            #print(f"Processed query {qid} with {len(gt_pos_doc)} positive and {len(gt_neg_doc)} negative documents.")

            print(json.dumps(debug_results, cls=NumpyEncoder, indent=4))
        
        print(f"Total processed qid's={i}")

    if args.sqo_stats_path is not None and _sqo_stats_log:
        os.makedirs(os.path.dirname(args.sqo_stats_path) or ".", exist_ok=True)
        with open(args.sqo_stats_path, "w") as f:
            json.dump(_sqo_stats_log, f, indent=2)
        n = len(_sqo_stats_log)
        mean_step = sum(s["stopped_at_step"] for s in _sqo_stats_log) / n
        frac_early = sum(1 for s in _sqo_stats_log if s["early_stopped"]) / n
        print(f"[SQO stats] n={n}  mean_stopped_at_step={mean_step:.1f}  frac_early_stopped={frac_early:.3f}")

    print("Done.")