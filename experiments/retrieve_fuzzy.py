### RETRIEVE_FUZZY.PY

import sys
import os
import ssl
# NOTE: the original internal development environment required routing
# through a local network proxy at module load time; that call has been
# removed here since it is machine-specific and would break network access
# (model/dataset downloads) on any other machine. If your environment needs
# a proxy, set the standard http_proxy/https_proxy environment variables
# before running this script.

from tqdm import tqdm
import argparse
import json
import time
import numpy as np
import faiss
import torch
#from dataset import get_boolquestions_evalset, get_fuzzy_evalset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from operator import itemgetter
from fuzzy_logic import FuzzyLogic, NumpyEncoder, create_geometric_logic_vector
from objective_hnsw import ObjectiveHNSW
from score_dump import ScoreDumper
import pickle
import openvino as ov


# ── Initialisation ──────────────────────────────────────────────────────────
if torch.xpu.is_available():
    device = torch.device("xpu")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from sentence_transformers import SentenceTransformer

# modelNomic = SentenceTransformer("nomic-ai/nomic-embed-text-v2-moe", trust_remote_code=True)
# test_embedding = modelNomic.encode("test")
# print(f"Dimensions: {len(test_embedding)}")



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, 
        default='logic_demo\\QUEST\\documents.jsonl', 
        help="Path to the embedding files of queries and documents.")
    
    parser.add_argument("--queries_path", type=str, 
        
        default='logic_demo\\QUEST\\val_corpus_no_neg_fixed.jsonl',        
        #default='logic_demo\\QUEST\\test_corpus_no_neg_fixed.jsonl',        
        help="Path to the queries file.")
    

    parser.add_argument("--faiss_data_path", type=str, 
                
        default='logic_demo\\QUEST\\faiss_quest_bge',
        #default='logic_demo\\QUEST\\faiss_quest_nomic-nomic-v2',
        #default='logic_demo\\QUEST\\faiss_quest_zeroshot-e5',
        #default='logic_demo\\QUEST\\faiss_quest_logicol-e5',

        help="Path to the faiss embedding files of queries and documents.")
    
    parser.add_argument("--saved_queries_path", 
        default='logic_demo\\QUEST\\faiss_quest_bge\\', 
        #default='logic_demo\\QUEST\\faiss_quest_nomic-nomic-v2',
        #default='logic_demo\\QUEST\\faiss_quest_zeroshot-e5',
        #default='logic_demo\\QUEST\\faiss_quest_logicol-e5',
        help="saved_queries_path")
    
    parser.add_argument('--query_encoder', type=str, 
        default='logic_demo\\models--BAAI--bge-large-en-v1.5\\',       
        #default='logic_demo\\models--nomic-embed-text-v2-moe\\',
        #default='logic_demo\\models--intfloat_e5-base-v2\\',

        help="Name or path of the query encoder model.")
    
    parser.add_argument(
        "--output_path", type=str, default='debug_out/retrieval/',
        help="Path to save the retrieval results.")

    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size for retrieval.")
    parser.add_argument(
        "--topk", type=int, default=100,
        help="Number of documents to retrieve per query.")
    parser.add_argument(
        "--topk_max", type=int, default=1000,
        help="Number of documents to retrieve per query.")
    parser.add_argument(
        "--index_batch_size", type=int, default=1000,
        help="Batch size for building the index.")
    parser.add_argument(
        "--rebuild_index", action="store_true",
        help="Whether to rebuild the index.")
    parser.add_argument("--is_faiss", type=int, default=1, help="is faiss db or in memory search")
    parser.add_argument("--is_debug", type=int, default=0, help="is debug")
    parser.add_argument("--is_queries", type=int, default=0, help="is data in queries")
    parser.add_argument("--is_decompose", type=int, default=1, help="decompose to boolean queries or not")
    parser.add_argument("--name", type=str, default='my_atoms', help="special name of run")
    parser.add_argument("--is_opt_onsphere", type=int, default=0, help="Activate custom QueryVector On-Sphere Optimization to fit Logic-Formulas ")
    parser.add_argument("--gpu", default='0', help="gpu id")
    parser.add_argument("--is_anns_atoms", type=int, default='0', help="is anns atoms")
    parser.add_argument("--is_hnsw", type=int, default='0', help="is hnsw index")
    parser.add_argument("--is_L2_index", type=int, default='0', help="is L2 index")
    parser.add_argument("--is_convert_db", type=int, default='0', help="is cpnvert db")
    parser.add_argument("--convert_db_path", default='logic_demo/QUEST/faiss_quest_bge_l2', help="is convert db")
    
    parser.add_argument("--is_geometric", type=int, default=0, help="Activate Geometric Vector Logic")
    parser.add_argument("--fuzzy_type", type=str, default="NSFL", help="NSFL / NSFL-DEBUG / GEO")
    parser.add_argument("--augment_type", type=str, default="QUEST", help="QUEST (that are also...) / SIMPLE (only 'AND') / OCTOBER")
    parser.add_argument('--dump_scores_path', type=str, default=None,
                        help="If set, dump per-candidate scores to this TSV.")
    parser.add_argument('--sqo_stats_path', type=str, default=None,
                        help="If set (and is_opt_onsphere=1), dump per-query SQO "
                             "convergence stats (stopped_at_step, early_stopped, "
                             "total_time) to this JSON file.")
    parser.add_argument('--sqo_steps', type=int, default=100,
                        help="SQO optimize_on_sphere: max gradient steps.")
    parser.add_argument('--sqo_lr', type=float, default=0.2,
                        help="SQO optimize_on_sphere: learning rate (alpha).")
    parser.add_argument('--sqo_patience', type=int, default=5,
                        help="SQO optimize_on_sphere: early-stopping patience "
                             "(matches the value hardcoded at the call site "
                             "before this became a CLI arg).")
    parser.add_argument('--sqo_min_step_before_stop', type=int, default=None,
                        help="SQO optimize_on_sphere: minimum step before early "
                             "stopping is allowed to fire (the 'step > steps//3' "
                             "floor). Default None reproduces steps//3 exactly; "
                             "pass 0 to let patience alone decide (EXP 2b).")
    parser.add_argument('--sqo_n_restarts', type=int, default=1,
                        help="SQO multi-start (EXP 3): number of restarts per "
                             "query, keeping the one with the highest final "
                             "objective value. 1 = current single-start behavior "
                             "(question_embed[0] init), unchanged. >1 uses "
                             "question_embed[0] for the first restart and random "
                             "sphere inits for the rest.")
    parser.add_argument('--sig_gate', type=int, default=0,
                        help="Significance-gated conjunctive correction: scale "
                             "the AND2/AND3 t-norm correction by its "
                             "significance relative to the per-query pool noise "
                             "floor (std of the deficit across the candidate "
                             "pool). 0 (default) = existing unconditional "
                             "correction, bit-identical to before this flag "
                             "existed. 1 = gate on. Only affects 'atom0 & atom1' "
                             "and 'atom0 & atom1 & atom2'; all other templates "
                             "are untouched regardless of this flag.")

    args = parser.parse_args()
    return args




class OpenVINOSentenceEncoder:
    """
    Drop-in replacement for SentenceTransformer that runs on Intel GPU (XPU)
    via OpenVINO using optimum-intel. Falls back to SentenceTransformer on
    other devices, or if the OpenVINO export fails.
    """

    def __init__(self, model_name_or_path: str, device: torch.device,
                 model_kwargs: dict = None, trust_remote_code: bool = False):
        self.model_name       = model_name_or_path
        self.device           = device
        self.trust_remote_code = trust_remote_code
        self.use_openvino     = (device == torch.device("xpu"))
        self._use_optimum     = False

        # print(f"\n[OpenVINOSentenceEncoder] Initialising '{model_name_or_path}'.  Requested torch device : {device}.  Use OpenVINO path      : {self.use_openvino}")

        if self.use_openvino:
            self._init_openvino(model_name_or_path)
        else:
            # print(f"  Loading via SentenceTransformer on {device}")
            self.model = SentenceTransformer(
                model_name_or_path,
                device=str(device),
                model_kwargs=model_kwargs or {},
                trust_remote_code=trust_remote_code,
            )
            self.max_seq_length = self.model.max_seq_length
            # print(f"  [OK] SentenceTransformer loaded on {device}")

    # ------------------------------------------------------------------ #
    #  String-like interface (so "Qwen" in encoder works)
    # ------------------------------------------------------------------ #
    def __contains__(self, item: str) -> bool:
        return item in self.model_name

    def __str__(self) -> str:
        return self.model_name

    def __repr__(self) -> str:
        device_str = self.ov_device if self.use_openvino else str(self.device)
        return f"OpenVINOSentenceEncoder(model='{self.model_name}', device='{device_str}')"

    # ------------------------------------------------------------------ #
    #  OpenVINO device discovery
    # ------------------------------------------------------------------ #
    @staticmethod
    def _pick_ov_device(core) -> str:
        available = core.available_devices
        str = f"\n  [OpenVINO] Available devices : {available}"
        for dev in available:
            try:
                full_name = core.get_property(dev, "FULL_DEVICE_NAME")
                str += f"    {dev:12s} -> {full_name}"
            except Exception:
                str += f"    {dev:12s} -> (name unavailable)"

        gpu_devices = [d for d in available if d.startswith("GPU")]
        if gpu_devices:
            chosen = gpu_devices[0]
            str += f"\n  [OpenVINO] Selecting GPU device : {chosen}"
            print(str)
            return chosen

        print("  [OpenVINO] No GPU found — falling back to CPU")
        return "CPU"


    # ------------------------------------------------------------------ #
    #  OpenVINO init via optimum-intel
    # ------------------------------------------------------------------ #
    def _init_openvino(self, model_name_or_path: str):
        import openvino as ov
        core = ov.Core()
        self.ov_device = self._pick_ov_device(core)

        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "ov_sentence_transformers")
        safe_name = model_name_or_path.replace("/", "_").replace("\\", "_")
        ov_model_dir = os.path.join(cache_dir, safe_name)

        try:
            from optimum.intel import OVModelForFeatureExtraction
            from optimum.exporters.onnx import OnnxConfig
            from optimum.exporters.onnx.model_configs import BertOnnxConfig
            from transformers import AutoTokenizer, AutoConfig
            from collections import OrderedDict

            # Custom export config for nomic_bert and other unsupported architectures.
            # We subclass BertOnnxConfig which covers any BERT-like model with
            # input_ids + attention_mask inputs and last_hidden_state output.
            class NomicBertOnnxConfig(BertOnnxConfig):
                @property
                def inputs(self) -> dict[str, dict[int, str]]:
                    return OrderedDict([
                        ("input_ids",      {0: "batch_size", 1: "sequence_length"}),
                        ("attention_mask", {0: "batch_size", 1: "sequence_length"}),
                    ])

                @property
                def outputs(self) -> dict[str, dict[int, str]]:
                    return OrderedDict([
                        ("last_hidden_state", {0: "batch_size", 1: "sequence_length"}),
                    ])

            config = AutoConfig.from_pretrained(
                model_name_or_path, trust_remote_code=self.trust_remote_code
            )
            custom_export_configs = {"model": NomicBertOnnxConfig(config)}

            if os.path.exists(ov_model_dir):
                print(f"  [OpenVINO] Loading cached OV model from {ov_model_dir}")
                self.ov_model = OVModelForFeatureExtraction.from_pretrained(
                    ov_model_dir,
                    device=self.ov_device,
                    trust_remote_code=self.trust_remote_code,
                )
            else:
                print(f"  [OpenVINO] Exporting via optimum-intel (one-time, may take a few minutes) ...")
                os.makedirs(ov_model_dir, exist_ok=True)
                self.ov_model = OVModelForFeatureExtraction.from_pretrained(
                    model_name_or_path,
                    export=True,
                    device=self.ov_device,
                    trust_remote_code=self.trust_remote_code,
                    custom_export_configs=custom_export_configs,
                )
                self.ov_model.save_pretrained(ov_model_dir)
                print(f"  [OpenVINO] OV model saved to {ov_model_dir}")

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path, trust_remote_code=self.trust_remote_code
            )
            self.max_seq_length = getattr(self.ov_model.config, "max_position_embeddings", 512)
            self._use_optimum = True

            try:
                runtime_device = self.ov_model.request.get_property("EXECUTION_DEVICES")
                # print(f"  [OpenVINO] Runtime execution device(s): {runtime_device}")
            except Exception:
                pass

            # print(f"  [OpenVINO] Init complete via optimum-intel on: {self.ov_device}\n")

        except Exception as e:
            # print(f"\n  [OpenVINO] optimum-intel export/load failed: {e}")
            # print(f"  [OpenVINO] Falling back to SentenceTransformer on CPU\n")
            self._use_optimum = False
            self.ov_device = "CPU (fallback)"
            self.model = SentenceTransformer(
                model_name_or_path, device="cpu",
                trust_remote_code=self.trust_remote_code,
            )
            self.max_seq_length = self.model.max_seq_length


            
    # ------------------------------------------------------------------ #
    #  Tokenisation
    # ------------------------------------------------------------------ #
    def _tokenize(self, texts: list[str]) -> dict:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="np",
        )
        return {
            "input_ids":      encoded["input_ids"].astype(np.int64),
            "attention_mask": encoded["attention_mask"].astype(np.int64),
        }

    # ------------------------------------------------------------------ #
    #  Public API — matches SentenceTransformer.encode()
    # ------------------------------------------------------------------ #
    def encode(self, sentences: list[str], batch_size: int = 32, **kwargs) -> np.ndarray:
        # Non-XPU: plain SentenceTransformer on CUDA/CPU
        if not self.use_openvino:
            return self.model.encode(sentences, batch_size=batch_size, **kwargs)

        # XPU but optimum export failed: SentenceTransformer CPU fallback
        if not self._use_optimum:
            return self.model.encode(sentences, batch_size=batch_size, **kwargs)

        # XPU + optimum: run on Intel GPU via OpenVINO
        all_embeddings = []
        for i, start in enumerate(range(0, len(sentences), batch_size)):
            batch = sentences[start: start + batch_size]
            inputs = self._tokenize(batch)

            if i == 0:
                print(f"  [OpenVINO] Running inference on {self.ov_device} "
                      f"(batch {len(batch)}/{len(sentences)} sentences)")

            torch_inputs = {k: torch.tensor(v) for k, v in inputs.items()}
            outputs = self.ov_model(**torch_inputs)

            token_embeddings = outputs.last_hidden_state.numpy()
            mask = inputs["attention_mask"][..., np.newaxis].astype(np.float32)
            pooled = (token_embeddings * mask).sum(1) / mask.sum(1).clip(min=1e-9)
            norms  = np.linalg.norm(pooled, axis=-1, keepdims=True).clip(min=1e-12)
            all_embeddings.append(pooled / norms)

        return np.vstack(all_embeddings)
    


def load_document_embeddings(documents_file, eval_data, query_encoder, is_encode_all, is_eval_data):
    doc_embeds = []
    docids = []
    documents = []
    # input_path = os.path.join(input_path, "embeddings")
    # file_names = os.listdir(input_path)
    # docids_files = sorted([x for x in file_names if x.startswith("docids.")])
    # docids = np.concatenate([np.load(os.path.join(input_path, x)) for x in docids_files], axis=0)
    # doc_embeds_files = sorted([x for x in file_names if x.startswith("doc_embeds.")])
   
    # doc_embeds = np.concatenate([np.load(os.path.join(input_path, x))
    #                              for x in tqdm(doc_embeds_files, desc="Loading document embeddings")], axis=0)
    # assert len(docids) == len(doc_embeds), f"Number of document ids and embeddings do not match: {len(docids)} != {len(doc_embeds)}"
    passage_id = 'passage_id'
    doc = 'answer'
        
    if is_eval_data:
        for d in eval_data:
            docids = docids + [pos_ctx[passage_id] for pos_ctx in d["positive_ctxs"]]
            docids = docids + [neg_ctx[passage_id] for neg_ctx in d["negative_ctxs"]]
            documents = documents + [pos_ctx[doc] for pos_ctx in d["positive_ctxs"]]
            documents = documents + [neg_ctx[doc] for neg_ctx in d["negative_ctxs"]]
    else:
        docids = []
        with open(documents_file) as d:
            for i, line in enumerate(d):
                documents_line = json.loads(line)
                #documents_line['id'] = str(i)
                #documents[documents_line['title']] = documents_line['text']
                documents.append(documents_line['text'])
                docids.append(str(i))

    if is_encode_all:
        doc_embeds = query_encoder.encode(documents)
    return docids, doc_embeds

def encode_query(query_encoder, query, prompt_name):
    a_embeds = query_encoder.encode([query])
    a_embeds = np.array(a_embeds)
    a_embeds = np.concatenate(a_embeds, axis=0)
    return a_embeds


def encode_queries_embedding(query_encoder, questions, qids, eval_qid2query, eval_qid2docids, qid2neg_docids, eval_qid2formulas, eval_qid2qtypes):
    prompt_name = "query" if "Qwen" in query_encoder else None
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

    printcounter=0
    printcounter2=0

    for queryid in tqdm(qids, desc="Encoding queries"):
        eval_qid2atoms =  eval_qid2query[queryid]
        atom_embeds = []
        atom_qtype = eval_qid2qtypes[queryid]
        qformulas = eval_qid2formulas[queryid]
        question_embed = query_encoder.encode([questions[queryid]])# normalize_embeddings=True)
        for atom in eval_qid2atoms.keys():
            a_embeds = encode_query(query_encoder, eval_qid2atoms[atom], prompt_name)
            atom_embeds.append(a_embeds)       
        formula1 = qformulas['formula1']

        if (args.augment_type.upper() == "OCTOBER"):
            if formula1 ==  "atom0 & atom1 & !atom2":
                atomAB = eval_qid2atoms['atom0'] + ' that are also ' + eval_qid2atoms['atom1']
                ab_embeds = encode_query(query_encoder, atomAB, prompt_name)
                atom_embeds.append(ab_embeds)   

                atomABC = eval_qid2atoms['atom0'] + ' that are also both ' + eval_qid2atoms['atom1'] + ' and ' + eval_qid2atoms['atom2']
                abc_embeds = encode_query(query_encoder, atomABC, prompt_name)
                atom_embeds.append(abc_embeds)    

            elif formula1 ==  formula1=="atom0 & atom1 & atom2":
                atomAB = eval_qid2atoms['atom0'] + ' that are also ' + eval_qid2atoms['atom1']
                ab_embeds = encode_query(query_encoder, atomAB, prompt_name)
                atom_embeds.append(ab_embeds)      
                
                atomAC = eval_qid2atoms['atom0'] + ' that are also ' + eval_qid2atoms['atom2']
                ac_embeds = encode_query(query_encoder, atomAC, prompt_name)
                atom_embeds.append(ac_embeds)      
                
                atomBC = eval_qid2atoms['atom1'] + ' that are also ' + eval_qid2atoms['atom2']
                bc_embeds = encode_query(query_encoder, atomBC, prompt_name)
                atom_embeds.append(bc_embeds)
                    
            elif formula1== "atom0 | atom1 | atom2" or formula1=="atom0 | atom1 & !atom2" :
                atomAB = eval_qid2atoms['atom0'] + ' or ' + eval_qid2atoms['atom1']
                ab_embeds = encode_query(query_encoder, atomAB, prompt_name)
                atom_embeds.append(ab_embeds)      

                atomAC = eval_qid2atoms['atom0'] + ' or ' + eval_qid2atoms['atom2']
                ac_embeds = encode_query(query_encoder, atomAC, prompt_name)
                atom_embeds.append(ac_embeds)      

                atomBC = eval_qid2atoms['atom1'] + ' or ' + eval_qid2atoms['atom2']
                bc_embeds = encode_query(query_encoder, atomBC, prompt_name)
                atom_embeds.append(bc_embeds)  

            elif formula1 ==  "atom0 & !atom1":
                atomAB = eval_qid2atoms['atom0'] + ' that are also ' + eval_qid2atoms['atom1']
                ab_embeds = encode_query(query_encoder, atomAB, prompt_name)
                atom_embeds.append(ab_embeds)   

        else:
            
            # Augment queries by atoms combinations
            if formula1 == "atom0 & !atom1" or formula1 == "atom0 & atom1" or formula1 == "atom0 | atom1":
                atomAB = eval_qid2atoms['atom0'] + separator1 + eval_qid2atoms['atom1']
                ab_embeds = encode_query(query_encoder, atomAB, prompt_name)
                atom_embeds.append(ab_embeds)      
                # if printcounter<3:
                #     print(f"queryid: {queryid}, formula1: {formula1}, atomAB: {atomAB}")
                #     printcounter = printcounter+1
            
            elif formula1 == "atom0 & atom1 & !atom2" or formula1 == "atom0 & atom1 & atom2" or formula1 == "atom0 | atom1 | atom2" or formula1=="atom0 | atom1 & !atom2":
                atomAB = eval_qid2atoms['atom0'] + separator1_1 + eval_qid2atoms['atom1']
                ab_embeds = encode_query(query_encoder, atomAB, prompt_name)
                atom_embeds.append(ab_embeds)      

                atomAC = eval_qid2atoms['atom0'] + separator1_1 + eval_qid2atoms['atom2']
                ac_embeds = encode_query(query_encoder, atomAC, prompt_name)
                atom_embeds.append(ac_embeds)      

                atomBC = eval_qid2atoms['atom1'] + separator1_1 + eval_qid2atoms['atom2']
                bc_embeds = encode_query(query_encoder, atomBC, prompt_name)
                atom_embeds.append(bc_embeds)  

                atomABC = eval_qid2atoms['atom0'] + separator1_2 + eval_qid2atoms['atom1'] + separator2 + eval_qid2atoms['atom2']
                abc_embeds = encode_query(query_encoder, atomABC, prompt_name)
                atom_embeds.append(abc_embeds) 

                # if printcounter2<3:
                #     print(f"queryid: {queryid}, formula1: {formula1}, atomAB: {atomAB}, atomAC: {atomAC}, atomBC: {atomBC}, atomABC: {atomABC}") 
                #     printcounter2 = printcounter2+1


        qatoms_names.append(list(eval_qid2atoms.keys()))
        query_embeds.append(atom_embeds)
        qtype_index.append(atom_qtype)
        qformula_index.append(qformulas)
        gt_docs.append(eval_qid2docids[queryid])
        gt_neg_docs.append(qid2neg_docids[queryid])
        question_embeds.append(question_embed)

    return  question_embeds, query_embeds, qtype_index, qformula_index, qatoms_names, gt_docs, gt_neg_docs


def get_fuzzy_evalset_queries(    eval_data  ):
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
        qid2neg_docids[qid] = [pos_ctx[passage_id] for neg_ctx in d["negative_ctxs"]]
    return qid2question, qid2query, qid2docids, qid2neg_docids, qid2formulas, qid2qtypes


# # Older function that does not use local pre-computed queries list
# def load_all_data(queries, is_encode_all):
#     if not (args.queries_path and args.query_encoder):
#         raise ValueError("Queries path and query encoder must be provided for online query encoding.")
#     if "Qwen" in args.query_encoder:
#         model_kwargs = {"torch_dtype":torch.float16}
#     else:
#         model_kwargs = None
    
#     ###query_encoder = SentenceTransformer(args.query_encoder, device="cuda", model_kwargs=model_kwargs, trust_remote_code=True)
#     query_encoder = OpenVINOSentenceEncoder(
#             args.query_encoder,
#             device=device,
#             model_kwargs=model_kwargs,
#             trust_remote_code=True,
#         )
    

#     #eval_qid2query, eval_qid2docids = get_fuzzy_evalset(queries, return_negatives=True, fuzzy=fuzzy)
#     questions, eval_qid2query, eval_qid2docids, qid2neg_docids, eval_qid2formulas, eval_qid2qtypes = get_fuzzy_evalset_queries(queries)
#     qids = sorted(list(eval_qid2docids.keys()))    

#     docids, doc_embeds = load_document_embeddings(args.input_path, queries, query_encoder, is_encode_all, args.is_queries)
#     query_embeds = []
#     qtype_index = []
#     qformula_index = []
#     qatoms_names = []
#     gt_docs = []
#     gt_neg_docs = []
#     question_embeds = []
#     prompt_name = ''

#     for queryid in tqdm(qids, desc="Encoding queries"):
#         eval_qid2atoms =  eval_qid2query[queryid]
#         atom_embeds = []
#         atom_qtype = eval_qid2qtypes[queryid]
#         qformulas = eval_qid2formulas[queryid]
#         question_embed = query_encoder.encode([questions[queryid]], prompt_name=prompt_name)
#         for atom in eval_qid2atoms.keys():
#             a_embeds = encode_query(query_encoder, eval_qid2atoms[atom], prompt_name)
#             atom_embeds.append(a_embeds)       
#         formula1 = qformulas['formula1']
        
#         if formula1 ==  "atom0 & atom1 & !atom2":
#             atomAB = eval_qid2atoms['atom0'] + ' that are also ' + eval_qid2atoms['atom1']
#             ab_embeds = encode_query(query_encoder, atomAB, prompt_name)
#             atom_embeds.append(ab_embeds)   

#             atomABC = eval_qid2atoms['atom0'] + ' that are also both ' + eval_qid2atoms['atom1'] + ' and ' + eval_qid2atoms['atom2']
#             abc_embeds = encode_query(query_encoder, atomABC, prompt_name)
#             atom_embeds.append(abc_embeds)    

#         elif  formula1=="atom0 & atom1 & atom2":
#             atomAB = eval_qid2atoms['atom0'] + ' that are also ' + eval_qid2atoms['atom1']
#             ab_embeds = encode_query(query_encoder, atomAB, prompt_name)
#             atom_embeds.append(ab_embeds)      
            
#             atomAC = eval_qid2atoms['atom0'] + ' that are also ' + eval_qid2atoms['atom2']
#             ac_embeds = encode_query(query_encoder, atomAC, prompt_name)
#             atom_embeds.append(ac_embeds)      
            
#             atomBC = eval_qid2atoms['atom1'] + ' that are also ' + eval_qid2atoms['atom2']
#             bc_embeds = encode_query(query_encoder, atomBC, prompt_name)
#             atom_embeds.append(bc_embeds)
                 
#         elif formula1== "atom0 | atom1 | atom2" or formula1=="atom0 | atom1 & !atom2" :
#             atomAB = eval_qid2atoms['atom0'] + ' or ' + eval_qid2atoms['atom1']
#             ab_embeds = encode_query(query_encoder, atomAB, prompt_name)
#             atom_embeds.append(ab_embeds)      

#             atomAC = eval_qid2atoms['atom0'] + ' or ' + eval_qid2atoms['atom2']
#             ac_embeds = encode_query(query_encoder, atomAC, prompt_name)
#             atom_embeds.append(ac_embeds)      

#             atomBC = eval_qid2atoms['atom1'] + ' or ' + eval_qid2atoms['atom2']
#             bc_embeds = encode_query(query_encoder, atomBC, prompt_name)
#             atom_embeds.append(bc_embeds)  

#         elif formula1 ==  "atom0 & !atom1" or formula1 ==  "atom0 & atom1":
#             atomAB = eval_qid2atoms['atom0'] + ' that are also ' + eval_qid2atoms['atom1']
#             ab_embeds = encode_query(query_encoder, atomAB, prompt_name)
#             atom_embeds.append(ab_embeds)      

#         qatoms_names.append(list(eval_qid2atoms.keys()))
#         query_embeds.append(atom_embeds)
#         qtype_index.append(atom_qtype)
#         qformula_index.append(qformulas)
#         gt_docs.append(eval_qid2docids[queryid])
#         gt_neg_docs.append(qid2neg_docids[queryid])
#         question_embeds.append(question_embed)

#     return  question_embeds, query_embeds, qtype_index, qformula_index, qatoms_names, gt_docs, gt_neg_docs


def load_queries_embeddings(saved_queries_path, query_encoder, questions, qids, eval_qid2query, eval_qid2docids, qid2neg_docids, eval_qid2formulas, eval_qid2qtypes, queries_path):
    #load queries
    # if 'test_' in queries_path:
    #     queries_saved_file = os.path.join(saved_queries_path, "queries_test.pkl") if saved_queries_path is not None else None
    # else:
    #     queries_saved_file = os.path.join(saved_queries_path, "queries.pkl") if saved_queries_path is not None else None
    
    if 'val_corpus_no_neg_fixed' in queries_path:
        if args.augment_type.upper() == "OCTOBER":
            queries_saved_file = os.path.join(saved_queries_path, "quest_val_queries(augment-october).pkl") if saved_queries_path is not None else None
        elif args.augment_type.upper() == "QUEST":
            queries_saved_file = os.path.join(saved_queries_path, "quest_val_queries(augment-quest).pkl") if saved_queries_path is not None else None
        else:
            queries_saved_file = os.path.join(saved_queries_path, "quest_val_queries(augment-simple).pkl") if saved_queries_path is not None else None

    elif 'test_corpus_no_neg_fixed' in queries_path:
        if args.augment_type.upper() == "OCTOBER":
            queries_saved_file = os.path.join(saved_queries_path, "quest_test_queries(augment-october).pkl") if saved_queries_path is not None else None
            
        elif args.augment_type.upper() == "QUEST":
            queries_saved_file = os.path.join(saved_queries_path, "quest_test_queries(augment-quest).pkl") if saved_queries_path is not None else None
        else:
            queries_saved_file = os.path.join(saved_queries_path, "quest_test_queries(augment-simple).pkl") if saved_queries_path is not None else None

    elif 'train_corpus_no_neg_fixed' in queries_path:
        if args.augment_type.upper() == "OCTOBER":
            queries_saved_file = os.path.join(saved_queries_path, "quest_train_queries(augment-october).pkl") if saved_queries_path is not None else None
            
        elif args.augment_type.upper() == "QUEST":
            queries_saved_file = os.path.join(saved_queries_path, "quest_train_queries(augment-quest).pkl") if saved_queries_path is not None else None
        else:
            queries_saved_file = os.path.join(saved_queries_path, "quest_train_queries(augment-simple).pkl") if saved_queries_path is not None else None

    elif 'debug_' in queries_path:
        if args.augment_type.upper() == "QUEST":
            queries_saved_file = os.path.join(saved_queries_path, "quest_debug(augment-quest).pkl") if saved_queries_path is not None else None
        else:
            queries_saved_file = os.path.join(saved_queries_path, "quest_debug(augment-simple).pkl") if saved_queries_path is not None else None
    elif 'filtered_100_from_all_coco2017_VAL' in queries_path:
        if args.augment_type.upper() == "QUEST":
            queries_saved_file = os.path.join(saved_queries_path, "filtered_100_from_all_coco2017_VAL(augment-simple).pkl") if saved_queries_path is not None else None
        else:
            queries_saved_file = os.path.join(saved_queries_path, "filtered_100_from_all_coco2017_VAL(augment-simple).pkl") if saved_queries_path is not None else None
    else:
        queries_saved_file = None
    
    if queries_saved_file is not None and os.path.exists(queries_saved_file):                  
        #load from saved fom pickle file
        with open(queries_saved_file, 'rb') as f:
           queries = pickle.load(f)
        question_embeds, query_embeds, qtype_index, qformula_index, qatoms_names, gt_docs, gt_neg_docs = queries['question_embeds'], queries['query_embeds'], queries['qtype_index'], queries['qformula_index'], queries['qatoms_names'], queries['gt_docs'], queries['gt_neg_docs']
    else:
        #encode queries
        question_embeds, query_embeds, qtype_index, qformula_index, qatoms_names, gt_docs, gt_neg_docs = encode_queries_embedding(
            query_encoder, questions, qids, eval_qid2query, eval_qid2docids, qid2neg_docids, eval_qid2formulas, eval_qid2qtypes)
        if saved_queries_path is not None:
            queries = {'question_embeds': question_embeds, 'query_embeds': query_embeds, 'qtype_index': qtype_index, 'qformula_index': qformula_index, 'qatoms_names': qatoms_names, 'gt_docs': gt_docs, 'gt_neg_docs': gt_neg_docs, 'qids': qids}
            os.makedirs(saved_queries_path, exist_ok=True)
            with open(queries_saved_file, 'wb') as f:
                pickle.dump(queries, f, protocol=pickle.HIGHEST_PROTOCOL)        

    return  question_embeds, query_embeds, qtype_index, qformula_index, qatoms_names, gt_docs, gt_neg_docs

def get_fuzzy_evalset_queries(
    eval_data
):
    
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
        qid2neg_docids[qid] = [pos_ctx[passage_id] for neg_ctx in d["negative_ctxs"]]
    return qid2question, qid2query, qid2docids, qid2neg_docids, qid2formulas, qid2qtypes

def load_all_data(queries, is_encode_documents, saved_queries_path):
    #fuzzy = True
    if not (args.queries_path and args.query_encoder):
        raise ValueError("Queries path and query encoder must be provided for online query encoding.")
    if "Qwen" in args.query_encoder:
        model_kwargs = {"torch_dtype":torch.float16}
    else:
        model_kwargs = None
    
    # query_encoder = SentenceTransformer(args.query_encoder, trust_remote_code=True)
    query_encoder = OpenVINOSentenceEncoder(args.query_encoder, device=device, model_kwargs=model_kwargs, trust_remote_code=True)


    #eval_qid2query, eval_qid2docids = get_fuzzy_evalset(queries, return_negatives=True, fuzzy=fuzzy)
    questions, eval_qid2query, eval_qid2docids, qid2neg_docids, eval_qid2formulas, eval_qid2qtypes = get_fuzzy_evalset_queries(queries)
    qids = sorted(list(eval_qid2docids.keys()))    

    #load documents
    docids, doc_embeds = load_document_embeddings(args.input_path, queries, query_encoder, is_encode_documents, args.is_queries)
    #load or encode queries
    question_embeds, query_embeds, qtype_index, qformula_index, qatoms_names, gt_docs, gt_neg_docs = load_queries_embeddings(
        saved_queries_path, query_encoder, questions, qids, eval_qid2query, eval_qid2docids, qid2neg_docids, eval_qid2formulas, eval_qid2qtypes, args.queries_path)
      
    return qids, question_embeds, query_embeds, docids, doc_embeds, qtype_index, qformula_index, qatoms_names, gt_docs, gt_neg_docs


def dump_fused_columns(formula1, n_cols):
    """
    Which columns of the (n_candidates, n_cols) atomic-scores matrix hold
    S_AB / S_ABC for the score dumper, given the query's formula string and
    the actual number of atom-embedding columns produced for it.

    Returns (ab_col, abc_col), each a column index or None if that fused
    score was not computed by the pipeline for this formula (None is written
    as NaN by ScoreDumper - never recomputed here).

    Traced directly from encode_queries_embedding()'s atom_embeds
    construction and evaluate_fuzzy_nsfl_clean()'s own indexing:
      - binary (atom0 & atom1 / atom0 & !atom1 / atom0 | atom1):
        columns = [a, b, ab]                              -> S_AB=col2, S_ABC=None
      - pure ternary (atom0 & atom1 & atom2 / atom0 | atom1 | atom2):
        columns = [a, b, c, ab, ac, bc, abc]               -> S_AB=None (pairwise
        ab/ac/bc are not part of the reported schema), S_ABC=col6
      - mixed (atom0 & atom1 & !atom2 / atom0 | atom1 & !atom2):
        columns = [a, b, c, ab, ac, bc, abc]               -> S_AB=col3, S_ABC=col6
    Do not extend this table by guessing - trace the actual embedding
    construction code for any new formula string, since --augment_type=OCTOBER
    produces a different column layout for some of these formulas.
    """
    BINARY = {"atom0 & atom1", "atom0 & !atom1", "atom0 | atom1"}
    PURE_TERNARY = {"atom0 & atom1 & atom2", "atom0 | atom1 | atom2"}
    MIXED_TERNARY = {"atom0 & atom1 & !atom2", "atom0 | atom1 & !atom2"}

    if formula1 in BINARY:
        return (2 if n_cols >= 3 else None), None
    if formula1 in PURE_TERNARY:
        return None, (6 if n_cols >= 7 else None)
    if formula1 in MIXED_TERNARY:
        return (3 if n_cols >= 4 else None), (6 if n_cols >= 7 else None)
    # "atom0" (no fused columns) or an unrecognized formula string: report
    # atoms only, leave fused columns as NaN rather than guess an index.
    return None, None


# Formulas where evaluate_fuzzy_nsfl_clean substitutes the monolithic score
# in place of the computed AB/ABC column (see its "atom0 & atom1" / "atom0 &
# atom1 & atom2" branches - the computed column is commented out in favor of
# monolithic_score). For every other formula, NSFL reads the computed column
# directly.
PURE_AND_FORMULAS = {"atom0 & atom1", "atom0 & atom1 & atom2"}


def dump_fused_used_array(formula1, ab_col, abc_col, query_documents_scores, monolithic_scores_row):
    """
    The array (n_candidates,) of the exact scalar value the NSFL operator
    itself consumed as its fused score for this query's formula - see
    ScoreDumper.add_query's fused_used_scores docstring for the pure-AND
    caveat. For mixed templates (both S_AB and S_ABC populated), NSFL
    actually consumes both together in a compound expression rather than a
    single scalar; S_ABC (the fuller fusion) is reported here as a
    documented simplification, matching validate_conjectures.py's own
    fused_of() preference order.
    """
    if formula1 in PURE_AND_FORMULAS:
        return monolithic_scores_row
    if abc_col is not None:
        return query_documents_scores[:, abc_col]
    if ab_col is not None:
        return query_documents_scores[:, ab_col]
    return None


def get_cosine_similarity(q_embeds, doc_embeds):
    # scores = []
    q_embeds = np.array(q_embeds)
    shapes = q_embeds.shape
    a = q_embeds.reshape(shapes[0]*shapes[1],-1)
    b = doc_embeds.transpose()
    res = a @ b    
    scores = res.reshape(shapes[0],shapes[1],-1).transpose([0,2,1])
    return scores

def get_cosine_similarity_new_bad(q_embeds, doc_embeds):
    query_atoms_embeds = np.array(q_embeds)  # (A, dim)
    doc_embeds2 = np.array(doc_embeds)                  # (D, dim)
    return (query_atoms_embeds @ doc_embeds.T).T 



def get_L2_similarity(q_embeds, doc_embeds):
    q_embeds = np.array(q_embeds)    
    doc_embeds = np.float128(doc_embeds)
    q_embeds = np.float128(q_embeds)
    diff = q_embeds[:, np.newaxis, :] - doc_embeds[np.newaxis, :, :]
    d = np.linalg.norm(diff, axis=2)
    #scores  = 1-(d/2)   
    scores  = 1-(np.power(d,2)/2)   
    return scores


def index_documents(input_path, docids, doc_embeds, index_batch_size=10000, rebuild_index=False):
    index_path = os.path.join(input_path, "index")
    os.makedirs(index_path, exist_ok=True)
    index_file = os.path.join(index_path, "index")
    index_meta_file = os.path.join(index_path, "index_meta.json")
    if os.path.exists(index_file) and os.path.exists(index_meta_file) and not rebuild_index:
        print(f"Loading index from {index_path}.")
        index = faiss.read_index(index_file)
        with open(index_meta_file) as f:
            docids = json.load(f)["docids"]
        print(f"Index loaded. ntotal:", index.ntotal)
    else:
        print("Building index from scratch.")
        #docids, doc_embeds = load_document_embeddings(input_path)
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
    print("Python exe:", sys.executable)
    print("Python path:", sys.path)
    
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    debug = args.is_debug
    is_faiss = args.is_faiss
    query_debug = False
    query_debug_id = '1'


    fl = FuzzyLogic(pylogics = False, type=args.fuzzy_type.upper(), sig_gate=bool(args.sig_gate))

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
    print (f"is_encode_all = {is_encode_all}")
    qids, question_embeds, query_embeds, docids, doc_embeds, ops_index, qformula_index, qatoms_names, gt_pos_docs, gt_neg_docs = \
        load_all_data(queries_data, is_encode_all, args.saved_queries_path)

    if is_faiss:
        doc_index, docids = index_documents(args.faiss_data_path, docids, doc_embeds, args.index_batch_size, args.rebuild_index)
        # get all vectors from faiss
        doc_embeds = doc_index.reconstruct_n(0, doc_index.ntotal)
        topk_max = min(args.topk_max, doc_index.ntotal)

        if args.is_convert_db and (args.is_L2_index or args.is_hnsw):
            #create HSNW index instead of SCAN
            if args.is_hnsw:
                M = 32      
                if args.is_L2_index:                 
                    index = faiss.IndexHNSWFlat(doc_embeds.shape[1] , M)
                else:
                    index = faiss.IndexHNSWFlat(doc_embeds.shape[1] , M, faiss.METRIC_INNER_PRODUCT)            
                # 3. Configure Accuracy/Speed Trade-offs
                # efConstruction: Depth of exploration during index building (higher = beencodetter graph)
                index.hnsw.efConstruction = 200  
                # efSearch: Depth of exploration during search (higher = more accurate)
                # This can be changed at any time before a search
                index.hnsw.efSearch = 64          
            else:
                #SCAN L2 index
                index = faiss.IndexFlatL2(doc_embeds.shape[1])           
            
            # 4. Add vectors (HNSW does NOT require training)            
            doc_embeds_batches = np.array_split(doc_embeds, max(1, len(doc_embeds) // args.index_batch_size), axis=0)
            for i, doc_embeds_batch in enumerate(tqdm(doc_embeds_batches, desc="Building index")):
                # ## faiss.normalize_L2(doc_embeds_batch)
                index.add(doc_embeds_batch)
            doc_index = index            
            input_path = args.convert_db_path
            index_path = os.path.join(input_path, "index")    
            os.makedirs(index_path, exist_ok=True)
            index_file = os.path.join(index_path, "index")
            index_meta_file = os.path.join(index_path, "index_meta.json")
            faiss.write_index(index, index_file)
            with open(index_meta_file, "w") as f:
                json.dump({"docids": docids}, f)

    # Check if the output path exists, if not create it
    debug_results = []
    if args.output_path is None:
        args.output_path = os.path.join(args.input_path, "retrieval")
    os.makedirs(args.output_path, exist_ok=True)
    
    if args.is_opt_onsphere:        
        ObjHNSW = ObjectiveHNSW()

    results = {}

    dumper = ScoreDumper(args.dump_scores_path) if args.dump_scores_path else None
    _sqo_stats_log = []

    # Save the results
    with open(os.path.join(args.output_path, f"results.decompose{args.is_decompose}_{args.name}.txt"), "w") as f:
        i=0
        for qid, question_embed, query_atoms_embeds, operator, atoms_list, formulas, gt_pos_doc, gt_neg_doc in zip(qids, question_embeds, query_embeds, ops_index, qatoms_names, qformula_index, gt_pos_docs, gt_neg_docs):
            i = i+1
            # # debug spcific query
            # formula1 = formulas['formula1']
            # if formula1 != "atom0 & atom1 & atom2":
            #     continue

            if args.is_geometric:
                new_vector = create_geometric_logic_vector(formulas['formula1'], question_embed, query_atoms_embeds)
                # monolitic_scores, monolitic_ids = doc_index.search(new_vector, args.topk)    
                # returned_docids = monolitic_ids[0][:args.topk]
                # returned_scores = monolitic_scores[0][:args.topk]
                # results[qid] = {'orig_doc_idx': returned_docids.tolist(), 'orig_scores': returned_scores }                
                question_embed = new_vector
            
            # elif args.is_opt_onsphere:
            if args.is_opt_onsphere:
                atoms_embeds = []
                for a in query_atoms_embeds:
                    atoms_embeds.append(torch.from_numpy(a))
                obj_fn = fl.get_formula_objective(formulas['formula1'], torch.from_numpy(question_embed), atoms_embeds)

                if args.sqo_n_restarts <= 1:
                    new_query_embed, _sqo_stats = ObjHNSW.optimize_on_sphere(
                        obj_fn, dim=question_embed.shape[1], steps=args.sqo_steps, lr=args.sqo_lr,
                        early_stopping_patience=args.sqo_patience, init_value=question_embed[0],
                        return_stats=True, min_step_before_stop=args.sqo_min_step_before_stop,
                    )
                else:
                    # EXP 3 multi-start: question_embed[0] is always one start (matches
                    # the single-start behavior exactly when it wins), the rest are
                    # random inits on the sphere. Keep the restart with the highest
                    # final_objective. Wall-clock is the sum across all restarts run
                    # for this query (sequential, not parallel).
                    _restart_results = []
                    _restart_wall_start = time.time()
                    for _r in range(args.sqo_n_restarts):
                        _init = question_embed[0] if _r == 0 else None
                        _cand, _stats = ObjHNSW.optimize_on_sphere(
                            obj_fn, dim=question_embed.shape[1], steps=args.sqo_steps, lr=args.sqo_lr,
                            early_stopping_patience=args.sqo_patience, init_value=_init,
                            return_stats=True, min_step_before_stop=args.sqo_min_step_before_stop,
                        )
                        _restart_results.append((_cand, _stats))
                    _best_cand, _sqo_stats = max(_restart_results, key=lambda cs: cs[1]["final_objective"])
                    _sqo_stats = dict(_sqo_stats)
                    _sqo_stats["n_restarts"] = args.sqo_n_restarts
                    _sqo_stats["wall_time_all_restarts"] = time.time() - _restart_wall_start
                    new_query_embed = _best_cand

                if args.sqo_stats_path is not None:
                    _sqo_stats_log.append(_sqo_stats)
                # new_query_embed = ObjHNSW.optimize_on_sphere(obj_fn, dim=question_embed.shape[1], steps=100, lr=0.1, early_stopping_patience=0, init_value=None)

                new_query_embed = np.expand_dims(new_query_embed, 0)
                monolitic_scores, monolitic_ids = doc_index.search(new_query_embed, topk_max)
                target_doc_embeds = doc_embeds[monolitic_ids].squeeze()
                if args.is_decompose:
                    ## custom 
                    
                    # query_documents_scores = get_cosine_similarity([query_atoms_embeds], target_doc_embeds)[0]
                    # query_documents_ids = monolitic_ids.transpose()
                    # results = fl.run_query_rescoring_docids(qid, monolitic_scores.transpose(), monolitic_ids.transpose(), query_documents_scores, query_documents_ids, atoms_list, formulas, gt_pos_doc, gt_neg_doc, question_embed, query_atoms_embeds, query_debug_id = query_debug_id, topk=args.topk, debug=debug)                     
                    ## custom
                    # scores = np.array(list(map(lambda x: obj_fn(x.to(ObjHNSW.device)).cpu().numpy(), target_doc_embeds)))
                    scores = obj_fn(torch.from_numpy(target_doc_embeds).to(ObjHNSW.device)).cpu().numpy()
                    # assert np.allclose(bla, scores)
                    # bla = get_cosine_similarity([query_atoms_embeds], target_doc_embeds.numpy())[0]
                    idx = np.argsort(-scores)[:args.topk]
                    rescored_docids = monolitic_ids[0,idx]
                    rescored_scores = scores[idx]
                    results[qid] = {'new_doc_idx': rescored_docids.tolist(), 'new_scores': rescored_scores }
                else:
                    rescored_docids = monolitic_ids[:args.topk].squeeze()
                    rescored_scores = monolitic_scores[:args.topk].squeeze()
                    results[qid] = {'orig_doc_idx': rescored_docids.tolist(), 'orig_scores': rescored_scores }

            # Main path of decompose
            elif is_faiss:            
                monolitic_scores, monolitic_ids = doc_index.search(question_embed, topk_max)                

                if args.is_decompose:
                    target_doc_embeds = doc_embeds[monolitic_ids].squeeze()

                    if args.is_L2_index:                        
                        #monolitic_scores = 1-(np.power(monolitic_scores,2)/2)
                        #query_documents_scores = get_L2_similarity(query_atoms_embeds, target_doc_embeds).transpose()
                        # calc monolitic scores with cosine
                        monolitic_scores = (target_doc_embeds@question_embed.transpose()).transpose()
                    
                    query_documents_scores = get_cosine_similarity([query_atoms_embeds], target_doc_embeds)[0]

                    query_documents_ids = monolitic_ids.transpose()
                    results = fl.run_query_rescoring_docids(qid, monolitic_scores.transpose(), monolitic_ids.transpose(), query_documents_scores, query_documents_ids, atoms_list, formulas, gt_pos_doc, gt_neg_doc, question_embed, query_atoms_embeds, query_debug_id = query_debug_id, topk=args.topk, debug=debug, target_doc_embeds=target_doc_embeds)

                    if dumper is not None:
                        ab_col, abc_col = dump_fused_columns(formulas['formula1'], query_documents_scores.shape[1])
                        n_real_atoms = min(len(atoms_list), 3)
                        fused_used_arr = dump_fused_used_array(
                            formulas['formula1'], ab_col, abc_col,
                            query_documents_scores, monolitic_scores[0],
                        )
                        dumper.add_query(
                            qid=qid,
                            doc_ids=monolitic_ids[0].tolist(),
                            atomic_scores=query_documents_scores[:, :n_real_atoms].T,
                            fused_ab_scores=query_documents_scores[:, ab_col] if ab_col is not None else None,
                            fused_abc_scores=query_documents_scores[:, abc_col] if abc_col is not None else None,
                            monolithic_scores=monolitic_scores[0],
                            relevant_doc_ids=gt_pos_doc,
                            template=operator,
                            fused_used_scores=fused_used_arr,
                        )
                else:
                    returned_docids = monolitic_ids[0][:args.topk]
                    returned_scores = monolitic_scores[0][:args.topk]
                    results[qid] = {'orig_doc_idx': returned_docids.tolist(), 'orig_scores': returned_scores }

            else:
                monolitic_scores = get_cosine_similarity([question_embed], doc_embeds)[0]
                if query_debug:
                    if qid == query_debug_id:
                        query_documents_scores = get_cosine_similarity([query_atoms_embeds], doc_embeds)[0]
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

    if dumper is not None:
        dumper.close()

    if args.sqo_stats_path is not None and _sqo_stats_log:
        os.makedirs(os.path.dirname(args.sqo_stats_path) or ".", exist_ok=True)
        with open(args.sqo_stats_path, "w") as f:
            json.dump(_sqo_stats_log, f, indent=2)
        n = len(_sqo_stats_log)
        mean_step = sum(s["stopped_at_step"] for s in _sqo_stats_log) / n
        frac_early = sum(1 for s in _sqo_stats_log if s["early_stopped"]) / n
        print(f"[SQO stats] n={n}  mean_stopped_at_step={mean_step:.1f}  frac_early_stopped={frac_early:.3f}")

    print("Done.")

