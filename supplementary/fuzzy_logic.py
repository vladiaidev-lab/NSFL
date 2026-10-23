### FUZZY_LOGIC.PY

import numpy as np
from numpy import linalg as LA
import sys
#sys.path.insert(0, '<PATH_TO_REPO_ROOT>')
from pylogics.parsers import parse_ltl 
from pylogics.semantics.ltl import evaluate_ltl, reset_cache, print_cache, eval_cache
from tqdm import tqdm
import json
from scipy.special import softmax
from utils.query_builder import query_parser
import torch.nn.functional as F
import torch
import math, statistics
epsilon = 1e-8

def create_geometric_logic_vector(formula, question_embed, query_embeds):
    new_vector = question_embed

    a_emb = safe_embed(query_embeds, 0)
    b_emb = safe_embed(query_embeds, 1)
    c_emb = safe_embed(query_embeds, 2)
    ab_emb = safe_embed(query_embeds, 3)

    if formula == "atom0":
        new_vector = question_embed
        
    elif formula == "atom0 | atom1":
        new_vector = question_embed
    
    elif formula == "atom0 | atom1 | atom2":
        new_vector = question_embed
        
    elif formula == "atom0 & atom1":
        # a_emb = torch.from_numpy(query_embeds[0]).unsqueeze_(0)
        # b_emb = torch.from_numpy(query_embeds[1]).unsqueeze_(0)
        
        # Works in reranking, bad in ANNS = -62%.
        # new_query_vector = torch.from_numpy(a_emb) + torch.from_numpy(b_emb)
        
        Sa_b = a_emb@b_emb.T
        a_emb + Sa_b* b_emb   #very bad in ANNS = -72%
        new_query_vector = a_emb + b_emb - Sa_b*(a_emb + b_emb)   #same ANNS = -62
        new_vector = (new_query_vector/new_query_vector.norm(2, -1, True))

        # new_vector = question_embed
        
    elif formula == "atom0 & atom1 & atom2":
        # a_emb = torch.from_numpy(query_embeds[0]).unsqueeze_(0)
        # b_emb = torch.from_numpy(query_embeds[1]).unsqueeze_(0)
        # c_emb = torch.from_numpy(query_embeds[2]).unsqueeze_(0)
        
        # Geometrical ANNS bad = -70%
        new_query_vector = a_emb + b_emb  + c_emb
        new_vector = new_query_vector/new_query_vector.norm(2, -1, True)

        # new_vector = question_embed

    elif formula == "atom0 & !atom1":
        # a_emb = torch.from_numpy(query_embeds[0]).unsqueeze_(0)
        # b_emb = torch.from_numpy(query_embeds[1]).unsqueeze_(0)
        
        #Gram–Schmidt: maximize Va, minimize Vb
        Sa_b = a_emb@b_emb.T
        new_query_vector = a_emb - Sa_b * b_emb        
        new_vector = new_query_vector/new_query_vector.norm(2, -1, True)
        # Sa_n = new_vector@a_emb.T
        # new_vector = new_vector/Sa_n

    elif formula == "atom0 & atom1 & !atom2":
        # a_emb = torch.from_numpy(query_embeds[0]).unsqueeze_(0)
        # b_emb = torch.from_numpy(query_embeds[1]).unsqueeze_(0)
        # c_emb = torch.from_numpy(query_embeds[2]).unsqueeze_(0)
        # ab_emb = torch.from_numpy(query_embeds[3]).unsqueeze_(0)

        # Two-step = ANNS bad = -57%
        # #step 1: Build A&B Vector
        # new_query_vectorAB = torch.from_numpy(a_emb) + torch.from_numpy(b_emb)
        # new_vectorAB = (new_query_vectorAB/new_query_vectorAB.norm(2, -1, True))#.unsqueeze(0)

        # #Gram–Schmidt: maximize Va&b, minimize Vc
        # Sab_c = new_vectorAB@c_emb
        # new_query_vector = new_vectorAB - Sab_c * torch.from_numpy(c_emb)
        # new_vector = (new_query_vector/new_query_vector.norm(2, -1, True)).unsqueeze(0)

        # Monolythic AB, then reduce C (graham-schmidt). ANNS bad = -48%,
        Sab_c = ab_emb@c_emb.T
        new_query_vector = ab_emb - Sab_c * c_emb       
        new_vector = new_query_vector/new_query_vector.norm(2, -1, True)

        # new_vector = question_embed

    elif formula == "atom0 | atom1 & !atom2":
        new_vector = question_embed
    
    return new_vector

def safe_embed(query_embeds, idx):
    if idx >= len(query_embeds):
        return None
    e = query_embeds[idx]
    e = e if isinstance(e, torch.Tensor) else torch.from_numpy(e)
    if e.dim() == 1:
        e = e.unsqueeze(0)
    elif e.dim() > 2:
        e = e.squeeze().unsqueeze(0)
    return e

def batch_wrap(f):
    def g(x):
        if len(x.shape) == 1:
            return f(x.unsqueeze(0)).squeeze()
        else:
            return f(x)
    return g

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
class FuzzyLogic:
    def __init__(self,  pylogics = True, type="NSFL", sig_gate=False):
        self.pylogic = pylogics
        # Significance-gated conjunctive correction (--sig_gate). Off by default,
        # in which case AND2/AND3 scoring in evaluate_fuzzy_nsfl_clean is
        # bit-identical to before this flag existed. See run_query_rescoring_docids
        # for where the per-query pool statistic (sigma_K) is computed.
        self.sig_gate = sig_gate
        self._sig_gate_sigma_K = None
        if self.pylogic:
            self.parser = parse_ltl
            self.evaluator = evaluate_ltl       
        else:
            if type=="GEOMETRIC" or type=="GEO":
                self.evaluator = self.evaluate_fuzzy_geometric
            elif type=="NSFL":
                self.evaluator = self.evaluate_fuzzy_nsfl_clean
            else:
                self.evaluator = self.evaluate_fuzzy_nsfl_debug


    def evaluate_fuzzy_geometric(self, monolithic_score, atomic_scores, formula, question_embed, query_embeds, target_doc_embeds, normalized_atomic_scores):  
        monolithic_score = monolithic_score[0]  # Assuming monolithic_score is a list with one element
        new_score = monolithic_score

        a_emb = safe_embed(query_embeds, 0)
        b_emb = safe_embed(query_embeds, 1)
        c_emb = safe_embed(query_embeds, 2)
        ab_emb = safe_embed(query_embeds, 3)

        if formula == "atom0":
            new_score = atomic_scores[0]        
            
        elif formula == "atom0 | atom1":
            a = atomic_scores[0]
            b = atomic_scores[1]
            ab = atomic_scores[2]
            
            new_score = max(monolithic_score, ab, a, b)      
            # Edge-case based on delta, where A,B>AB (both are negative, both reduce the score)
            if (ab<a and ab<b): 
                new_score = ab  #min       
        
        elif formula == "atom0 | atom1 | atom2":
            a = atomic_scores[0]
            b = atomic_scores[1]
            c = atomic_scores[2]
            ab = atomic_scores[3]
            ac = atomic_scores[4]
            bc = atomic_scores[5]
            abc = atomic_scores[6]
            
            new_score = max(monolithic_score, abc, a, b, c, ab, ac, bc)       


        elif formula == "atom0 & atom1":
            # a_emb = torch.from_numpy(query_embeds[0]).unsqueeze_(0)
            # b_emb = torch.from_numpy(query_embeds[1]).unsqueeze_(0)
            new_query_vector = a_emb + b_emb
            new_query_vector_normed = new_query_vector/new_query_vector.norm(2, -1, True)

            new_score1 = F.cosine_similarity( torch.from_numpy(target_doc_embeds).unsqueeze(0) , new_query_vector_normed)
            new_score = np.float32( new_score1.item() )   

        
        elif formula == "atom0 & atom1 & atom2":    
            # a_emb = torch.from_numpy(query_embeds[0]).unsqueeze_(0)
            # b_emb = torch.from_numpy(query_embeds[1]).unsqueeze_(0)
            # c_emb = torch.from_numpy(query_embeds[2]).unsqueeze_(0)
            new_query_vector = a_emb + b_emb  + c_emb
            new_query_vector_normed = new_query_vector/new_query_vector.norm(2, -1, True)

            new_score1 = F.cosine_similarity( torch.from_numpy(target_doc_embeds).unsqueeze(0) , new_query_vector_normed)
            new_score = np.float32(new_score1.item() )  

                
        elif formula == "atom0 & !atom1":
            # a_emb = torch.from_numpy(query_embeds[0]).unsqueeze_(0)
            # b_emb = torch.from_numpy(query_embeds[1]).unsqueeze_(0)
            Sa_b = a_emb@b_emb.T
            
            new_query_vector = a_emb - Sa_b * b_emb
            new_query_vector_normed = new_query_vector/new_query_vector.norm(2, -1, True)

            new_score1 = F.cosine_similarity( torch.from_numpy(target_doc_embeds).unsqueeze(0) , new_query_vector_normed)  
            new_score = np.float32( new_score1.item() ) 

        
        elif formula == "atom0 & atom1 & !atom2":   #requires more research: 1) b vs c as cooefficent on delta c, 
            # a_emb = torch.from_numpy(query_embeds[0]).unsqueeze_(0)
            # b_emb = torch.from_numpy(query_embeds[1]).unsqueeze_(0)
            # c_emb = torch.from_numpy(query_embeds[2]).unsqueeze_(0)
            # ab_emb = torch.from_numpy(query_embeds[3]).unsqueeze_(0)

            # Gram–Schmidt: maximize Va&b, minimize Vc
            Sab_c = ab_emb@c_emb.T
            new_query_vector = ab_emb - Sab_c * c_emb
            new_vector = (new_query_vector/new_query_vector.norm(2, -1, True))

            #score
            new_score1 = F.cosine_similarity( torch.from_numpy(target_doc_embeds).unsqueeze(0) , new_vector) 
            new_score = np.float32( new_score1.item() )   #bug
            
                
        elif formula == "atom0 | atom1 & !atom2":     #not used in QUEST
            a = atomic_scores[0]
            b = atomic_scores[1]
            c = atomic_scores[2]    #not needed
            ab = atomic_scores[3]
            ac = atomic_scores[4]   #not needed
            bc = atomic_scores[5]   #not needed
            abc = atomic_scores[6]
            
            new_score= 2*max(a,b,ab) + ab - abc
            
        
        elif formula == "atom0 ^ atom1":     #XOR, not used in QUEST.   A or B, but not both A and B 
            a = atomic_scores[0]
            b = atomic_scores[1]
            ab = atomic_scores[2]
            
            new_score= max((a + b*a - b*ab), (b + b*a - a*ab))

        return new_score
    

    def evaluate_fuzzy_nsfl_clean(self, monolithic_score, atomic_scores, formula, question_embed, query_embeds, target_doc_embeds, normalized_atomic_scores):  
        monolithic_score = monolithic_score[0]  # Assuming monolithic_score is a list with one element
        new_score = monolithic_score

        if formula == "atom0":
            new_score = atomic_scores[0]        
            
        elif formula == "atom0 | atom1":
            a = atomic_scores[0]
            b = atomic_scores[1]
            ab = atomic_scores[2]
            
            new_score = max(monolithic_score, ab, a, b)      
            # Edge-case based on delta, where A,B>AB (both are negative, both reduce the score)
            if (ab<a and ab<b): #Dec10 fix, reverse < to >
                new_score = ab  #min       
        
        elif formula == "atom0 | atom1 | atom2":
            a = atomic_scores[0]
            b = atomic_scores[1]
            c = atomic_scores[2]
            ab = atomic_scores[3]
            ac = atomic_scores[4]
            bc = atomic_scores[5]
            abc = atomic_scores[6]
            
            new_score = max(monolithic_score, abc, a, b, c, ab, ac, bc)       

        elif formula == "atom0 & atom1":
            a = atomic_scores[0]
            b = atomic_scores[1]
            ab = monolithic_score    #atomic_scores[2] 
            
            #Custom AND T-NORM - still the best:   VAL(BGE)=+43           ,                                      TEST= +37%/ +16%/ +3%/ +26% /-7% (BGE / E5V2 / E5V2-LogiCoL / BLIP/ SigLIP)
            new_score = 2*ab - max(a,b)     
           

            ## April 9th - Confidence-gated (tough call, slightly better except for BLIP):     VAL(BGE)=+41% (vs.+43%),  TEST= +36%/ +19%/ +5%/ +17% /-1% (BGE / E5V2 / E5V2-LogiCoL / BLIP/ SigLIP)
            # new_score = ab + min(normalized_atomic_scores[0], normalized_atomic_scores[1]) * (ab - max(a, b))
            ## - another (bad) idea: Harmonic                                                                            TEST BLIP +8% (vs. 36%)                # harmonic = 2 * a * b / (a + b + epsilon)               # s_and = 2 * ab - harmonic               # new_score = s_and

            if (ab > a+b):
                new_score = ab
            elif self.sig_gate and self._sig_gate_sigma_K is not None:
                # Significance-gated conjunctive correction (gamma_2 = 1).
                # Scale the existing correction R = ab - max(a,b) by its
                # significance relative to the per-query pool noise floor
                # (sigma_K), instead of applying it unconditionally.
                R = ab - max(a, b)
                sigma_K = self._sig_gate_sigma_K
                g = abs(R) / (abs(R) + sigma_K) if sigma_K > 1e-12 else (1.0 if R != 0 else 0.0)
                new_score = ab + g * R

            return new_score


        elif formula == "atom0 & atom1 & atom2":
            a = atomic_scores[0]
            b = atomic_scores[1]
            c = atomic_scores[2]
            ab = atomic_scores[3]
            ac = atomic_scores[4]
            bc = atomic_scores[5]
            abc = monolithic_score  #VAL=0.6101 (+94%)    # atomic_scores[6] with abc=a+' AND '+b + ' AND '+ c GIVES VAL=0.02083 -34%
    
            #Custom Double AND T-Norm
            new_score = 2*abc-max(a,b,c)                  # Good Theoretically: BAAI: VAL great 0.06101 (+94%), TEST 0.03973 bad (-6%),     TRAIN good(+28%). NOMIC: Improves but less in Train,Test, Better in Train;   COCO BLIP-L +18%, SigLIP +4%
                            #new_score = abc + 2*(abc-max(a,b,c)) #test Feb 16   # Theory idea: double the impact. Results: VAL worse 0.0472 +50%,  TEST very bad 0.03337 -21% (worse)
            
                            # # New idea: add only one smallest delta - VAL=0.06079(+92.8%, max:+94%) , TEST(Quest augmentation)=0.0423 (-0.027%, 2nd best) and TEST(simple augmentation)=0.03754 (-11%)
                            # deltaA_b = ab-b            # deltaA_c = ac-c            # deltaB_c = bc-c            # deltaB_a = ab-a            # deltaC_a = ac-a            # deltaC_b = bc-b
                            # new_score = abc + min (deltaA_b, deltaA_c, deltaB_a, deltaB_c, deltaC_a, deltaC_b)  # VAL=0.06079(+92.8%, max:+94%) , TEST=0.0423 (-0.027%, so far the best)????????????????
                            
                            
                            # # Sorting based a,b,c - then find other deltas. then add to abc the extra deltas. #Bad. VAL=0.03752 +19% (worse (max:+94%)), TEST=0.03454 -17% (max:-6%)
                            # if (a==max(a,b,c)):     #-13% damage (VAL) (else=monolithic);   #     deltaB = ab-a;   #     deltaC = ac-a; #     new_score = abc + deltaB + deltaC;       # elif (b==max(a,b,c)):;         # # if (b==max(a,b,c))     #+19% help (VAL);             #     deltaA = ab-b;            #     #deltaC = abc-ab #bc-a            #     deltaC = bc-b            #     new_score = abc + deltaA + deltaC            # else:            # # if (c==max(a,b,c)):       #+7% help (VAL)            #     deltaA = ac-c            #     deltaB = bc-c            #     new_score = abc + deltaA + deltaB            
                            
                            # # New idea: add only one smallest delta which is smallest by absolute value - VAL=0.03624(+15%) , TEST=0.0435 (+2.8%)
                            #new_score = abc                deltaA_b_a = abs(deltaA_b)              deltaA_c_a = abs(deltaA_c)              deltaB_c_a = abs(deltaB_c)              deltaB_a_a = abs(deltaB_a)              deltaC_a_a = abs(deltaC_a)              deltaC_b_a = abs(deltaC_b)              if (deltaA_b_a==min(deltaA_b_a, deltaA_c_a, deltaB_c_a, deltaB_a_a, deltaC_a_a, deltaC_b_a)):                new_score = abc + deltaA_b                 elif (deltaA_c_a==min(deltaA_b_a, deltaA_c_a, deltaB_c_a, deltaB_a_a, deltaC_a_a, deltaC_b_a)):                new_score = abc + deltaA_c               elif (deltaB_c_a==min(deltaA_b_a, deltaA_c_a, deltaB_c_a, deltaB_a_a, deltaC_a_a, deltaC_b_a)):                 new_score = abc + deltaB_c              elif (deltaB_a_a==min(deltaA_b_a, deltaA_c_a, deltaB_c_a, deltaB_a_a, deltaC_a_a, deltaC_b_a)):                 new_score = abc + deltaB_a              elif (deltaC_a_a==min(deltaA_b_a, deltaA_c_a, deltaB_c_a, deltaB_a_a, deltaC_a_a, deltaC_b_a)):                 new_score = abc + deltaC_a              elif (deltaC_b_a==min(deltaA_b_a, deltaA_c_a, deltaB_c_a, deltaB_a_a, deltaC_a_a, deltaC_b_a)):                 new_score = abc + deltaC_b

            
            # April 9th - Fix Over-penalization. The delta gets one chance to correct in binary (one weak atom). In ternary, there are two potentially weak atoms, but you're still applying a single full-strength correction.
            new_score = abc + 1/2 * (abc - max(a,b,c))                              # VAL worse (+35%, vs. +94%), but TEST better (+5% / +1% - BGE/E5v2), still not perfect - good enough. latest.
                            #option2: iteratively (doesn't work) ####new_score = 2*abc - max((ab if ab > a + b else 2*a - max(a, b)), c)     # VAL worse but better (+40%),    TEST terrible (-375% / -55% /-40% - BGE/E5v2/E5v2Logicol) - does not work!
            
            
            # SIGLIP: if the monolithic score is bigger the sum of it's components, then semantics in the monolithical query has extra value which must be kept. However, this doesn't improve anything, just don't hurt (no change)
            if (abc > a+b+c):
                new_score = abc
            elif self.sig_gate and self._sig_gate_sigma_K is not None:
                # Significance-gated conjunctive correction (gamma_3 = 0.5).
                R = abc - max(a, b, c)
                sigma_K = self._sig_gate_sigma_K
                g = abs(R) / (abs(R) + sigma_K) if sigma_K > 1e-12 else (1.0 if R != 0 else 0.0)
                new_score = abc + 0.5 * g * R

        elif formula == "atom0 & !atom1":
            a = atomic_scores[0]
            b = atomic_scores[1]
            ab = atomic_scores[2]
            
            #new_score =  b*a - b*ab + a   # Equals: b*(2*a-ab) + (1-b)*a     #VAL=0.04738 (+89%)                   

            #April 8th with Normalized scores (scientifically more correct)
            bN = normalized_atomic_scores[1]    #bTag is the normalized score of atom1, which is the one we want to negate. If bTag is high, it means that atom1 is likely to be true, and therefore we want to reduce the score of the whole formula. If bTag is low, it means that atom1 is likely to be false, and therefore we want to increase the score of the whole formula. The formula b*a - b*ab + a captures this intuition: if b is high, then the term -b*ab will have a larger negative impact on the score, while if b is low, then the term b*a will have a larger positive impact on the score. This way, we are effectively using the normalized score of atom1 to modulate the contribution of atom0 and the interaction between atom0 and atom1 (ab) in determining the final score of the formula.
            new_score = a - bN*(ab - a)     #VAL = 0.0468 (+87%)    TEST=

            #Oct 24 - "Smart+Edgecase" Deal with situation when A,B>AB --> both A and B are negative (incorrect tokens)
            if (a>ab and b>ab):  #when A,B>AB then all are strong negative, take worst score, which is AB
                new_score = ab                    #Val=0.0481 (better or best?, +92%)
        
        elif formula == "atom0 & atom1 & !atom2":   #requires more research: 1) b vs c as cooefficent on delta c, 
            a = atomic_scores[0]
            b = atomic_scores[1]
            c = atomic_scores[2]    #not needed
            ab = atomic_scores[3]
            ac = atomic_scores[4]
            bc = atomic_scores[5]   #not needed
            abc = atomic_scores[6]
            
            # Comp(Oct8): A&B + c*(-Δ_c) +ab?  .   But, why extra +ab?
            a_and_b = (2*ab - max(a,b))
            #Siglip fix
            if (ab>(a+b)):
                a_and_b = ab
            #  Base function: 𝑆𝑐𝑜𝑟𝑒_(𝐴&𝐵&!𝐶)=𝑆𝑐𝑜𝑟𝑒_(𝐴&𝐵)− ∆_(𝐶 ) = 𝑆_𝐴𝐵 + ∆_min⁡(𝑆_𝐴, 𝑆_𝐵 ) −∆_(𝐶 ) . Increase impacts of the weakest parts A, B and then decreases the impact of  C. Compositional use of Basic formula for AND + Basic-B formula for NOT
            # new_score = a_and_b - c*(abc - 2*ab)          #wrong          #Val: 0.02595 +32% (debug=0.02595 best!)     TEST: 0.0198 (+30%) 

             #April 8th wit Normalized scores (scientifically more correct)
            cN = normalized_atomic_scores[2]    #bTag is the normalized score of atom1, which is the one we want to negate. If bTag is high, it means that atom1 is likely to be true, and therefore we want to reduce the score of the whole formula. If bTag is low, it means that atom1 is likely to be false, and therefore we want to increase the score of the whole formula. The formula b*a - b*ab + a captures this intuition: if b is high, then the term -b*ab will have a larger negative impact on the score, while if b is low, then the term b*a will have a larger positive impact on the score. This way, we are effectively using the normalized score of atom1 to modulate the contribution of atom0 and the interaction between atom0 and atom1 (ab) in determining the final score of the formula.
            new_score = a_and_b - cN*(abc - ab)         #Val: 0.02621 +12%      TEST: 0.0246 (+61%, close to best) - scientifically most correct
            # print(f"c={c}, cN={cN},  monolithic_score={monolithic_score},   ab={ab},  c={c},  abc={abc},  a_and_b={a_and_b},  abc-2*ab={abc-2*ab},  new_score C={a_and_b - c*(abc - 2*ab)},  new_score C_normalized={new_score}")
            
                
        elif formula == "atom0 | atom1 & !atom2":     #not used in QUEST
            a = atomic_scores[0]
            b = atomic_scores[1]
            c = atomic_scores[2]    #not needed
            ab = atomic_scores[3]
            ac = atomic_scores[4]   #not needed
            bc = atomic_scores[5]   #not needed
            abc = atomic_scores[6]
            
            new_score= 2*max(a,b,ab) + ab - abc
            
        
        elif formula == "atom0 ^ atom1":     #XOR, not used in QUEST.   A or B, but not both A and B 
            a = atomic_scores[0]
            b = atomic_scores[1]
            ab = atomic_scores[2]
            
            new_score= max((a + b*a - b*ab), (b + b*a - a*ab))

        return new_score


    def evaluate_fuzzy_nsfl_debug(self, monolithic_score, atomic_scores, formula, question_embed, query_embeds, target_doc_embeds, normalized_atomic_scores):  
        monolithic_score = monolithic_score[0]  # Assuming monolithic_score is a list with one element
        
        if formula == "atom0":
            new_score = atomic_scores[0]        
        
        elif formula == "atom0 | atom1":
            a = atomic_scores[0]
            b = atomic_scores[1]
            ab = atomic_scores[2]
            #Custom OR T-NORM 
            new_score = max(monolithic_score, ab, a, b)      #Val=0.1402 (+4%),       Test=0.1465 (+8%)
            #Oct24 - Edge-case based on delta, where A,B<AB (both are negative, both reduce the score - "airplane" to "dog or algebra")
            # if (monolithic_score<a and monolithic_score<b): #Dec 10 orfix- I reversed them, found a bug
            #     new_score = monolithic_score  #min       #Val=0.1421 (+5%)
            if (ab<min(a,b)): 
                new_score = ab  #min       
    
        elif formula == "atom0 | atom1 | atom2":
            a = atomic_scores[0]
            b = atomic_scores[1]
            c = atomic_scores[2]
            ab = atomic_scores[3]
            ac = atomic_scores[4]
            bc = atomic_scores[5]
            #Custom OR T-NORM
            new_score = max(monolithic_score, a, b, c, ab, ac, bc)       #Val=0.1941 (+14%)     Test=0.1336 (+24%)
            #Oct24 - Attempt to deal with Edge-case based on delta, where A,B,C<ABC (all are negative, both reduce the score therefore negative classification for all) --> take lowest which is full abc
            # if (monolithic_score<ab and monolithic_score<bc and monolithic_score<ac):
            #     new_score = monolithic_score                             #Val=0.1925 (+13%)     Test=0.1281 (+19%)
            
            # #Dec 10 - ororfix - proven not good
            # if (monolithic_score<a and monolithic_score<b and monolithic_score<c):
            #     new_score = monolithic_score                             #Val=0.1911 (+12%)     Test=0.1286 (+20%)

            #Dec 15 - simple max between atoms' scores. Reason: sometimes atom2 is wrong, but sub-element there complements atom1, and together instead of reducing the score of atom1 it improves the atom1_and_atom2 (example: man, giraffe near a fence. there is a man, a giraffe but no fence)
            # new_score = max(a,b,c)                                        #Val=0.1801 (+5%, less)   Test=0.12605 (+17%)
        
        elif formula == "atom0 & atom1":
            a = atomic_scores[0]
            b = atomic_scores[1]
            ab = atomic_scores[2]
            # q_scores = question_embed @ np.array(query_embeds).transpose().squeeze()
            # b_query_sim = q_scores[0][1]
            
            #Custom AND T-NORM
            new_score = 2*monolithic_score - max(a,b)                                                    # Works great. 0.055 +43% (val),  0.05248 +37% (test)  , proven theoretically .
            #new_score = 4.8*monolithic_score -0.46*a -2.9*max(a,b)                                
            # new_score = 2.32*monolithic_score -0.186*a -1.337*max(a,b)                                
            # new_score = 2.26*monolithic_score -1.4*max(a,b)                                
            # new_score = 4.2*monolithic_score -0.5*a -2*max(a,b)
            # new_score = 4.23*monolithic_score -0.47*a -2*max(a,b) -0.135*b_query_sim
            # new_score = 6.7*monolithic_score -1.9*a -b +0.14*min(a,b) -3.6*max(a,b)

            #Sept 27 
            # a_emb = query_embeds[0]
            # b_emb = query_embeds[1]
            # if a_emb.shape == torch.Size([1, 256]) and b_emb.shape == torch.Size([1, 256]):
            #     query_parts_sim = F.cosine_similarity(b_emb, a_emb)
            # else:
            #     query_parts_sim = b_emb@a_emb
            
            # Sept 27 
            # new_score = monolithic_score + (monolithic_score - max(a,b))*min(5,1/(1-query_parts_sim))   # Works less well. 0.0483 +24% (val)

            # Cooefficient1
            #new_score = monolithic_score + (monolithic_score - max(a,b))*(max(a,b)/min(a,b))              # VAL: 0.0597 +53% (val, +10% vs. base),   TEST: 0.0522 (+36%)      COCO-BLIP: 0.170 (+21%)     COCO-SIGLIP:0.006 (-96.6%)


            # Oct 16th
            # New idea: A&B you interested in scores of both, so their average makes sense. But if one item is 0.4, 0.3 and second item is 0.35, 0.35 then the second item is better
            #new_score = statistics.mean([a,b]) * min(a,b)/max(a,b)                                     #Val=0.0104 (very bad)
            #new_score = statistics.mean([a,b, monolithic_score])                                        #Val=0.0257 (very bad)

            # Standard fuzzy logic:
            #new_score = min(a,b)                                                                        #Val=0.0139 (very bad) - Fuzzy Logic does not work!


            # Oct 8th 
            # SIGLIP Fix : if the monolithic score is bigger the sum of it's components, then semantics in the monolithical query has extra value which must be kept. However, this doesn't improve anything, just don't hurt (no change)
            if (monolithic_score > a+b): 
                new_score = monolithic_score                                                             
            
            # Oct 29th - Use delta only when it is dual positive or dual negative.                       #Val=0.0522 (+34%, less good than basic custom TNORM)
            if ((a<ab and b<ab) or (a>ab and b>ab) ) :
                # Either A,B positive - lets increase the score
                # Or     A,B negative - lets decrease the score
                new_score = 2*ab - max(a,b)     #Custom T-NORM

                # SIGLIP fix
                if ((a+b)<ab):
                    new_score = ab
            else:
                # Either A<AB<AB or B<AB<A and then if |delta_A|>|delta_B| or vice versa - A strong, B weak (negative/positive, depenging case)
                # keep the scores
                new_score = ab                                                                          #Val=0.0522 (+34%, less good than basic custom TNORM)

            new_score_nsfl=new_score
            
            #Cooefficient2-e
            # e_value= math.e
            # new_score = monolithic_score + (monolithic_score - max(a,b))*e_value                         # Works BAD. Val=0.0484
            
            #new_score = (a*b)            
            #new_score = 2*monolithic_score - a                       # VV verifed and proved (as expected) that it is not good.
            # # alternate subspace formulation
            # a_emb = query_embeds[0]
            # b_emb = query_embeds[1]
            # norm = 2
            # new_score = (a*LA.norm(a_emb,norm) + b*LA.norm(b_emb,norm))/LA.norm((a_emb+b_emb),norm)         
             
            #Oct24 - When rescoring is too dramatic
            # #Before                                             #       Val=0.0055
            # if (query_parts_sim<0.5):                           #0.7--> Val=0.0410  0.5--> Val=0.055
            #     new_score = monolithic_score*0.8
            # if ((new_score/monolithic_score)<0.9):
            #     new_score = monolithic_score*0.9

            #Jan 21 with Sefi - new vector
            # a_emb = query_embeds[0]
            # b_emb = query_embeds[1]
            # new_query_vector = torch.from_numpy(a_emb) + torch.from_numpy(b_emb)
            # new_query_vector_normed = new_query_vector/new_query_vector.norm(2, -1, True)

            # #new_score = F.cosine_similarity( torch.from_numpy(target_doc_embeds).view(1,-1) , new_query_vector_normed)
            # new_score = F.cosine_similarity( torch.from_numpy(target_doc_embeds).unsqueeze(0) , new_query_vector_normed.unsqueeze(0))
            # if (ab>new_score):
            #     new_score = torch.tensor([ab], dtype=torch.float32)

            # a_emb = torch.from_numpy(query_embeds[0]).unsqueeze_(0)
            # b_emb = torch.from_numpy(query_embeds[1]).unsqueeze_(0)
            # new_query_vector = a_emb + b_emb
            # new_query_vector_normed = new_query_vector/new_query_vector.norm(2, -1, True)

            # new_score1 = F.cosine_similarity( torch.from_numpy(target_doc_embeds).unsqueeze(0) , new_query_vector_normed)
            # new_score = np.float32( new_score1.item() )   #no bug?

        elif formula == "atom0 & atom1 & atom2":
            a = atomic_scores[0]
            b = atomic_scores[1]
            c = atomic_scores[2]
            ab = atomic_scores[3]
            ac = atomic_scores[4]
            bc = atomic_scores[5]
            #abc = atomic_scores[6] #needs it 
            
            #new_score = 3*monolithic_score-max(a,b,c)                        #Best on VAL/TEST (+35%/+5%) by Ofer. But, no theory!
            
            #Custom Double AND T-Norm
            #new_score = 2*monolithic_score + ab - max(a,b) - max(ab,c)       #Good Theoretically: BAAI: VAL 0.04622 (+47%), TEST 0.04017(-5%), TRAIN (+33%)
            abc = monolithic_score  #needs atomic_scores[6] but it is missing in the local file. 
            new_score = 2*abc - max(a,b,c)                         #Good Theoretically: BAAI: VAL great (+94%), TEST bad (-6%),      TRAIN good(+28%). NOMIC: Improves but less in Train,Test, Better in Train
            new_score_nsfl=new_score
            
            if (abc>(a+b+c)):
                new_score = abc
            
            #new_score = monolithic_score + ab + bc - max(a,b) - max(b,c)    #bad, why?         VAL 0.02896 (-8%)???
            
            #new_score = (a*b*c)
            #new_score = min(a,b,c)
            #new_score = 4*monolithic_score-max(a,b,c)            
            #new_score = 2*(2*ab-max(a,b))-monolithic_score            
            #new_score = min(a,b,c,ab,ac,bc) + 4*monolithic_score - max(a,b,c,ab,ac,bc)

            #Dec16
            # new_score = (a+b+c)/3                                           #VAL=0.00965 (-69%, very bad),  TEST=0.02326 (-45%, very bad)

            # # # Geometrical A&B&C   VAL 0 = very bad.
            # a_emb = torch.from_numpy(query_embeds[0]).unsqueeze_(0)
            # b_emb = torch.from_numpy(query_embeds[1]).unsqueeze_(0)
            # c_emb = torch.from_numpy(query_embeds[2]).unsqueeze_(0)
            # new_query_vector = a_emb + b_emb  + c_emb
            # new_query_vector_normed = new_query_vector/new_query_vector.norm(2, -1, True)

            # new_score1 = F.cosine_similarity( torch.from_numpy(target_doc_embeds).unsqueeze(0) , new_query_vector_normed)
            # new_score = np.float32(new_score1.item() )  #no-bug?

            #also separately they are bad (VAL=0)
            # new_query_vector_ab = a_emb + b_emb
            # new_query_vector_ab_normed = new_query_vector_ab/new_query_vector_ab.norm(2, -1, True)

            # new_query_vector_abc = new_query_vector_ab_normed + c_emb
            # new_query_vector_abc_normed = new_query_vector_abc/new_query_vector_abc.norm(2, -1, True)

            # new_score = F.cosine_similarity( torch.from_numpy(target_doc_embeds).unsqueeze(0) , new_query_vector_abc_normed)


            # # if (ab>new_score):
            # #     new_score = torch.tensor([ab], dtype=torch.float32)

            

        elif formula == "atom0 & !atom1":
            a = atomic_scores[0]
            b = atomic_scores[1]
            ab = atomic_scores[2]

            # new_score = (a*(1 - b))
            #Custom NOT T-NORM = 2*AtomPositive - FullQuery
            # whereas 
            # - AtomPositive = Innerproduct (AtomPositiveVector, DataItemVector)
            # - FullQuery = Innerproduct (FullQueryVector, DataItemVector)
            #new_score = a*2 - monolithic_score

            a_emb = query_embeds[0]
            b_emb = query_embeds[1]

            # if a_emb.shape == torch.Size([1, 256]) and b_emb.shape == torch.Size([1, 256]):
            #     query_parts_sim = F.cosine_similarity(b_emb, a_emb)
            # else:
            #     query_parts_sim = b_emb@a_emb
            
#            if query_parts_sim > 0.8:      #works on test, but theoretically bad? 
#            if b > 0.8:                    #Better theorerically - compare b to data-item, if too far, then use only A. 
 #               #new_score = 2*a - monolithic_score                   # proven theoretically and proctically as best on VAL (0.0528), on TEST is good (not best) (0.0741)
  #              new_score = 2*a - ab                                # theoretically is better , practically is OK on VAL   (0.0501). On TEST is better          0.0758
   #         else:
    #            new_score = a                                       # no theoretical proof, but works well: on VAL: 0.0528.   without this, the performance (on VAL) drops to 0.045 (-0.007)
                                
            #new_score = a      #better than monolytic
            #new_score = b*(2*a-monolithic_score) + (1-b)*a           
            #SIGMOID doesn't work, needs a fix and test #new_score = self.sigmoid(100*(b-0.8))*(2*a-ab) + (1-self.sigmoid(100*(b-0.8)))*a           
            new_score = b*(2*a-ab) + (1-b)*a                        # OK on VAL= 0.0473, best on TEST=0.07734 

            #Oct 24 - "Smart+Edgecase" Deal with situation when A,B>AB --> both A and B are negative (incorrect tokens)
            if (a>ab and b>ab):  #when A,B>AB take worse score which is AB
                new_score = ab                    #Val=0.0481 (better or best?, +92%)
                # new_score = ab*0.8              #Val=0.0476 (worse, +90%)
                # new_score = monolithic_score     #Val=0.0470 (same, +88%)

            else:                #either correct (1,0 or 0,1) - use the formula) 
                new_score = b*(2*a-ab) + (1-b)*a  #Val=0.0481 (better or best?, +92%)
                
            new_score_nsfl=new_score

            #Dec24 increase (a-delta_b) impact by alpha%
            # bStar = b+(1-b)*0.10                                 # VAL(25%)=0.04822 (best, 92.4% better by 0.3%), but VAL(50%)=0.0464(worse by -5%), also VAL(10%)=0.0476 (worse by 1%); Conclusion - no stable meaning.
            # new_score = bStar*(2*a-ab) + (1-bStar)*a

            # if (a<ab and b<ab):  (mistake, still some numbers)
            #     new_score = ab                  #Val=0.0447 (less good, +79%)
            #     new_score = ab*0.8              #Val=0.0470 (OK, +88%)
            #     new_score = ab*0.5              #Val=0.0470 (same, +88%)

            #OCt24 Gil - base on Delta classifier
            # delta_a = ab-b
            # delta_b = ab-a
            # if (delta_a>0 and delta_b<0): # the only correct case
            #     new_score = monolithic_score*1.2   #Val=0.0405 (not so good)
            #     new_score = monolithic_score*1.5   #Val=0.0405 (not so good)
            #     new_score = ab*1.2                #Val=0.0370 (bad)
            #     new_score = a+delta_a-delta_b     #Val=0.0478 (Ok)
            # else:
            #     new_score = monolithic_score/1.5   #Val=0.0405 (not so good)
            #     new_score = monolithic_score/1.5   #Val=0.0405 (not so good)
            #     new_score = ab/1.2                #Val=0.0370 (bad)
            #     new_score = a                     #Val=0.0478 (Ok)

            #Dec 29th test: b*DeltaB - use smoothing only on the Delta, keep A&B as is.
            #new_score = a-b*(ab-a)  #also pretty good, a bit less. VAL=0.0463(+85%)    TEST=0.07793 (+55%, best)

            # #Jan 21 (with Sefi) - Gram–Schmidt: maximize Va, minimize Vb
            # if a_emb.shape == torch.Size([1, 256]) and b_emb.shape == torch.Size([1, 256]):
            #     Sa_b = F.cosine_similarity(b_emb, a_emb)
            # else:
            #     Sa_b = b_emb@a_emb
            # Sa_b = b_emb@a_emb
            # # new_score = a-Sa_b*b        #Val- bad, 0.0398 (+59% vs optimum +92%)

            # new_query_vector = torch.from_numpy(a_emb) - Sa_b * torch.from_numpy(b_emb)
            # new_query_vector_normed = new_query_vector/new_query_vector.norm(2, -1, True)

            # new_score1 = F.cosine_similarity( torch.from_numpy(target_doc_embeds).unsqueeze(0) , new_query_vector_normed.unsqueeze(0))  
            # new_score = np.float32( new_score1.item() ) #bug
            
            # if (ab<min(a,b)) :
            #     new_score = torch.tensor([ab], dtype=torch.float32)
            
            #TESTS      baseline      a only      if a@b>0.8, -mono   ifa@b>0.8 -ab   if b>0.8 -mono     if b>0.8 -ab   ifb>0.7 -ab     smart-mono      smart-ab            Smart+Edgecase
            #VAL        0.0239        0.0495      0.0528 !!(best)     0.0501          0.0495              0.0495        0.0498          0.0465          0.0477              0.0481
            #TEST       0.0501        0.0753      0.0741              0.0758          ?                     ?                           0.0732          0.07734 !! BEST     
            

        elif formula == "atom0 & atom1 & !atom2":
            a = atomic_scores[0]
            b = atomic_scores[1]
            c = atomic_scores[2]
            ab = atomic_scores[3]
            abc = atomic_scores[4]    #fits older list, now this is bc
            #abc = atomic_scores[6]
            a_emb = query_embeds[0]
            b_emb = query_embeds[1]
            c_emb = query_embeds[2]
            ab_emb = query_embeds[3]
            #q_scores = (question_embed @ np.array(query_embeds).transpose()).squeeze()
            # if c_emb.shape == torch.Size([1, 256]) and ab_emb.shape == torch.Size([1, 256]):
            #     query_parts_sim = F.cosine_similarity(c_emb, ab_emb)
            # else:
            #     query_parts_sim = c_emb@ab_emb
            # if NOT term similar to query
#            if query_parts_sim > 0.8: 
 #               new_score = 2*ab - monolithic_score                  #VAL=0.0231 earlier variant (BEST on TEST)
  #              #new_score = 2*ab - abc                              #VAL=0.0220
   #         else: #most cases are here
    #            new_score = ab                                      
     #           #new_score = 2*ab - max(a,b)                        #VAL=0.020 (with monolitic in the if). Theorerical correct
            
            #new_score = ab                                         #VAL=0.022

            #new_score = b*(2*ab - monolithic_score) + (1-b)*ab        #VAL=0.0191
            #new_score = b*(2*ab - abc) + (1-b)*ab                    #VAL=0.0240  !! ,   TEST = 
            
            ## smart-abc-comp(bug b) Best August formula (but wrong b/1-b) - Plain not C, vs compositional
            #new_score = b*(2*ab - abc) + (1-b)*(2*ab - max(a,b))     #VAL=0.0236 !(close to best) ,  TEST=0.0270 (best)

            ##!!!!
            # # Sept 1st = Plain AB (not logic) + !C), +(1-c)* Logic A&B    
            ###new_score = c*(2*ab - abc) + (1-c)*(2*ab - max(a,b))      #VAL=0.0232 !(????) ,  TEST=0.0259  COMPOSITIONAL! (????), ???     COCO-BLIP=0.262/0.265 (+72%)    COCO-SIGLIP: 0.193 (+40.4%)

            # # Plain AB + !C
            # new_score = c*(2*ab - abc) + (1-c)*(ab)                   #VAL=0.0235   , TEST=0.0251

            # # Sept 27th - CompMath (compositional math)
            # new_score = c*(3*ab - abc - max(a,b)) + (1-c)*(2*ab - max(a,b))     #VAL=0.0205,  TEST=0.0232
            
            # # Sept 27th - CompMath2 not C, or else plain AB
            # new_score = c*(3*ab - abc - max(a,b)) + (1-c)*ab                    #VAL=0.0224,  TEST=0.0259

            # # CompMath2smart cooefficient + fully compositional
            # new_score = c * (2*ab - abc + (max(a,b)/min(a,b))*(ab - max(a,b)))   +  (1-c) * (ab + (max(a,b)/min(a,b))*(ab-max(a,b)))    
            # #                                                         #VAL=0.0199 (not good), TEST=0.0226 (not good)

            # Oct 8th - 
            #org Sept 1st = c(plain AB - deltaC) + (1-c)*(A&B): new_score = c * (2*ab - abc) + (1-c)*(2*ab - max(a,b)) = 2*ab*c - c*abc + 2*ab - max(a,b) - 2*c*ab - c*max(a,b)
            # Compositional Fixed(Oct8) - Compositional A&B-C Math Fix (had wrong delta-c calculation, but which? I lost it...): BEST on VAL, but so so on TEST !!!
            #new_score = (2*ab - max(a,b)) + c*(2*ab - abc)                              #Val: 0.0259 (best!)     TEST: 0.0198 (improves, but slightly)

            # Comp(Oct8): A&B+b*(AB_Plain-Δ_c) - b
            #new_score = (2*ab - max(a,b)) + b*(2*ab - abc)                              #Val: 0.0241 (+23%, 2nd best!) TEST: 0.0274 (best!) . but why b*, not c*? theoretically c is more correct
            a_and_b = (2*ab - max(a,b))
            if (ab>(a+b)):
                a_and_b = ab
            new_score = a_and_b - b*(abc - 2*ab)                                         #Val: 0.0241 (+22.5%, 2nd best!) TEST: 0.0274 (+79% best!) 
            new_score = a_and_b - c*(abc - 2*ab)                                         #Val: 0.02595 (+32%, best!)  
            new_score_nsfl=new_score
            
            # new_score = a_and_b - (abc-ab)                                               #Val: 0.02199 (+12%). OK, not best. 
            # new_score = a_and_b - c*(abc-ab)                                               #Val: 0.021474 (+9%). OK, not best. 
            # new_score = a_and_b - 2*c*(abc-ab)                                               #Val: 0.02282 (+16%). OK, not best. 
            # new_score = a_and_b - 3*c*(abc-ab)                                               #Val: 0.003904 Super bad
            # new_score = a_and_b - 1.6*c*(abc-ab)                                               #Val: 0.02413 (+22.5%). OK
            # new_score = a_and_b - 1.8*c*(abc-ab)                                               #Val: 0.02429 (+23.3%). BEST BEST BEST
            # new_score = a_and_b - 1.9*c*(abc-ab)                                               #Val: 0.02279 (+15.7%). OK

            #Best, but b is heuristical
            


            #(Dec09) - better deltaC with c coefficient - Need to test
            #Reminder for A & !B = b*(2*a-ab) + (1-b)*a
            #test1 to test:
            #new_score = c*(2*a_and_b - abc) + (1-c)*a_and_b                             #Val: 0.0205 (worse)

            #test2 to test:
            #new_score = c*(2*ab - abc) + (1-c)*a_and_b                                  #Val: 0.0235 (better, 3rd best)     TEST: 0.262 (second best)

            #test 3
            #new_score = c*(a_and_b - abc + ab) + (1-c)*a_and_b                          #Val: 0.0215 (worse)   TEST: 0.0243 (worse)

            
            #Compositional + A&B with coefficient (worse than without coefficient)
            #new_score = ab + (ab - max(a,b))*(max(a,b)/min(a,b)) + c*(2*ab - abc)      #Val: 0.0254             TEST: 0.0195


            #Oct 8th Comp-Math2+C_coeff  (mathematically best?, but results are not good)
            #new_score = 2*ab-max(a,b) + c*(ab - abc)                                    #VAL: 0.0215 (not so good)    TEST: 0.0198


            #Oct 8th Comp-Math2f: A&B - delta-C 
            #new_score = 2*ab-max(a,b) - (abc-ab)
            #new_score = 3*ab - max(a,b) - abc                                           #VAL: 0.0220 (not so good)    TEST: 0.0245  (not bad)      COCO-BLIP: 0.247 (61.5%)      COCO-SIGLIP: 0.159 (15.6%)

            # SIGLIP mini-fix for Comp-Math2f
            #if (ab > a+b):  #Plain AB - delta c
            #    new_score = ab - (abc-ab)
           

            # if (ab > a+b): 
            #     new_score = c*(2*ab - abc) + (1-c)*(ab)
            # else:
            #     new_score = c*(2*ab - abc) + (1-c)*(2*ab - max(a,b))                    #Val: 0.235

            #Dec 29 - simple negative smoothing (ALSO GOOD, ON TEST AT LEAST)
            #new_score = a_and_b - c*(abc - ab)                                         #Val: 0.0214 (less good, +9%!) TEST: 0.0243 (+59% (best is +79%))     
            

            #TESTS          baseline      ab only      if ab@c>0.8, -mono  ifa@b>0.8 -ab   if b>0.8 -mono    if b>0.8 -ab   ifb>0.7 -ab     smart-mono     smart-ab      smart-abc-comp(bug b, best)   smart-abc-comp-fix     Plain AB + !C   CompMath    CompMath2    CompMath2smart   Compositional Fixed(Oct8)    Compositional Fixed + Coef(Oct8)   Comp(Oct8): A&B+b*(AB_Plain-Δ_c)      Comp-Math2f: A&B - delta-C 
            #VAL            0.0197        0.022        0.0231              0.0220                                                           0.0191         0.0240 !!     0.0236 (!)                    0.0232                 0.0235          0.0205      0.0224      0.0199            0.0259 (!!!)                  0.0254                            0.241 (! 3rd)                         0.0220
            #TEST           0.01529                    0.0223                                                                               0.0218         0.0253        0.0270 !!                     0.0259                 0.0251          0.0232      0.0259      0.0226            0.0198                        0.195                             0.274 (!!! 1st)                       0.0245
            #COCO-L-BLIP    0.152                                                                                                                                                                      0.262 (+72.6%)                                                                                                                                                                         0.247 (+61.5%)
            #COCO-L-SIGLIP  0.137                                                                                                                                                                      0.193 (+40%)                                                                                                                                                                           0.159 (+15.6%)


            #Old variant
            #new_score = 2*(2*ab-max(a,b))-monolithic_score   
            #new_score = query_parts_sim*(2*ab - monolithic_score) + (1-query_parts_sim)*ab
            #new_score = ab
            #new_score = (a*b*(1-c))
            #new_score = 8*(a+b)-monolithic_score            
            #new_score = 2*max(a,b,ab) - monolithic_score

            # #Feb 9th
            # #step 1: Build A&B Vector
            # new_query_vectorAB = torch.from_numpy(a_emb) + torch.from_numpy(b_emb)
            # new_vectorAB = (new_query_vectorAB/new_query_vectorAB.norm(2, -1, True))#.unsqueeze(0)

            # #Gram–Schmidt: maximize Va&b, minimize Vc
            # Sab_c = new_vectorAB@c_emb
            # new_query_vector = new_vectorAB - Sab_c * torch.from_numpy(c_emb)
            # new_vector = (new_query_vector/new_query_vector.norm(2, -1, True)).unsqueeze(0)

            # #score
            # new_score1 = F.cosine_similarity( torch.from_numpy(target_doc_embeds).unsqueeze(0) , new_vector) 
            # new_score = np.float32( new_score1.item() )   #bug

        elif formula == "atom0 | atom1 & !atom2":
            a = atomic_scores[0]
            b = atomic_scores[1]
            c = atomic_scores[2]
            ab = atomic_scores[3]
            abc = atomic_scores[4]
            
            #new_score= 2*max(a,b,ab) - monolithic_score     #untested, but I think is wrong
            new_score = 2*max(a,b,ab) + ab - abc    #untested

        else:
            raise ValueError(f"Unknown formula: {formula}")
        return new_score
    

    def normalize_scores(self, scores):
        """
        Normalize the scores to be in the range [0, 1].
        :param scores: List of scores to be normalized
        :return: Normalized scores
        """
        if len(scores) == 0:
            return []
        max_score = max(scores)
        if max_score == 0:
            return [0] * len(scores)
        return [score / max_score for score in scores]

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def heaviside(x):
        return 1 if x > 0 else 0
    

    def run_query_rescoring(self, qid, monolithic_scores, query_documents_scores, atoms_list, formulas, gt_pos_docs, gt_neg_docs, topk = 100, debug=False, query_debug_id = None):
        """
        Apply fuzzy logic to re-score the document based on their atomic scores.
        :param query_documents_scores: List of document atomic scores 
        :param atoms_list: List of atomic formulas  
        :param formulas: Dictionary of formulas to be evaluated
        :param topk: Number of top documents to return
        :param debug: If True, print the debug information
        :return: Dictionary of results with document indices and their scores
        
        """
        
        results = {}
        query_debug_results = {}
    
        #for formula in formulas:
        formula = 'formula1'  # Assuming we are only interested in 'formula1' for this example
        new_scores = []
        orig_scores = []
        
        #debug stats
        log = {}
        new_false_pos = []
        new_false_neg = []
        orig_false_pos = []
        orig_false_neg = []
        orig_pos_rank = 0
        orig_neg_rank = 0
        new_pos_rank = 0
        new_neg_rank = 0
        #query_documents_scores = softmax(query_documents_scores, axis=0)  # Normalize the scores
        if self.pylogic:
            f_parsed = self.parser(formulas[formula])
        for doc_id, (monolithic_score, atomic_score) in enumerate(zip(monolithic_scores, query_documents_scores)):
            query = {}
            orig_score = monolithic_score[0]
            #atomic_score = softmax(atomic_score, axis=0)  # Normalize the atomic scores
            if self.pylogic:
                # For each atomic score, we create a query with the corresponding atom
                # and its score, if the atom is in the formula
                for score, atom in zip(atomic_score, atoms_list):
                    if atom in formulas[formula]:
                        query[atom] = score
                reset_cache()
                new_score = self.evaluator(f_parsed ,0,[query])
            else:
                # If not using pylogics, we assume a simple linear combination of scores
                new_score = self.evaluator(orig_score, atomic_score, formulas[formula])
            
            orig_scores.append(orig_score)
            new_scores.append(new_score)
            if debug:
                
                log[doc_id] = ({'new_score': new_score, 'original_score': orig_score, 
                                'atoms_scores': query})
        
        rescored_docids = np.argsort(new_scores)[::-1][:topk]
        rescored_scores = np.sort(new_scores)[::-1][:topk]
        orig_docids = np.argsort(orig_scores)[::-1][:topk]
        orig_scores = np.sort(orig_scores)[::-1][:topk]
        if debug:
            topk_meta = []
            for doc_id in rescored_docids:
                topk_meta.append(log[doc_id])
            for doc_id in rescored_docids:
                if str(doc_id) not in gt_pos_docs:
                    new_false_pos.append(log[doc_id])
                    new_pos_rank = new_pos_rank + 1
                else:
                    break
            for doc_id in rescored_docids:
                if str(doc_id) not in gt_neg_docs:
                    new_false_neg.append(log[doc_id])
                    new_neg_rank = new_neg_rank + 1
                else:
                    break
            for doc_id in orig_docids:
                if str(doc_id) not in gt_pos_docs:
                    orig_false_pos.append(log[doc_id])
                    orig_pos_rank = orig_pos_rank + 1
                else:
                    break
            for doc_id in orig_docids:
                if str(doc_id) not in gt_neg_docs:
                    orig_false_neg.append(log[doc_id])
                    orig_neg_rank = orig_neg_rank + 1
                else:
                    break        
                
            query_debug_results[qid] = {'formula': formulas[formula], 'gt_pos_docs': gt_pos_docs, 'gt_neg_docs': gt_neg_docs,
                                            'new_doc_idx': rescored_docids.tolist()[:10], 
                                            #'new_scores': rescored_scores.tolist()[:10],
                                            'orig_doc_idx': orig_docids.tolist()[:10],
                                            'scores':topk_meta[:5],
                                            #'orig_scores': orig_scores.tolist()[:10], 
                                            'new_false_pos': new_false_pos[:5], 'new_false_neg': new_false_neg[:5],
                                            'orig_false_pos': orig_false_pos[:5], 'orig_false_neg': orig_false_neg[:5], 'orig_pos_rank': orig_pos_rank, 'orig_neg_rank': orig_neg_rank,
                                            'new_pos_rank': new_pos_rank, 'new_neg_rank': new_neg_rank}
            
        
        
        results[qid] = {'new_doc_idx': rescored_docids, 'new_scores': rescored_scores, 'orig_doc_idx': orig_docids, 
                            'orig_scores': orig_scores, 'new_false_pos': new_false_pos, 'new_false_neg': new_false_neg, 
                            'orig_false_pos': orig_false_pos, 'orig_false_neg': orig_false_neg}
        if debug:
            return query_debug_results
        return results
    

    def run_query_rescoring_docids(self, qid, monolithic_scores, monolitic_ids, query_documents_scores, query_documents_ids, atoms_list, formulas, gt_pos_docs, gt_neg_docs, question_embed, query_atoms_embeds, topk = 100, debug=False, query_debug_id = None, target_doc_embeds = None):
        """
        Apply fuzzy logic to re-score the document based on their atomic scores.
        :param query_documents_scores: List of document atomic scores 
        :param atoms_list: List of atomic formulas  
        :param formulas: Dictionary of formulas to be evaluated
        :param topk: Number of top documents to return
        :param debug: If True, print the debug information
        :return: Dictionary of results with document indices and their scores
        
        """
        
        results = {}
        query_debug_results = {}
    
        #for formula in formulas:
        formula = 'formula1'  # Assuming we are only interested in 'formula1' for this example
        new_scores = []
        orig_scores = []
        
        #debug stats
        log = {}
        new_false_pos = []
        new_false_neg = []
        orig_false_pos = []
        orig_false_neg = []
        orig_pos_rank = 0
        orig_neg_rank = 0
        new_pos_rank = 0
        new_neg_rank = 0
        #query_documents_scores = softmax(query_documents_scores, axis=0)  # Normalize the scores
        if self.pylogic:
            f_parsed = self.parser(formulas[formula])

        
        # Compute per-atom min and max scores across all documents for normalization
        #Find min/max per atom across all documents for normalization
        # Dict keyed by atom name -> scalar min/max across all documents
        atom_min = {atom: float('inf')  for atom in atoms_list}
        atom_max = {atom: float('-inf') for atom in atoms_list}

        for atomic_scores in query_documents_scores:
            for score, atom in zip(atomic_scores, atoms_list):
                if score < atom_min[atom]:
                    atom_min[atom] = score
                if score > atom_max[atom]:
                    atom_max[atom] = score

        # Alternatively, we can compute min/max using numpy for higher distribution of scores, getting more correct normalization (but less interpretable, as we will have one min/max for all atoms, not per atom)
        num_atoms = len(atoms_list)
        atoms_min = np.min(query_documents_scores)  # , axis=0 shape: (num_atoms,)
        atoms_max = np.max(query_documents_scores)  # , axis=0 shape: (num_atoms,)
        #print (f"Shape:{query_documents_scores.shape}, Atom min scores: {atom_min} , Atom max scores: {atom_max}")

        # --sig_gate: significance-gated conjunctive correction.
        # sigma_K = std of the existing (unconditional) AND-correction deficit
        # R_n = S_fused - max(atom scores), computed once per query over the
        # full topk_max candidate pool, before the per-document rescoring loop
        # (same place atom_max is computed for the inhibitory-gate normalization
        # above). Only meaningful for the two conjunctive templates; left None
        # (and unused) for every other formula, and whenever self.sig_gate is off.
        self._sig_gate_sigma_K = None
        if self.sig_gate and formulas[formula] in ("atom0 & atom1", "atom0 & atom1 & atom2"):
            is_and3 = (formulas[formula] == "atom0 & atom1 & atom2")
            deficits = []
            for m_score, a_scores in zip(monolithic_scores, query_documents_scores):
                s_fused = m_score[0]  # anchored on S_M, same as the real scorer (RESULTS.md 8.2)
                if is_and3:
                    s_max_atoms = max(a_scores[0], a_scores[1], a_scores[2])
                else:
                    s_max_atoms = max(a_scores[0], a_scores[1])
                deficits.append(s_fused - s_max_atoms)
            self._sig_gate_sigma_K = float(np.std(deficits))


        for (monolithic_score, monolitic_id, atomic_scores, atomic_id, target_doc_embed) in zip(monolithic_scores, monolitic_ids, query_documents_scores, query_documents_ids, target_doc_embeds):
            query = {}
            orig_score = monolithic_score[0]
            #atomic_score = softmax(atomic_score, axis=0)  # Normalize the atomic scores
            if self.pylogic:
                for score, atom in zip(atomic_scores, atoms_list):
                    if atom in formulas[formula]:
                        query[atom] = score
                reset_cache()
            
            # Normalized scores:
            normalized_atomic_scores = np.array([
                    #(score - atom_min[atom]) / (atom_max[atom] - atom_min[atom] + epsilon)
                    #(score - atom_min) / (atom_max - atom_min + epsilon)
                    #(score) / (atoms_max + epsilon)    
                    (score) / (atom_max[atom] + epsilon)
                    for score, atom in zip(atomic_scores, atoms_list)
                ])

            if self.pylogic:
                new_score = self.evaluator(f_parsed ,0,[query])
            else:
                new_score = self.evaluator(monolithic_score, atomic_scores, formulas[formula], question_embed, query_atoms_embeds, target_doc_embed, normalized_atomic_scores)
            orig_scores.append(orig_score)
            # if new_score.dtype!='float32':
            #     new_score = new_score.numpy()
            new_scores.append(new_score)
            doc_id = monolitic_id[0]
            if debug:                
                log[doc_id] = ({'new_score': new_score, 'original_score': orig_score, 
                                'atoms_scores': query})
        
        #rescored_docids = np.argsort(new_scores, kind='stable')[::-1][:topk]   
        rescored_docids = np.argsort(new_scores)[::-1][:topk]        
        rescored_docids = monolitic_ids[rescored_docids].squeeze()
        rescored_scores = np.sort(new_scores)[::-1][:topk]
        orig_docids = np.argsort(orig_scores)[::-1][:topk]
        orig_docids = monolitic_ids[orig_docids].squeeze()
        orig_scores = np.sort(orig_scores)[::-1][:topk]
        if debug:
            topk_meta = []
            for doc_id in rescored_docids:
                topk_meta.append(log[doc_id])
            for doc_id in rescored_docids:
                if str(doc_id) not in gt_pos_docs:
                    new_false_pos.append(log[doc_id])
                    new_pos_rank = new_pos_rank + 1
                else:
                    break
            for doc_id in rescored_docids:
                if str(doc_id) not in gt_neg_docs:
                    new_false_neg.append(log[doc_id])
                    new_neg_rank = new_neg_rank + 1
                else:
                    break
            for doc_id in orig_docids:
                if str(doc_id) not in gt_pos_docs:
                    orig_false_pos.append(log[doc_id])
                    orig_pos_rank = orig_pos_rank + 1
                else:
                    break
            for doc_id in orig_docids:
                if str(doc_id) not in gt_neg_docs:
                    orig_false_neg.append(log[doc_id])
                    orig_neg_rank = orig_neg_rank + 1
                else:
                    break        
        
        results[qid] = {'new_doc_idx': rescored_docids.tolist(), 'new_scores': rescored_scores, 'orig_doc_idx': orig_docids.tolist(), 
                            'orig_scores': orig_scores, 'new_false_pos': new_false_pos, 'new_false_neg': new_false_neg, 
                            'orig_false_pos': orig_false_pos, 'orig_false_neg': orig_false_neg}
        if debug:
            query_debug_results[qid] = {'formula': formulas[formula], 'gt_pos_docs': gt_pos_docs, 'gt_neg_docs': gt_neg_docs,
                                        'new_doc_idx': rescored_docids.tolist()[:10], 
                                        'new_scores': rescored_scores.tolist()[:10],
                                        'orig_doc_idx': orig_docids.tolist()[:10],
                                        'scores':topk_meta[:5],
                                        'orig_scores': orig_scores.tolist()[:10], 
                                        'new_false_pos': new_false_pos[:5], 'new_false_neg': new_false_neg[:5],
                                        'orig_false_pos': orig_false_pos[:5], 'orig_false_neg': orig_false_neg[:5], 'orig_pos_rank': orig_pos_rank, 'orig_neg_rank': orig_neg_rank,
                                        'new_pos_rank': new_pos_rank, 'new_neg_rank': new_neg_rank}            
            return query_debug_results
        return results        

    def get_formula_objective(self, formula, full_embed, atoms_embeds):
        if formula == "atom0":
            def f(x):
                """
                Objective function for a single atom.
                :param x: Embedding vector
                :return: Score based on the atom
                """
                return (x * atoms_embeds[0]).sum(axis=1)
            return batch_wrap(f)

        elif formula == "atom0 | atom1":
            def f(x):
                """
                Objective function for the OR operation between two atoms.
                :param x: Embedding vector
                :return: Score based on the OR operation

                Fidelity fix (audit item 2): use the genuine, separately-encoded
                S_AB (atoms_embeds[2]) rather than substituting S_M
                (full_embed) for it. evaluate_fuzzy_nsfl_clean's "atom0 | atom1"
                branch uses max(S_M, S_AB, S_A, S_B) with S_AB and S_M as two
                distinct terms - the prior version of this objective collapsed
                them into one (mislabeling S_M as "ab"), silently dropping the
                genuine S_AB term entirely. S_M is still included, as its own
                term, to match the real scorer exactly.
                """
                a = (x * atoms_embeds[0]).sum(axis=1)
                b = (x * atoms_embeds[1]).sum(axis=1)
                ab = (x * atoms_embeds[2]).sum(axis=1)
                sm = (x * full_embed).sum(axis=1)
                return torch.where(
                    (ab < a) & (ab < b),
                    ab,
                    torch.max(torch.stack([sm, ab, a, b]), axis=0)[0]
                )
            return batch_wrap(f)

        elif formula == "atom0 | atom1 | atom2":
            def f(x):
                """
                Objective function for the OR operation between three atoms.
                :param x: Embedding vector
                :return: Score based on the OR operation

                Fidelity fix (audit item 3): restore all 8 terms
                evaluate_fuzzy_nsfl_clean's "atom0 | atom1 | atom2" branch
                uses (S_M, S_ABC, S_A, S_B, S_C, S_AB, S_AC, S_BC). The prior
                version of this objective used only 4 (S_A, S_B, S_C, and S_M
                mislabeled "ab"), dropping S_AB, S_AC, S_BC, and S_ABC
                entirely. No boundary/stability override exists for this
                formula in the real scorer either, so none is added here.
                """
                a = (x * atoms_embeds[0]).sum(axis=1)
                b = (x * atoms_embeds[1]).sum(axis=1)
                c = (x * atoms_embeds[2]).sum(axis=1)
                ab = (x * atoms_embeds[3]).sum(axis=1)
                ac = (x * atoms_embeds[4]).sum(axis=1)
                bc = (x * atoms_embeds[5]).sum(axis=1)
                abc = (x * atoms_embeds[6]).sum(axis=1)
                sm = (x * full_embed).sum(axis=1)
                return torch.max(torch.stack([sm, abc, a, b, c, ab, ac, bc]), axis=0)[0]
            return batch_wrap(f)

        elif formula == "atom0 & atom1":
            def f(x):
                """
                Objective function for the AND operation between two atoms.
                :param x: Embedding vector
                :return: Score based on the AND operation
                """
                a = (x * atoms_embeds[0]).sum(axis=1)
                b = (x * atoms_embeds[1]).sum(axis=1)
                full = (x * full_embed).sum(axis=1)
                
                # return 2*full - torch.max(torch.stack([a,b]),axis=0)[0]  # Custom AND T-NORM, without SIGLIP/Conjecture-2
                # return full if (full > a + b) else 2 * full - torch.max(torch.stack([a, b]), axis=0)[0]   #RuntimeError: Boolean value of Tensor with more than one value is ambiguous
                
                # Assuming 'full', 'a', and 'b' are tensors of the same shape
                condition = full > (a + b)

                # The 'else' part of your logic
                # We use .values if using torch.max on a stack, 
                # or simply torch.maximum(a, b) for cleaner element-wise max
                alternate = 2 * full - torch.maximum(a, b)

                return torch.where(condition, full, alternate)

            return batch_wrap(f)

        elif formula == "atom0 & atom1 & atom2":
            def f(x):
                """
                Objective function for the AND operation between three atoms.
                :param x: Embedding vector
                :return: Score based on the AND operation
                """
                a = (x * atoms_embeds[0]).sum(axis=1)
                b = (x * atoms_embeds[1]).sum(axis=1)
                c = (x * atoms_embeds[2]).sum(axis=1)
                abc = (x * full_embed).sum(axis=1)
                # return 2*abc - torch.max(torch.stack([a, b, c]),axis=0)[0]  # Custom AND T-NORM
                return torch.where(
                    abc > a + b + c,
                    abc,
                    abc + 1/2 * (abc - torch.max(torch.stack([a, b, c]), axis=0)[0])
                )

            return batch_wrap(f)

        elif formula == "atom0 & !atom1":   
            def f(x):
                """
                Objective function for the AND NOT B operation between three atoms.
                :param x: Embedding vector
                :return: Score based on the AND operation
                """
                a = (x * atoms_embeds[0]).sum(axis=1)
                b = (x * atoms_embeds[1]).sum(axis=1)
                ab = (x * atoms_embeds[2]).sum(axis=1)  

                # Create the condition mask
                # We use & instead of 'and' for element-wise logical AND
                condition = (a > ab) & (b > ab)   
                
                # Define the 'else' result
                else_result = b * (2 * a - ab) + (1 - b) * a

                # return b*(2*a-ab) + (1-b)*a     
                # return ab if (a > ab and b > ab) else b * (2 * a - ab) + (1 - b) * a      # RuntimeError: Boolean value of Tensor with more than one value is ambiguous
                return torch.where(condition, ab, else_result)
            
            return batch_wrap(f)

        elif formula == "atom0 & atom1 & !atom2":
            def f(x):
                """
                Objective function for the AND AND NOT B operation between three atoms.
                :param x: Embedding vector
                :return: Score based on the AND operation

                Fidelity fix (audit item 1): abc must read atoms_embeds[6]
                (S_ABC), not atoms_embeds[4]. Per the [a,b,c,ab,ac,bc,abc]
                construction order (encode_queries_embedding in
                retrieve_fuzzy.py), index 4 is S_AC, not S_ABC - the prior
                version of this objective was silently optimizing against the
                wrong fused embedding (S_AC in place of S_ABC) for this
                formula. evaluate_fuzzy_nsfl_clean's "atom0 & atom1 & !atom2"
                branch uses atomic_scores[3] (S_AB) and atomic_scores[6]
                (S_ABC) - matched here.

                NOTE (audit item 4, deliberately NOT fixed - see report): the
                real scorer's coefficient here is cN, the pool-max-normalized
                S_C (normalized_atomic_scores[2]), not raw S_C. This objective
                still uses raw c. Threading in the pool-max normalizer is
                blocked structurally, not just unwired - see the audit
                report for why - and is left as-is pending a decision.
                """
                a = (x * atoms_embeds[0]).sum(axis=1)
                b = (x * atoms_embeds[1]).sum(axis=1)
                c = (x * atoms_embeds[2]).sum(axis=1)
                ab = (x * atoms_embeds[3]).sum(axis=1)
                abc = (x * atoms_embeds[6]).sum(axis=1)
                # return b*(2*ab - abc) + (1-b)*(2*ab - torch.max(torch.stack([a,b]),axis=0)[0])  # Custom AND T-NORM

                a_and_b = torch.where(
                    ab > (a + b),
                    ab,
                    2 * ab - torch.max(torch.stack([a, b]), axis=0)[0]
                )
                # Wrong 2*abc (April9th)  #return a_and_b - c * (abc - 2 * ab) #biggest change is c coefficient, which is more correct than b, but results are less good
                return a_and_b - c * (abc - ab) #biggest change is c coefficient, which is more correct than b, but results are less good
            return batch_wrap(f)

        elif formula == "atom0 | atom1 & !atom2":
            def f(x):
                """
                Objective function for the XOR operation between three atoms.
                :param x: Embedding vector
                :return: Score based on the AND operation

                Fidelity fix (audit item 5): restore S_ABC (atoms_embeds[6])
                in place of S_M (full_embed), and restore the "+ab" term that
                was dropped. evaluate_fuzzy_nsfl_clean's
                "atom0 | atom1 & !atom2" branch computes
                2*max(a,b,ab) + ab - abc (genuine S_ABC); the prior version of
                this objective computed 2*max(a,b,ab) - S_M, both substituting
                the wrong term for abc and dropping the +ab term entirely.
                This formula is not used in QUEST (rare edge case).
                """
                a = (x * atoms_embeds[0]).sum(axis=1)
                b = (x * atoms_embeds[1]).sum(axis=1)
                ab = (x * atoms_embeds[3]).sum(axis=1)
                abc = (x * atoms_embeds[6]).sum(axis=1)
                return 2 * torch.max(torch.stack([a, b, ab]), axis=0)[0] + ab - abc
            return batch_wrap(f)

        else:
            raise ValueError(f"Unknown formula: {formula}")
        
