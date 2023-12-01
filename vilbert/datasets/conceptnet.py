from selectors import EpollSelector
import networkx as nx
import nltk
import json
import math
from tqdm import tqdm
import numpy as np
import sys
import itertools
import dgl
import torch 
import random
import pandas as pd
import requests
from os import path

__all__ = ['extract_english', 'construct_graph', 'merged_relations']

relation_groups = [
    'atlocation/locatednear',
    'capableof',
    'causes/causesdesire/*motivatedbygoal',
    'createdby',
    'desires',
    'antonym/distinctfrom',
    'hascontext',
    'hasproperty',
    'hassubevent/hasfirstsubevent/haslastsubevent/hasprerequisite/entails/mannerof',
    'isa/instanceof/definedas',
    'madeof',
    'notcapableof',
    'notdesires',
    'partof/*hasa',
    'relatedto/similarto/synonym',
    'usedfor',
    'receivesaction',
]
sorted_relations = [
    'isa',
    'atlocation',
    'usedfor',
    'hascontext',
    'derivedfrom',
    'synonym',
    'mannerof',
    'hasproperty',
    'desires',
    'antonym',
    'capableof',
    'partof',
    'relatedto',
]
rel2pri = {w: i for i, w in enumerate(sorted_relations)}
def rel2prior(rel):
    try:
        return rel2pri[rel]
    except:
        return 13
    


merged_relations = [
    'antonym',
    'atlocation',
    'capableof',
    'causes',
    'createdby',
    'derivedfrom',
    'desires',
    'hasa',
    'hascontext',
    'hasproperty',
    'hassubevent',
    'isa',
    'madeof',
    'mannerof',
    'notcapableof',
    'notdesires',
    'partof',
    'receivesaction',
    'relatedto',
    'synonym',
    'usedfor',
]

rcnnrel_to_concept_relations = {
    "above":"atlocation", "across":"atlocation", "against":"atlocation", "along":"atlocation", "at":"atlocation", "attached to":"partof",\
    "behind":"atlocation", "belonging to":"partof", "between":"atlocation", "has":"hasa", "holding":"hasproperty", "in":"hassubevent", \
    "in front of":"atlocation", "made of":"madeof", "near":"atlocation", "of": "partof", "part of":"partof", "under":"atlocation", "using":"usedfor", "with":"hasproperty"
}

relation_text = [
    'is the antonym of',
    'is at location of',
    'is capable of',
    'causes',
    'is created by',
    'derived from',
    'desires',
    'has a',
    'has context',
    'has property',
    'has subevent',
    'is a',
    'is made of',
    'is a manner of',
    'is not capable of',
    'does not desires',
    'is part of',
    'receives action',
    'is related to',
    'is the synonym of',
    'is used for',
]

ans2ent = pd.read_csv('/data/gjr/OK-VQA/myAnnotations/answer2entities.csv', delimiter=',')

cpnet_vocab_path = '/data/gjr/ConceptNet/numberbatch-19.08-entity.txt'
concept2id = {}
id2concept = []
with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
    id2concept = [w.strip() for w in fin]
    print('id2conceptnet load finished')


concept2id = {w: i for i, w in enumerate(id2concept)}

id2relation = merged_relations
relation2id = {r: i for i, r in enumerate(id2relation)}

import nltk
nltk_stopwords = nltk.corpus.stopwords.words('english')
nltk_stopwords += ["like", "gone", "did", "going", "would", "could",
                    "get", "in", "up", "may", "wanter"]  # issue: mismatch with the stop words in grouding.py
blacklist = set(["uk", "us", "take", "make", "object", "person", "people"])  # issue: mismatch with the blacklist in grouding.py
all_neighbor_codes = {}
neighbor_dict_path = "/data/gjr/ConceptNet/All_Neighbor_Codes.json"
if path.isfile(neighbor_dict_path):
    all_neighbor_codes = json.load(open(neighbor_dict_path))
    all_neighbor_codes = {int(key): {v[1]: [v[0].lower(),v[1],v[2]] for v in value}  for key, value in all_neighbor_codes.items()}

def compute_all_neighbors():
    df = pd.read_csv('/data/gjr/ConceptNet/conceptnet-assertions-5.7.0.csv.gz', header=None, sep='\t', compression='gzip', usecols=[1,2,3,4])
    df = df.loc[(df[2].str.startswith('/c/en/')) & (df[3].str.startswith('/c/en/')) ]
    df[1] = df[1].map(lambda x: x.replace('/r/', ''))
    df[2] = df[2].map(lambda x: x.split('/')[3])
    df[3] = df[3].map(lambda x: x.split('/')[3])
    df = df.loc[df[2].isin(concept2id)]
    df = df.loc[df[3].isin(concept2id)]
    df[4] = df[4].map(lambda x: json.loads(x)["weight"])
    df_mirror = df.copy()
    df_mirror = df_mirror.rename(columns={2: 3, 3: 2})
    df[5] = df[2].map(lambda x: concept2id[x]).astype(int)
    df_mirror[5] = df_mirror[2].map(lambda x: concept2id[x]).astype(int)
    df_full = pd.concat([df, df_mirror])
    all_neighbor_codes = df_full.groupby(5)[[1, 3, 4]].apply(lambda g: g.values.tolist()).to_dict()
    # add nodes that are not in conceptnet-assertions-5.7.0.csv.gz
    # for node in tqdm(concept2id.keys()):
    #     code = concept2id[node]
    #     if code in all_neighbor_codes:
    #         continue 
    #     else:
    #         all_neighbor_codes[code] = request_connect_nodes(node)

    # Merge Two Dict
    # all_neighbor_codes1 = json.load(open("/data/gjr/ConceptNet/All_Neighbor_Codes_newdf.json"))
    # all_neighbor_codes1 = {int(key): value  for key, value in all_neighbor_codes1.items()}
    # all_neighbor_codes2 = json.load(open("/data/gjr/ConceptNet/All_Neighbor_Codes_1.json"))
    # all_neighbor_codes2 = {int(key): [[v[0].lower(),v[1],v[2]] for v in value.values()]  for key, value in all_neighbor_codes2.items()}
    # for key, value in all_neighbor_codes1.items():
    #     if key in all_neighbor_codes2:
    #         all_neighbor_codes1[key].extend(all_neighbor_codes2[key])
    #         kk = all_neighbor_codes1[key]
    #         kk.sort()
    #         all_neighbor_codes1[key] = list(k for k,_ in itertools.groupby(kk))


    # json.dump(all_neighbor_codes1, open("/data/gjr/ConceptNet/All_Neighbor_Codes.json", "w"))
    

def request_connect_nodes(key):
    try:
        obj = requests.get('http://api.conceptnet.io/query?node=/c/en/%s&other=/c/en&limit=1000'% key).json()
        if 'error' in  obj:
            return []
    except Exception as e:
        print(e)
        return []
    nodes = [edge['start']['@id'] if key in edge['end']['@id'] else edge['end']['@id'] for edge in obj['edges']]
    nodes = [x.split('/')[3] for x in nodes]
    nodes_weight = [[edge['rel']['@id'].replace('/r/','').lower(), round(edge['weight'], 2)] for edge in obj['edges']]
    node2relweight = dict(zip(nodes[::-1], nodes_weight[::-1]))
    nodes = list(filter(lambda x: x in concept2id, nodes))
    nodes = list(set(nodes))
    results = {n: [node2relweight[n][0], n, node2relweight[n][1]] for n in nodes}
    return results

def load_merge_relation():
    relation_mapping = dict()
    for line in relation_groups:
        ls = line.strip().split('/')
        rel = ls[0]
        for l in ls:
            if l.startswith("*"):
                relation_mapping[l[1:]] = "*" + rel
            else:
                relation_mapping[l] = rel
    return relation_mapping


def del_pos(s):
    """
    Deletes part-of-speech encoding from an entity string, if present.
    :param s: Entity string.
    :return: Entity string with part-of-speech encoding removed.
    """
    if s.endswith("/n") or s.endswith("/a") or s.endswith("/v") or s.endswith("/r"):
        s = s[:-2]
    return s


def extract_english(conceptnet_path, output_csv_path, output_vocab_path):
    """
    Reads original conceptnet csv file and extracts all English relations (head and tail are both English entities) into
    a new file, with the following format for each line: <relation> <head> <tail> <weight>.
    :return:
    """
    print('extracting English concepts and relations from ConceptNet...')
    relation_mapping = load_merge_relation()
    num_lines = sum(1 for line in open(conceptnet_path, 'r', encoding='utf-8'))
    cpnet_vocab = []
    concepts_seen = set()
    with open(conceptnet_path, 'r', encoding="utf8") as fin, \
            open(output_csv_path, 'w', encoding="utf8") as fout:
        for line in tqdm(fin, total=num_lines):
            toks = line.strip().split('\t')
            if toks[2].startswith('/c/en/') and toks[3].startswith('/c/en/'):
                """
                Some preprocessing:
                    - Remove part-of-speech encoding.
                    - Split("/")[-1] to trim the "/c/en/" and just get the entity name, convert all to 
                    - Lowercase for uniformity.
                """
                rel = toks[1].split("/")[-1].lower()
                head = del_pos(toks[2]).split("/")[-1].lower()
                tail = del_pos(toks[3]).split("/")[-1].lower()

                if not head.replace("_", "").replace("-", "").isalpha():
                    continue
                if not tail.replace("_", "").replace("-", "").isalpha():
                    continue
                if rel not in relation_mapping:
                    continue

                rel = relation_mapping[rel]
                if rel.startswith("*"):
                    head, tail, rel = tail, head, rel[1:]

                data = json.loads(toks[4])

                fout.write('\t'.join([rel, head, tail, str(data["weight"])]) + '\n')

                for w in [head, tail]:
                    if w not in concepts_seen:
                        concepts_seen.add(w)
                        cpnet_vocab.append(w)

    with open(output_vocab_path, 'w') as fout:
        for word in cpnet_vocab:
            fout.write(word + '\n')

    print(f'extracted ConceptNet csv file saved to {output_csv_path}')
    print(f'extracted concept vocabulary saved to {output_vocab_path}')
    print()


def construct_graph(cpnet_csv_path, cpnet_vocab_path, output_path, prune=True):
    print('generating ConceptNet graph file...')

    nltk.download('stopwords', quiet=True)
    nltk_stopwords = nltk.corpus.stopwords.words('english')
    nltk_stopwords += ["like", "gone", "did", "going", "would", "could",
                       "get", "in", "up", "may", "wanter"]  # issue: mismatch with the stop words in grouding.py

    blacklist = set(["uk", "us", "take", "make", "object", "person", "people"])  # issue: mismatch with the blacklist in grouding.py

    concept2id = {}
    id2concept = {}
    with open(cpnet_vocab_path, "r", encoding="utf8") as fin:
        id2concept = [w.strip() for w in fin]
    concept2id = {w: i for i, w in enumerate(id2concept)}

    id2relation = merged_relations
    relation2id = {r: i for i, r in enumerate(id2relation)}

    graph = nx.MultiDiGraph()
    nrow = sum(1 for _ in open(cpnet_csv_path, 'r', encoding='utf-8'))
    with open(cpnet_csv_path, "r", encoding="utf8") as fin:

        def not_save(cpt):
            if cpt in blacklist:
                return True
            '''originally phrases like "branch out" would not be kept in the graph'''
            # for t in cpt.split("_"):
            #     if t in nltk_stopwords:
            #         return True
            return False

        attrs = set()

        for line in tqdm(fin, total=nrow):
            ls = line.strip().split('\t')
            rel = relation2id[ls[0]]
            subj = concept2id[ls[1]]
            obj = concept2id[ls[2]]
            weight = float(ls[3])
            if prune and (not_save(ls[1]) or not_save(ls[2]) or id2relation[rel] == "hascontext"):
                continue
            # if id2relation[rel] == "relatedto" or id2relation[rel] == "antonym":
            # weight -= 0.3
            # continue
            if subj == obj:  # delete loops
                continue
            # weight = 1 + float(math.exp(1 - weight))  # issue: ???

            if (subj, obj, rel) not in attrs:
                graph.add_edge(subj, obj, rel=rel, weight=weight)
                attrs.add((subj, obj, rel))
                graph.add_edge(obj, subj, rel=rel + len(relation2id), weight=weight)
                attrs.add((obj, subj, rel + len(relation2id)))

    nx.write_gpickle(graph, output_path)
    print(f"graph file saved to {output_path}")
    print()


def glove_init(input, output, concept_file):
    embeddings_file = output + '.npy'
    vocabulary_file = output.split('.')[0] + '.vocab.txt'
    output_dir = '/'.join(output.split('/')[:-1])
    output_prefix = output.split('/')[-1]

    words = []
    vectors = []
    print("loading embedding")
    with open(input, 'rb') as f:
        for line in f:
            fields = line.split()
            if len(fields) <= 2:
                continue
            word = fields[0].decode('utf-8')
            words.append(word)
            vector = np.fromiter((float(x) for x in fields[1:]),
                                 dtype=np.float)

            vectors.append(vector)
        dim = vector.shape[0]
    print("converting")
    matrix = np.array(vectors, dtype="float32")
    print("writing")
    np.save(embeddings_file, matrix)
    text = '\n'.join(words)
    with open(vocabulary_file, 'wb') as f:
        f.write(text.encode('utf-8'))

    def load_glove_from_npy(glove_vec_path, glove_vocab_path):
        vectors = np.load(glove_vec_path)
        with open(glove_vocab_path, "r", encoding="utf8") as f:
            vocab = [l.strip() for l in f.readlines()]

        assert (len(vectors) == len(vocab))

        glove_embeddings = {}
        for i in range(0, len(vectors)):
            glove_embeddings[vocab[i]] = vectors[i]
        print("Read " + str(len(glove_embeddings)) + " glove vectors.")
        return glove_embeddings

    def weighted_average(avg, new, n):
        # TODO: maybe a better name for this function?
        return ((n - 1) / n) * avg + (new / n)

    def max_pooling(old, new):
        # TODO: maybe a better name for this function?
        return np.maximum(old, new)

    def write_embeddings_npy(embeddings, embeddings_cnt, npy_path, vocab_path):
        words = []
        vectors = []
        for key, vec in embeddings.items():
            words.append(key)
            vectors.append(vec)

        matrix = np.array(vectors, dtype="float32")
        print(matrix.shape)

        print("Writing embeddings matrix to " + npy_path, flush=True)
        np.save(npy_path, matrix)
        print("Finished writing embeddings matrix to " + npy_path, flush=True)

        if not check_file(vocab_path):
            print("Writing vocab file to " + vocab_path, flush=True)
            to_write = ["\t".join([w, str(embeddings_cnt[w])]) for w in words]
            with open(vocab_path, "w", encoding="utf8") as f:
                f.write("\n".join(to_write))
            print("Finished writing vocab file to " + vocab_path, flush=True)

    def create_embeddings_glove(pooling="max", dim=100):
        print("Pooling: " + pooling)

        with open(concept_file, "r", encoding="utf8") as f:
            triple_str_json = json.load(f)
        print("Loaded " + str(len(triple_str_json)) + " triple strings.")

        glove_embeddings = load_glove_from_npy(embeddings_file, vocabulary_file)
        print("Loaded glove.", flush=True)

        concept_embeddings = {}
        concept_embeddings_cnt = {}
        rel_embeddings = {}
        rel_embeddings_cnt = {}

        for i in tqdm(range(len(triple_str_json))):
            data = triple_str_json[i]

            words = data["string"].strip().split(" ")

            rel = data["rel"]
            subj_start = data["subj_start"]
            subj_end = data["subj_end"]
            obj_start = data["obj_start"]
            obj_end = data["obj_end"]

            subj_words = words[subj_start:subj_end]
            obj_words = words[obj_start:obj_end]

            subj = " ".join(subj_words)
            obj = " ".join(obj_words)

            # counting the frequency (only used for the avg pooling)
            if subj not in concept_embeddings:
                concept_embeddings[subj] = np.zeros((dim,))
                concept_embeddings_cnt[subj] = 0
            concept_embeddings_cnt[subj] += 1

            if obj not in concept_embeddings:
                concept_embeddings[obj] = np.zeros((dim,))
                concept_embeddings_cnt[obj] = 0
            concept_embeddings_cnt[obj] += 1

            if rel not in rel_embeddings:
                rel_embeddings[rel] = np.zeros((dim,))
                rel_embeddings_cnt[rel] = 0
            rel_embeddings_cnt[rel] += 1

            if pooling == "avg":
                subj_encoding_sum = sum([glove_embeddings.get(word, np.zeros((dim,))) for word in subj])
                obj_encoding_sum = sum([glove_embeddings.get(word, np.zeros((dim,))) for word in obj])

                if rel in ["relatedto", "antonym"]:
                    # Symmetric relation.
                    rel_encoding_sum = sum([glove_embeddings.get(word, np.zeros((dim,))) for word in
                                            words]) - subj_encoding_sum - obj_encoding_sum
                else:
                    # Asymmetrical relation.
                    rel_encoding_sum = obj_encoding_sum - subj_encoding_sum

                subj_len = subj_end - subj_start
                obj_len = obj_end - obj_start

                subj_encoding = subj_encoding_sum / subj_len
                obj_encoding = obj_encoding_sum / obj_len
                rel_encoding = rel_encoding_sum / (len(words) - subj_len - obj_len)

                concept_embeddings[subj] = subj_encoding
                concept_embeddings[obj] = obj_encoding
                rel_embeddings[rel] = weighted_average(rel_embeddings[rel], rel_encoding, rel_embeddings_cnt[rel])

            elif pooling == "max":
                subj_encoding = np.amax([glove_embeddings.get(word, np.zeros((dim,))) for word in subj_words], axis=0)
                obj_encoding = np.amax([glove_embeddings.get(word, np.zeros((dim,))) for word in obj_words], axis=0)

                mask_rel = []
                for j in range(len(words)):
                    if subj_start <= j < subj_end or obj_start <= j < obj_end:
                        continue
                    mask_rel.append(j)
                rel_vecs = [glove_embeddings.get(words[i], np.zeros((dim,))) for i in mask_rel]
                rel_encoding = np.amax(rel_vecs, axis=0)

                # here it is actually avg over max for relation
                concept_embeddings[subj] = max_pooling(concept_embeddings[subj], subj_encoding)
                concept_embeddings[obj] = max_pooling(concept_embeddings[obj], obj_encoding)
                rel_embeddings[rel] = weighted_average(rel_embeddings[rel], rel_encoding, rel_embeddings_cnt[rel])

        print(str(len(concept_embeddings)) + " concept embeddings")
        print(str(len(rel_embeddings)) + " relation embeddings")

        write_embeddings_npy(concept_embeddings, concept_embeddings_cnt, f'{output_dir}/concept.{output_prefix}.{pooling}.npy',
                             f'{output_dir}/concept.glove.{pooling}.txt')
        write_embeddings_npy(rel_embeddings, rel_embeddings_cnt, f'{output_dir}/relation.{output_prefix}.{pooling}.npy',
                             f'{output_dir}/relation.glove.{pooling}.txt')

    create_embeddings_glove(dim=dim)

def lookup_answer_entity(answer=""):
    # case 1 use surface form
    # case 2 look up the answer
    entity = answer
    ans_df = ans2ent[ans2ent["gt"]==answer]
    entity = ans_df['concept_id'].values[0][6:] if ans_df['concept_id'].notna().values.any()==True else \
                ans_df['relate_concept_id'].values[0][6:] if ans_df['relate_concept_id'].notna().values.any()==True else answer.replace(' ','_')
    print(answer, entity)
    return entity

def not_save(cpt):
    if cpt in blacklist:
        return True
    '''originally phrases like "branch out" would not be kept in the graph'''
    # for t in cpt.split("_"):
    #     if t in nltk_stopwords:
    #         return True
    return False

pronoun_list = set(["a", "the", "my", "you", "it", "its", "your", "i", "he", "she", "his", "her", "they", "them", "their", "our", "we"])
def rectify_entity(doc):
    doc = doc.split("_")
    for i, word in enumerate(doc):
        if word in pronoun_list:
            doc.pop(i)
    return "_".join(doc)

def parse_rel(word):
    try: rel = relation2id[word]
    except: 
        try: rel = relation2id[rcnnrel_to_concept_relations[word]]
        except: rel = relation2id['relatedto']
    return rel

def parse_node(w):
    if not_save(w):
        return -1
    try:
        w = concept2id[w]
    except: w = -1
    return w

def get_code_neighbors(code):
    node = id2concept[code]
    if code in all_neighbor_codes:
        neighbors = all_neighbor_codes[code]
    else:
        return []
    for ignore_w in ['internet_slang', 'slang', 'nautical']:
        if ignore_w in neighbors:
            _ = neighbors.pop(ignore_w)
    neighbor_list = sorted(neighbors.values(), key=lambda x:x[2],reverse=True)
    neighbor_list = [x for x in neighbor_list if float(x[2]) >= 0.05]
    return neighbor_list


def get_code_neighbor_rels(code):
    neighbors = get_code_neighbors(code)
    # neighbor_codes = [concept2id[n[1]] for n in neighbors if n[1] in concept2id]
    neighbor_rels = {concept2id[n[1]]:n[0] for n in neighbors if n[1] in concept2id}
    return neighbor_rels

def get_code_neighbor_codes(code):
    neighbors = get_code_neighbors(code)
    neighbor_codes = [concept2id[n[1]] for n in neighbors if n[1] in concept2id]
    # neighbor_rels = {concept2id[n[1]]:n[0] for n in neighbors if n[1] in concept2id}
    return neighbor_codes


def find_gt(question_id, concept_strs, q_chunks, gt_codes):
    all_attrs = set()
    all_ents = set()
    q_ents = set()
    v_ents = set()
    dist = 100
    flag = 'none'
    rel = -1
    rel_dict = {}
    answer= '...'
    
    # Step 1. Add edges and nodes from scene graph triples
    for line in concept_strs:
        # ls = [x.lower().replace(" ","_") for x in line.split(' [SEP] ')[1].split('<SEP>')] # for mavex knowledge
        # For Scene Graphs
        score = float(line.split(':')[0])
        ls = [x.lower().replace(" ","_") for x in line.split(':')[1].split('<SEP>')] 
        ls = [rectify_entity(c) for c in ls]
        # parse relation
        # parse subject and object
        subj = parse_node(ls[0])
        obj = parse_node(ls[2])
        for w in [subj, obj]:
            if w != -1:
                all_ents.add(w)
                v_ents.add(w)
            
    for q_node in q_chunks:
        try:
            q_code = concept2id[q_node]
        except: continue
        q_ents.add(q_code)
        all_ents.add(q_code)
    
    if len(all_ents & set(gt_codes)) > 0:
        dist = 0
    else:
        rels = [get_code_neighbor_rels(c) for c in gt_codes]
        for r in rels: 
            rel_dict.update(r) 
        neighbor_codes = set(sum([list(r.keys()) for r in rels], []))
        if len(q_ents & neighbor_codes) > 0:
            rel = rel_dict[list(q_ents & neighbor_codes)[0]]
            answer = list(q_ents & neighbor_codes)
            dist = 1
            flag = 'q'
        elif len(v_ents & neighbor_codes) > 0:
            rel = rel_dict[list(v_ents & neighbor_codes)[0]]
            answer = list(v_ents & neighbor_codes)
            dist = 1
            flag = 'v'
        else:
            rels = [get_code_neighbor_rels(c) for c in neighbor_codes]
            for r in rels: 
                rel_dict.update(r) 
            twohop_neighbor_codes = set(sum([list(r.keys()) for r in rels], []))
            if len(q_ents & twohop_neighbor_codes) > 0:
                rel = rel_dict[list(q_ents & twohop_neighbor_codes)[0]]
                answer = list(q_ents & twohop_neighbor_codes)
                dist = 2
                flag = 'q'
            elif len(v_ents & twohop_neighbor_codes) > 0:
                rel = rel_dict[list(v_ents & twohop_neighbor_codes)[0]]
                answer = list(v_ents & twohop_neighbor_codes)
                dist = 1
                flag = 'v'

    return dist, flag, rel

def build_graph(question_id, concept_strs, q_chunks, cand_answers, max_size=100):
    graph = nx.DiGraph()
    all_attrs = set()
    all_ents = set()

    
    # Step 1. Add edges and nodes from scene graph triples
    for line in concept_strs:
        # ls = [x.lower().replace(" ","_") for x in line.split(' [SEP] ')[1].split('<SEP>')] # for mavex knowledge
        # For Scene Graphs
        score = float(line.split(':')[0])
        ls = [x.lower().replace(" ","_") for x in line.split(':')[1].split('<SEP>')] 
        ls = [rectify_entity(c) for c in ls]
        # parse relation
        rel = parse_rel(ls[1])
        # parse subject and object
        subj = parse_node(ls[0])
        obj = parse_node(ls[2])
        for w in [subj, obj]:
            if w != -1:
                all_ents.add(w)
                graph.add_node(w, id=w, v=1, q=0, a=0, n=0)
        if subj != -1 and obj != -1:
            weight = score # 1.0
            # if (not_save(ls[1]) or not_save(ls[2]) or id2relation[rel] == "hascontext"): continue
            # if id2relation[rel] == "relatedto" or id2relation[rel] == "antonym": continue
            if subj == obj:  continue # delete loops
            # bidirectional edges
            if (subj, obj, rel) not in all_attrs:
                graph.add_edge(subj, obj, rel=rel, weight=weight)
                all_attrs.add((subj, obj, rel))
                graph.add_edge(obj, subj, rel=rel + len(relation2id), weight=weight)
                all_attrs.add((obj, subj, rel + len(relation2id)))

    # Step 2. Add question nodes
    for q_node in q_chunks:
        try:
            q_code = concept2id[q_node]
            if q_code not in all_ents:
                all_ents.add(q_code)
                graph.add_node(q_code, id=q_code, q=1, v=0, a=0, n=0)
            else:
                graph.nodes[q_code]['q']=1

            if check_question(question_id):
                print(q_node, q_code)
        except: continue


    # Step 3.1. Add onehop nodes for all nodes that are q & v 
    qv_codes = [x for x, y in graph.nodes(data=True) if (y['q'] == 1 and y['v'] == 1)]
    qv_nodes = [id2concept[x] for x in qv_codes]
    all_ents_node = [id2concept[x] for x in all_ents]

    # Use Invert Index get frequent nodes that connect nodes in q or v
    onehop_dict = {}
    onehop_anchors = all_ents
    for code in list(set( onehop_anchors)):
        node = id2concept[code]
        neighbors = get_code_neighbors(code)
        for neighbor in neighbors:
            rel, n_node, weight = neighbor
            try:
                n_code = concept2id[n_node]
            except: continue
            if n_code not in onehop_dict:
                onehop_dict[n_code] = [[node], [weight], [rel], [code]]
            else:
                onehop_dict[n_code][0].append(node)
                onehop_dict[n_code][1].append(weight)
                onehop_dict[n_code][2].append(rel)
                onehop_dict[n_code][3].append(code)

    # keep onehop nodes connected by at least one q_node
    onehop_nodes = {k:v for k, v in onehop_dict.items() if k not in all_ents_node}
    onehop_nodes = sorted(onehop_nodes.items(), key=lambda x:sum(x[1][1]), reverse=True)
    onehop_nodes = sorted(onehop_nodes, key=lambda x:sum([rel2prior(u) for u in x[1][2]]) / len(x[1][2]))
    onehop_nodes = sorted(onehop_nodes, key=lambda x:len(x[1][0]), reverse=True)
    onehop_nodes = sorted(onehop_nodes, key=lambda x:len(set(x[1][0]) & set(qv_nodes)), reverse=True)
    onehop_nodes = sorted(onehop_nodes, key=lambda x:len(set(x[1][0]) & set(q_chunks)), reverse=True)
    onehop_nodes = [x[0] for x in onehop_nodes]
    # onehop_nodes = onehop_nodes[:(2 * max_size // 3)]

    for new_code in onehop_nodes:
        if graph.number_of_nodes() >= (max_size // 2): # 1000
            break
        if new_code not in all_ents:
            all_ents.add(new_code)
            graph.add_node(new_code, id=new_code, a=0, q=0, v=0, n=1)
        else:
            graph.nodes[new_code]['n']=1

    for key_code, value in onehop_dict.items():
        if key_code in all_ents:
            n, weights, rels, codes = value
            for w,r,c in zip(weights, rels, codes):
                if c not in all_ents: continue
                r = parse_rel(r)
                graph.add_edge(c, key_code, rel=r, weight=float(w))
                all_attrs.add((c, key_code, r))
                # graph.add_edge(key_code, c, rel=r + len(relation2id), weight=float(w))
                # all_attrs.add((key_code, c, r + len(relation2id)))

    # if check_question(question_id):
    #     # print([(i, x[1][0] , x[0]) for i,x in enumerate(onehop_nodes)])
    #     for x in onehop_nodes[:50]: # 50
    #         print( x[1][0], ' -------> ', x[0])

    for pair in itertools.permutations(list(all_ents), r=2):
        code_a, code_b = list(pair)
        if (code_a, code_b, relation2id['relatedto']) not in all_attrs:
            node_b = id2concept[code_b]
            try:
                if node_b in all_neighbor_codes[code_a]:
                    rel = parse_rel(all_neighbor_codes[code_a][node_b][0])
                    wgt = all_neighbor_codes[code_a][node_b][2]
                    graph.add_edge(code_a, code_b, rel=rel, weight=wgt)
                    all_attrs.add((code_a, code_b, rel))
                    # graph.add_edge(code_b, code_a, rel=rel + len(relation2id), weight=wgt)
                    # all_attrs.add((code_b, code_a, rel + len(relation2id)))
            except: continue



    # Step 3.2. Add two hop answers
    twohop_dict = {}
    for code in list(all_ents):
        node = id2concept[code]
        neighbors = get_code_neighbors(code)
        for neighbor in neighbors:
            rel, n_node, weight = neighbor
            try:
                n_code = concept2id[n_node]
            except: continue
            if n_code not in twohop_dict:
                twohop_dict[n_code] = [[node], [weight], [rel], [code]]
            else:
                twohop_dict[n_code][0].append(node)
                twohop_dict[n_code][1].append(weight)
                twohop_dict[n_code][2].append(rel)
                twohop_dict[n_code][3].append(code)

    all_ents_node = [id2concept[x] for x in all_ents]
    twohop_nodes = {k:v for k, v in twohop_dict.items() if k not in all_ents_node}
    twohop_nodes = sorted(twohop_nodes.items(), key=lambda x:sum(x[1][1]), reverse=True) # sort by weight sum
    twohop_nodes = sorted(twohop_nodes, key=lambda x:sum([rel2prior(u) for u in x[1][2]]) / len(x[1][2]))
    twohop_nodes = sorted(twohop_nodes, key=lambda x:len(x[1][0]), reverse=True) # sort by number of source neighbor nodes
    twohop_nodes = sorted(twohop_nodes, key=lambda x:len(set(x[1][0]) & set(qv_nodes)), reverse=True)
    twohop_nodes = sorted(twohop_nodes, key=lambda x:len(set(x[1][0]) & set(q_chunks)), reverse=True)
    twohop_nodes = [x[0] for x in twohop_nodes]

    for new_code in twohop_nodes:
        if graph.number_of_nodes() >= max_size: # 1000
            break
        if new_code not in all_ents:
            all_ents.add(new_code)
            graph.add_node(new_code, id=new_code, a=0, q=0, v=0, n=1)
        else:
            graph.nodes[new_code]['n']=1
            
    for key_code, value in twohop_dict.items():
        if key_code in all_ents:
            n, weights, rels, codes = value
            for w,r,c in zip(weights, rels, codes):
                if c not in all_ents: continue
                r = parse_rel(r)
                graph.add_edge(c, key_code, rel=r, weight=float(w))
                all_attrs.add((c, key_code, r))
                # graph.add_edge(key_code, c, rel=r+len(relation2id), weight=float(w))
                # all_attrs.add((key_code, c, r+len(relation2id)))



    # if check_question(question_id):
    #     print(question_id)
    #     # print(onehop_nodes[:80])
    #     for x in twohop_nodes[:50]: # 50
    #         print( x[1][0], ' ---> qv ---> ', x[0])
    
    # Step 4. Add answer nodes
    # for a_node in cand_answers:
    #     try:
    #         a_code = concept2id[a_node]
    #         if a_code not in all_ents:
    #             all_ents.add(a_code)
    #             graph.add_node(a_code, id=a_code, a=1, v=0, q=0, n=0)
    #         else:
    #             graph.nodes[a_code]['a']=1
    #     except: pass

    # Step 5. Augmente small graphs
    # if graph.number_of_nodes() <= 30 or graph.number_of_edges() <= 30:
    #     print(question_id)
    #     print(augment_nodes)
    #     print([(id2concept[x[0]],x[1]) for x in list(graph.nodes.data())]) 
        # for x_code in list(all_ents):
        #     neighbors = get_code_neighbors(x_code)
        #     for neighbor in neighbors:
        #         rel, new_node, weight = neighbor
        #         rel = parse_node(rel)
        #         if graph.number_of_nodes() >= 100 :
        #             break
        #         try: 
        #             new_code = concept2id[new_node]
        #         except: continue
        #         if new_code not in all_ents:
        #             all_ents.add(new_code)
        #             graph.add_node(new_code, id=new_code, n=1, q=0, v=0, a=0)
        #         else:
        #             graph.nodes[new_code]['n']=1
        #         all_attrs.add((x_code, new_code, rel))
        #         graph.add_edge(x_code, new_code, rel=rel, weight=weight)


    # Step 6.1 Add bridge nodes between every two existing nodes
    # combinations or permutations, consider directional or not
    # for pair in itertools.combinations(list(all_ents), r=2): # start, end
    #     pair = list(pair)
    #     if graph.number_of_nodes() >= 100 :
    #         break
    #     try:
    #         neighbors = [all_neighbor_codes[pair[0]], all_neighbor_codes[pair[1]]]
    #     except: continue
    #     third_nodes = list(set(neighbors[0].keys()) & set(neighbors[1].keys()))
    #     if check_question(question_id):
    #         print([id2concept[x] for x in pair], ' ----n---> ', third_nodes)
    #     for new_node in third_nodes:
    #         if graph.number_of_nodes() >= 100 :
    #             break
    #         try:
    #             new_code = concept2id[new_node]
    #         except: continue
    #         # add a new node, and bridge it as (start <-> new node <-> end)
    #         if new_code not in all_ents:
    #             all_ents.add(new_code)
    #             graph.add_node(new_code, id=new_code, n=1, q=0, v=0, a=0)
    #             for p_code, p_neighbors in zip(pair, neighbors):
    #                 # TODO undefined relation type should be replaced by relatedto, done
    #                 p_rel = parse_rel(p_neighbors[new_node][0])
    #                 graph.add_edge(p_code, new_code, rel=p_rel, weight=float(p_neighbors[new_node][2]))
    #                 graph.add_edge(new_code, p_code, rel=p_rel + len(relation2id), weight=float(p_neighbors[new_node][2]))
    #                 all_attrs.add((p_code, new_code, p_rel))
    #                 all_attrs.add((new_code, p_code, p_rel + len(relation2id)))

    # Step 6.2 Add edges between all nodes if it exist in conceptnet
    # Fully connect the whole graph
    # for pair in itertools.permutations(list(all_ents), r=2):
    #     code_a, code_b = list(pair)
    #     all_attrs.add((code_a, code_b, relation2id['relatedto']))
    #     graph.add_edge(code_a, code_b, rel=relation2id['relatedto'], weight=1.0)

    # for pair in itertools.combinations(list(all_ents), r=2): # start, end
    #     pair = list(pair)
    #     try:
    #         neighbors = [all_neighbor_codes[pair[0]], all_neighbor_codes[pair[1]]]
    #     except: continue
    #     for p, q, p_neighbors in zip(pair, pair[::-1], neighbors):
    #         if q in p_neighbors:
    #             p_rel = parse_rel(p_neighbors[q][0])
    #             graph.add_edge(p, q, rel=p_rel, weight=float(p_neighbors[q][2]))
    #             all_attrs.add((p, q, p_rel))


    # Pad if needed to
    # general_words = ['number','object','one','two','three','four','five','six','day','hours','week','catch']
    # num_entities = graph.number_of_nodes()
    # # case 1: less than 50 nodes, randomly add general words and edges
    # if num_entities < 50:
    #     random.shuffle(general_words)
    #     for word in general_words[:50+1-num_entities]: 
    #         new_code = concept2id[word]
    #         if new_code not in all_ents:
    #             all_ents.add(new_code)
    #             graph.add_node(new_code, id=new_code, n=1, v=0, a=0, q=0)
    #         else:
    #             graph.nodes[new_code]['n']=1
    #     print('Graph has %s nodes now' % len(all_ents))

    # case 2: no edge, add edge for every two node
    if nx.is_empty(graph):
        print('Graph has no edges with %d entities' % (graph.number_of_nodes()))
        print(question_id)
        print([(id2concept[x[0]],x[1]) for x in list(graph.nodes.data())]) 
        all_ents_list = list(all_ents)
        random.shuffle(all_ents_list)
        for pair in itertools.permutations(all_ents_list[:30], r=2):
            graph.add_edge(pair[0], pair[1], rel=relation2id['relatedto'], weight=1.0)

    # if graph.number_of_nodes() < 20 :
    #     print('ATTENTION!! FINAL!! LESS THAN 20 ENTITIES! ONLY ', graph.number_of_nodes())
    #     # print(list(graph.nodes.data()))
    #     for x in graph.nodes.data():
    #         print(id2concept[x[0]], x[1])
    #     print(question_id, concept_strs, q_chunks)

    # output_path = '/data/gjr/OK-VQA/myAnnotations/conceptnet.en.unpruned/graph_%d.gpickle' % (question_id)
    # nx.write_gpickle(graph, output_path)

    # print(all_ents)
    if check_question(question_id):
        print(question_id)
        print(concept_strs)
        print([(id2concept[x[0]],x[1]) for x in list(graph.nodes.data())]) 
    # print(graph.edges.data())
    graph = dgl.from_networkx(graph, node_attrs=['id','v','q','a','n'], edge_attrs=['weight','rel'])
    # if question_id % 8 == 0:
    #     write_neighbor_json()
    return graph, graph.number_of_nodes()


import pickle
concept2label = pickle.load(open('/data/gjr/OK-VQA/myAnnotations/trainval_concept2label_729.pkl', "rb"))
# qid2clscandidates = json.load(open('/data/gjr/OK-VQA/myAnnotations/new_val_split_8501/closeset/qid2clscandidates.json'))
def build_close_graph(question_id, concept_strs, q_chunks, cand_answers, max_size=100):
    graph = nx.DiGraph()
    all_attrs = set()
    all_ents = set()

    # Step 0.0. classification answer candidates 
    # cand_entids = qid2clscandidates[str(question_id)]
    # for a_code in cand_entids:
    #     all_ents.add(a_code)
    #     graph.add_node(a_code, id=a_code, q=0, v=0, a=1, n=0)


    # # Step 0.1.  add conceptnet edges
    # for pair in itertools.permutations(list(all_ents), r=2):
    #     code_a, code_b = list(pair)
    #     if (code_a, code_b, relation2id['relatedto']) not in all_attrs:
    #         node_b = id2concept[code_b]
    #         try:
    #             if node_b in all_neighbor_codes[code_a]:
    #                 rel = parse_rel(all_neighbor_codes[code_a][node_b][0])
    #                 wgt = all_neighbor_codes[code_a][node_b][2]
    #                 all_attrs.add((code_a, code_b, rel))
    #                 graph.add_edge(code_a, code_b, rel=rel, weight=wgt)
    #         except: continue

    # graph = dgl.from_networkx(graph, node_attrs=['id','v','q','a','n'], edge_attrs=['weight','rel'])
    # return graph, graph.number_of_nodes()

    # Step 1. Add edges and nodes from scene graph triples
    for line in concept_strs:
        # ls = [x.lower().replace(" ","_") for x in line.split(' [SEP] ')[1].split('<SEP>')] # for mavex knowledge
        # For Scene Graphs
        score = float(line.split(':')[0])
        ls = [x.lower().replace(" ","_") for x in line.split(':')[1].split('<SEP>')] 
        ls = [rectify_entity(c) for c in ls]
        # parse relation
        rel = parse_rel(ls[1])
        # parse subject and object
        subj = parse_node(ls[0])
        obj = parse_node(ls[2])
        for w in [subj, obj]:
            if w != -1:
                all_ents.add(w)
                graph.add_node(w, id=w, v=1, q=0, a=0, n=0)
        if subj != -1 and obj != -1:
            weight = score # 1.0
            # if (not_save(ls[1]) or not_save(ls[2]) or id2relation[rel] == "hascontext"): continue
            # if id2relation[rel] == "relatedto" or id2relation[rel] == "antonym": continue
            if subj == obj:  continue # delete loops
            if (subj, obj, rel) not in all_attrs:
                graph.add_edge(subj, obj, rel=rel, weight=weight)
                all_attrs.add((subj, obj, rel))
                graph.add_edge(obj, subj, rel=rel + len(relation2id), weight=weight)
                all_attrs.add((obj, subj, rel + len(relation2id)))

    # Step 2. Add question nodes
    for q_node in q_chunks:
        try:
            q_code = concept2id[q_node]
            if q_code not in all_ents:
                all_ents.add(q_code)
                graph.add_node(q_code, id=q_code, q=1, v=0, a=0, n=0)
            else:
                graph.nodes[q_code]['q']=1
            if check_question(question_id):
                print(q_node, q_code)
        except: continue

    # Step 3. Fully connect all q/v nodes  
    # for q_node in q_chunks:
    #     try:
    #         q_code = concept2id[q_node]
    #         for c_code in cand_entids:
    #             all_attrs.add((q_code, c_code, relation2id['relatedto']))
    #             graph.add_edge(q_code, c_code, rel=relation2id['relatedto'], weight=1.0)
    #     except: continue

    # Step 5.  add conceptnet edges
    for pair in itertools.permutations(list(all_ents), r=2):
        code_a, code_b = list(pair)
        if (code_a, code_b, relation2id['relatedto']) not in all_attrs:
            node_b = id2concept[code_b]
            try:
                if node_b in all_neighbor_codes[code_a]:
                    rel = parse_rel(all_neighbor_codes[code_a][node_b][0])
                    wgt = all_neighbor_codes[code_a][node_b][2]
                    all_attrs.add((code_a, code_b, rel))
                    graph.add_edge(code_a, code_b, rel=rel, weight=wgt)
                    all_attrs.add((code_b, code_a, rel+len(relation2id)))
                    graph.add_edge(code_b, code_a, rel=rel+len(relation2id), weight=wgt)
            except: continue


    # Step 4.1 Add one hop answers
    q_codes = [x for x, y in graph.nodes(data=True) if (y['q'] == 1 )]
    qv_codes = [x for x, y in graph.nodes(data=True) if (y['q'] == 1 and y['v'] == 1)]
    q_v_codes = [x for x, y in graph.nodes(data=True) if (y['q'] == 1 or y['v'] == 1)]
    qv_nodes = [id2concept[x] for x in qv_codes]
    all_ents_node = [id2concept[x] for x in all_ents] 
    # Use Invert Index get frequent nodes that connect nodes in q or v
    onehop_dict = {}
    # onehop_anchors = qv_codes if len(qv_codes) >0 else q_v_codes
    onehop_anchors = all_ents
    for code in list(set( onehop_anchors)):
        node = id2concept[code]
        neighbors = get_code_neighbors(code)
        for neighbor in neighbors:
            rel, n_node, weight = neighbor
            try:
                n_code = concept2id[n_node]
            except: continue
            if n_code in concept2label:
                if n_code not in onehop_dict:
                    onehop_dict[n_code] = [[node], [weight], [rel], [code]]
                else:
                    onehop_dict[n_code][0].append(node)
                    onehop_dict[n_code][1].append(weight)
                    onehop_dict[n_code][2].append(rel)
                    onehop_dict[n_code][3].append(code)


    onehop_nodes = {k:v for k, v in onehop_dict.items() if k not in all_ents_node}
    onehop_nodes = sorted(onehop_nodes.items(), key=lambda x:sum(x[1][1]), reverse=True)
    onehop_nodes = sorted(onehop_nodes, key=lambda x:sum([rel2prior(u) for u in x[1][2]]) / len(x[1][2]))
    onehop_nodes = sorted(onehop_nodes, key=lambda x:len(x[1][0]), reverse=True)
    onehop_nodes = sorted(onehop_nodes, key=lambda x:len(set(x[1][0]) & set(qv_nodes)), reverse=True)
    onehop_nodes = sorted(onehop_nodes, key=lambda x:len(set(x[1][0]) & set(q_chunks)), reverse=True)
    onehop_nodes = [x[0] for x in onehop_nodes]

    for new_code in onehop_nodes:
        if graph.number_of_nodes() >= (2 * max_size // 3): # 1000
            break
        if new_code not in all_ents:
            all_ents.add(new_code)
            graph.add_node(new_code, id=new_code, a=0, q=0, v=0, n=1)
        else:
            graph.nodes[new_code]['n']=1

    for key_code, value in onehop_dict.items():
        if key_code in all_ents:
            n, weights, rels, codes = value
            for w,r,c in zip(weights, rels, codes):
                if c not in all_ents: continue
                r = parse_rel(r)
                all_attrs.add((c, key_code, r))
                graph.add_edge(c, key_code, rel=r, weight=float(w))
                all_attrs.add((key_code, c, r+len(relation2id)))
                graph.add_edge(key_code, c, rel=r+len(relation2id), weight=float(w))



    # Step 4.2 Add two hop answers
    twohop_dict = {}
    for code in list(all_ents):
        node = id2concept[code]
        neighbors = get_code_neighbors(code)
        for neighbor in neighbors:
            rel, n_node, weight = neighbor
            try:
                n_code = concept2id[n_node]
            except: continue
            if n_code in concept2label:
                if n_code not in twohop_dict:
                    twohop_dict[n_code] = [[node], [weight], [rel], [code]]
                else:
                    twohop_dict[n_code][0].append(node)
                    twohop_dict[n_code][1].append(weight)
                    twohop_dict[n_code][2].append(rel)
                    twohop_dict[n_code][3].append(code)

    all_ents_node = [id2concept[x] for x in all_ents]
    twohop_nodes = {k:v for k, v in twohop_dict.items() if k not in all_ents_node}
    twohop_nodes = sorted(twohop_nodes.items(), key=lambda x:sum(x[1][1]), reverse=True) # sort by weight sum
    twohop_nodes = sorted(twohop_nodes, key=lambda x:sum([rel2prior(u) for u in x[1][2]]) / len(x[1][2]))
    twohop_nodes = sorted(twohop_nodes, key=lambda x:len(x[1][0]), reverse=True) # sort by number of source neighbor nodes
    twohop_nodes = sorted(twohop_nodes, key=lambda x:len(set(x[1][0]) & set(qv_nodes)), reverse=True)
    twohop_nodes = sorted(twohop_nodes, key=lambda x:len(set(x[1][0]) & set(q_chunks)), reverse=True)
    twohop_nodes = [x[0] for x in twohop_nodes]

    for new_code in twohop_nodes:
        if graph.number_of_nodes() >= max_size: # 1000
            break
        if new_code not in all_ents:
            all_ents.add(new_code)
            graph.add_node(new_code, id=new_code, a=0, q=0, v=0, n=1)
        else:
            graph.nodes[new_code]['n']=1
            
    for key_code, value in twohop_dict.items():
        if key_code in all_ents:
            n, weights, rels, codes = value
            for w,r,c in zip(weights, rels, codes):
                if c not in all_ents: continue
                r = parse_rel(r)
                all_attrs.add((c, key_code, r))
                graph.add_edge(c, key_code, rel=r, weight=float(w))
                all_attrs.add((key_code, c, r+len(relation2id)))
                graph.add_edge(key_code, c, rel=r+len(relation2id), weight=float(w))

    if graph.number_of_nodes() < max_size :
        print('ATTENTION!! FINAL!! LESS THAN 100 ENTITIES! ONLY ', graph.number_of_nodes())
        candidates = list(concept2label.keys())
        random.shuffle(candidates)
        for x in candidates[:max_size-graph.number_of_nodes()]:
            graph.add_node(x, id=x, a=0, q=0, v=0, n=1)


    # for key_code in q_v_codes:
    #     for c in all_ents:
    #         if (key_code, c, relation2id['relatedto']) not in all_attrs:
    #             graph.add_edge(key_code, c, rel=relation2id['relatedto'], weight=float(w))
    #             all_attrs.add((key_code, c, relation2id['relatedto']))

    # case 2: no edge, add edge for every two node
    if nx.is_empty(graph):
        print('Graph has no edges with %d entities' % (graph.number_of_nodes()))
        print(question_id)
        print([(id2concept[x[0]],x[1]) for x in list(graph.nodes.data())]) 
        all_ents_list = list(all_ents)
        random.shuffle(all_ents_list)
        for pair in itertools.permutations(all_ents_list[:50], r=2):
            graph.add_edge(pair[0], pair[1], rel=relation2id['relatedto'], weight=1.0)

    if check_question(question_id):
        print(question_id)
        print([(id2concept[x[0]],x[1]) for x in list(graph.nodes.data())]) 
    # try:
    graph = dgl.from_networkx(graph, node_attrs=['id','v','q','a','n'], edge_attrs=['weight','rel'])
    return graph, graph.number_of_nodes()

def clean_neighbors():
    neighbor_dict_path = "/data/gjr/ConceptNet/All_Neighbor_Codes.json"
    all_neighbor_codes = json.load(open(neighbor_dict_path))
    all_neighbor_codes = {int(key): {v[1]: [v[0].lower(),v[1],v[2]] for v in value}  for key, value in all_neighbor_codes.items()}
    for ignore_c in [concept2id[w] for w in ['internet_slang', 'slang', 'nautical']]:
        if ignore_c in all_neighbor_codes:
            _ = all_neighbor_codes.pop(ignore_c)


    # add reverse index to all neighbor nodes
    new_dict = {}
    for key, value in all_neighbor_codes.items():
        for ignore_w in ['internet_slang', 'slang', 'nautical']:
            if ignore_w in value:
                _ = value.pop(ignore_w)
        if key in new_dict:
            new_dict[key].update(value)
        else:
            new_dict[key] = value
        try:
            node = id2concept[key]
        except: continue
        for nn in value.values():
            rel, n, w = nn
            try:
                c = concept2id[n]
            except: continue
            if c in new_dict:
                new_dict[c].update({node:[rel, node, round(w*0.6, 2)]})
            else:
                new_dict[c] = {node:[rel, node, round(w*0.6, 2)]}


    new_new_dict = {}
    for key, value in new_dict.items():
        kk = list(new_dict[key].values())
        kk.sort()
        new_new_dict[key] = list(k for k,_ in itertools.groupby(kk))

        
    json.dump(new_new_dict, open(neighbor_dict_path, "w"))

def check_question(question_id):
    qids = [425]#, 1395, 1435, 3285,5205, 5815, 550855, 1163265]
    # qids=[]
    return question_id in qids

def write_neighbor_json():
    all_neighbor_codes_l = {int(key): list(value.values()) for key, value in all_neighbor_codes.items()}
    json.dump(all_neighbor_codes_l, open(neighbor_dict_path, "w"))
    
def find_neighbor(a,b):
    a = all_neighbor_codes[concept2id[a]]
    b = all_neighbor_codes[concept2id[b]]
    return set(a.keys() )& set(b.keys())

if __name__ == "__main__":
    glove_init("../data/glove/glove.6B.200d.txt", "../data/glove/glove.200d", '../data/glove/tp_str_corpus.json')
    compute_all_neighbors()

