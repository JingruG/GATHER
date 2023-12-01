# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import _pickle as cPickle
import pickle
import logging
import h5py
import torch
from torch.utils.data import Dataset
import numpy as np
from ._image_features_reader import ImageFeaturesH5Reader
import copy
from transformers import AutoTokenizer, AutoModel
from .conceptnet import build_graph, build_close_graph, write_neighbor_json, id2concept, find_gt
import networkx as nx
import dgl
from dgl.data.utils import load_graphs, save_graphs
import random
from tqdm import tqdm
import Levenshtein


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
from .feat_utils import read_features
from .concept_utils import concept_to_sentence

def assert_eq(real, expected):
    assert real == expected, "%s (true) vs %s (expected)" % (real, expected)


def _create_entry(question, answer):
    # answer.pop("image_id")
    # answer.pop("question_id")
    entry = {
        "question_id": question["question_id"],
        "image_id": int(question["question_id"] // 10),
        "question": question["question"],
        "answer": answer,
    }
    return entry


def _load_dataset(dataroot, name, clean_datasets, postfix, use_split_name):
    """Load entries

    dataroot: root path of dataset
    name: 'train', 'val', 'trainval', 'minsval'
    """

    question_path = os.path.join(
        '/data/gjr/OK-VQA/myAnnotations/new_val_split_%s' % (use_split_name), "OpenEnded_mscoco_%s2014_questions%s.json" % (name, postfix)
    )
    answer_path = os.path.join('/data/gjr/OK-VQA/myAnnotations/new_val_split_%s' % (use_split_name), "%s_target_729%s.pkl" % (name, postfix))

    # If test on entire OK-VQA dataset, use the following two lines
    # question_path = '/data/gjr/OK-VQA/mavex/OpenEnded_mscoco_%s2014_questions.json' % (name)
    # answer_path = '/data/gjr/OK-VQA/mavex/%s_target_729.pkl' % (name)
    
    questions = sorted(json.load(open(question_path))["questions"], key=lambda x: x["question_id"])
    answers = cPickle.load(open(answer_path, "rb"))
    answers = sorted(answers, key=lambda x: x["question_id"])
    assert_eq(len(questions), len(answers))
    entries = []
    id = 0
    for question in questions:
        answer = answers[id]
        id += 1
        assert_eq(question["question_id"], answer["question_id"])
        entries.append(_create_entry(question, answer))

    return entries

class VQAClassificationDataset(Dataset):
    def __init__(
            self,
            task,
            dataroot,
            annotations_jsonpath,
            split,
            image_features_reader,
            gt_image_features_reader,
            tokenizer,
            bert_model,
            clean_datasets=None,
            padding_index=0,
            max_seq_length=16,
            max_region_num=101,
            args=None,
    ):
        super().__init__()
        self.graph_size = args.graph_size
        self.use_split_name = args.use_split_name
        self.split = split
        self.ablation = args.ablation
        ans2label_path = os.path.join('/data/gjr/OK-VQA/myAnnotations/new_val_split_%s' % (self.use_split_name), "trainval_ans2label_729.pkl")
        label2ans_path = os.path.join('/data/gjr/OK-VQA/myAnnotations/new_val_split_%s' % (self.use_split_name), "trainval_label2ans_729.pkl" )
        self.ans2label = cPickle.load(open(ans2label_path, "rb"))
        self.label2ans = cPickle.load(open(label2ans_path, "rb"))
        self.num_labels = len(self.ans2label)
        self._max_region_num = max_region_num
        self._max_seq_length = max_seq_length
        self._image_features_reader = image_features_reader
        self._tokenizer = tokenizer
        self._padding_index = padding_index
        self.separate = 0
        
        self.num_wiki_sentences = args.num_wiki_sentences
        self.num_concepts = args.num_concepts
        self.wiki_seq_length = 20
        self.concept_seq_length = 14

        self.use_ConceptNet = True
        if self.use_ConceptNet:
            self.postfix = '_in_ConceptNet'
        else: 
            self.postfix = ''
        if split == 'test':
            new_split = 'val'
        else:
            new_split = 'train'
        self.new_split = new_split
        self.entries = _load_dataset(dataroot, new_split, clean_datasets, self.postfix, self.use_split_name)
        self.image_feats_hf = h5py.File('/data/gjr/OK-VQA/mavex/h5py_accumulate/image_%s.hdf5' % new_split, 'r')
        self.image_feats = self.image_feats_hf.get('features')
        self.image_qid2ind = cPickle.load(open('/data/gjr/OK-VQA/mavex/h5py_accumulate/image_%s_qid_ans2idx.pkl' % new_split, 'rb'))

        self.use_search_word = args.use_search_word
        self.qid2results = pickle.load(open('qid2answer_candidates_trainval.pkl', 'rb'))
        self.qid2results.update(pickle.load(open('qid2answer_candidates_test.pkl', 'rb')))
        self.qid2knowledge = cPickle.load(open('/data/gjr/OK-VQA/mavex/qid2knowledge_cleaned_mavex.pkl', 'rb'))
        self.qid2newconcept = cPickle.load(open('/data/gjr/OK-VQA/myAnnotations/qid2knowledge_augmented_concept.pkl', 'rb'))
        self.small_bert_tokenizer = AutoTokenizer.from_pretrained("/data/gjr/huggingface/bert-small")
        self.id2path = cPickle.load(open('/data/gjr/OK-VQA/mavex/coco_iid2feat_path.pkl', 'rb'))
        self.qid2qchunks = cPickle.load(open('/data/gjr/OK-VQA/myAnnotations/question_chunk_entities.pkl', 'rb'))
        self.tokenize(max_seq_length)
        self.tensorize()     
        
        self.targets = pickle.load(open('/data/gjr/OK-VQA/myAnnotations/new_val_split_%s/%s_target_729%s.pkl' % (self.use_split_name, self.new_split, self.postfix), 'rb'))

        self.qid2gt_answer = {}
        for val_target in self.targets:
            question_id = val_target['question_id']
            labels = val_target['labels']
            scores = val_target['scores']
            self.qid2gt_answer[question_id] = {}
            for i in range(len(labels)):
                self.qid2gt_answer[question_id][self.label2ans[labels[i]]] = scores[i] 
            
        print('targets', len(self.targets))
        self.qid2gt_concept, self.qid2gt_concept_scores = {}, {}
        if self.use_ConceptNet:
            for val_target in self.targets:
                question_id = val_target['question_id']
                self.qid2gt_concept[question_id] = val_target['concept_ids']
                self.qid2gt_concept_scores[question_id] = val_target['concept_scores']

        # Scene Graph
        self.sg_prediction = json.load(open('/data/gjr/OK-VQA/myAnnotations/scene_graph/okvqa_scene_graph_prediction.json' ))

        self.close_set = 'close' if 'close' in self.ablation else 'open'
        self.graph_list, label_dict = load_graphs("/data/gjr/OK-VQA/myAnnotations/schemagraphs_full/%s_graphs_%s_relsorted_%s.bin" % (self.new_split, self.close_set, self.graph_size))
        self.qid2index = {x:i for i,x in enumerate(label_dict['question_id'].tolist())}

        new_data_split_dict = json.load(open('/data/gjr/OK-VQA/myAnnotations/new_val_split_%s/new_open_split_qids.json' % (self.use_split_name)))
        self.open_test_qids = new_data_split_dict['open_test']
        self.partial_open_test_qids = new_data_split_dict['partial_open_test']
        self.close_test_qids = new_data_split_dict['close_test']
        self.concept2label = pickle.load(open('/data/gjr/OK-VQA/myAnnotations/trainval_concept2label_729.pkl', "rb"))

        # self.gt_check = json.load(open('/home/jingrugan/KBVQA/MAVEX/%s_gt_results.json' % (self.new_split)))

    def tokenize(self, max_length=16):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_index in embedding
        """
        for entry in self.entries:
            tokens = self._tokenizer.encode(entry["question"])
            tokens = tokens[: max_length - 2]
            tokens = self._tokenizer.add_special_tokens_single_sentence(tokens)

            segment_ids = [0] * len(tokens)
            input_mask = [1] * len(tokens)

            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self._padding_index] * (max_length - len(tokens))
                tokens = tokens + padding
                input_mask += padding
                segment_ids += padding

            assert_eq(len(tokens), max_length)
            entry["q_token"] = tokens
            entry["q_input_mask"] = input_mask
            entry["q_segment_ids"] = segment_ids

    def tokenize_sentence(self, sentence, sw='', max_length=16):
        if len(sw):
            # tokens = self._tokenizer.convert_tokens_to_ids(self._tokenizer.tokenize(sentence) + ['[SEP]'] + self._tokenizer.tokenize(sw))
            tokens1 = self.small_bert_tokenizer.convert_tokens_to_ids(self.small_bert_tokenizer.tokenize(sentence))
            segment_ids1 = [0] * len(tokens1)
            tokens1 = tokens1[: max_length - 3 - 4]
            segment_ids1 = segment_ids1[: max_length - 3 - 4]

            tokens2 = self.small_bert_tokenizer.convert_tokens_to_ids(self.small_bert_tokenizer.tokenize(sw))
            segment_ids2 = [1] * len(tokens2)
            tokens2 = tokens2[:  4]
            segment_ids2 = segment_ids2[: 4]
            # SEP is 102 CLS 101 PAD 0
            tokens = [101] + tokens1 + [102] + tokens2 + [102]
            segment_ids = [0] + segment_ids1 + [0] + segment_ids2 + [1]
        else:
            tokens = self.small_bert_tokenizer.convert_tokens_to_ids(self.small_bert_tokenizer.tokenize(sentence))
            tokens = [101] + tokens[: max_length - 2] + [102]
            segment_ids = [0] * len(tokens)

        input_mask = [1] * len(tokens)

        if len(tokens) < max_length:
            # Note here we pad in front of the sentence
            padding = [0] * (max_length - len(tokens))
            tokens = tokens + padding
            input_mask += padding
            segment_ids += padding

        return tokens, input_mask, segment_ids

    def tensorize(self):

        for entry in self.entries:
            question = torch.from_numpy(np.array(entry["q_token"]))
            entry["q_token"] = question

            q_input_mask = torch.from_numpy(np.array(entry["q_input_mask"]))
            entry["q_input_mask"] = q_input_mask

            q_segment_ids = torch.from_numpy(np.array(entry["q_segment_ids"]))
            entry["q_segment_ids"] = q_segment_ids

            # if "test" not in self.split:
            if True:
                answer = entry["answer"]
                labels = np.array(answer["labels"])
                scores = np.array(answer["scores"], dtype=np.float32)
                if len(labels):
                    labels = torch.from_numpy(labels)
                    scores = torch.from_numpy(scores)
                    entry["answer"]["labels"] = labels
                    entry["answer"]["scores"] = scores
                else:
                    entry["answer"]["labels"] = None
                    entry["answer"]["scores"] = None

    def __getitem__(self, index):
        #  mode 1 no share | share features = zero(1, 768)
        #  mode 2 only share score 0 features
        #  mode 3 share all features

        entry = self.entries[index]
        image_id = entry["image_id"]
        question_id = entry["question_id"]
        features, num_boxes, boxes, xxx = read_features('/data/gjr/OK-VQA/mavex/'+self.id2path[image_id])

        mix_num_boxes = min(int(num_boxes), self._max_region_num)
        mix_boxes_pad = np.zeros((self._max_region_num, 5))
        mix_features_pad = np.zeros((self._max_region_num, 2048))

        image_mask = [1] * (int(mix_num_boxes))
        while len(image_mask) < self._max_region_num:
            image_mask.append(0)

        mix_boxes_pad[:mix_num_boxes] = boxes[:mix_num_boxes]
        mix_features_pad[:mix_num_boxes] = features[:mix_num_boxes]

        features = torch.tensor(mix_features_pad).float()
        image_mask = torch.tensor(image_mask).long()
        spatials = torch.tensor(mix_boxes_pad).float()

        question = entry["q_token"]
        input_mask = entry["q_input_mask"]
        segment_ids = entry["q_segment_ids"]

        co_attention_mask = torch.zeros((self._max_region_num, self._max_seq_length))
        target = torch.zeros(self.num_labels)
        wiki_tokens, wiki_input_masks, wiki_segment_ids, wiki_sents = self.gather_wiki_features(question_id, self.wiki_seq_length)
        concept_tokens, concept_input_masks, concept_segment_ids, concept_sents, concept_strs = self.gather_concept_features(question_id, self.concept_seq_length)

        verif_answers = []
        verif_labels = []
        verif_inds = []
        ctt = 0
        for pred_answer in self.qid2results[question_id][:5]:
            verif_answers.append(pred_answer)
            verif_inds.append(self.ans2label[pred_answer])
            if pred_answer in self.qid2gt_answer[question_id]:
                verif_labels.append(self.qid2gt_answer[question_id][pred_answer])
                ctt += 1
            else:
                verif_labels.append(0.)
                ctt += 1
        verif_labels = torch.from_numpy(np.array(verif_labels)).float()
        verif_inds = torch.from_numpy(np.array(verif_inds)).float()

        answer = entry["answer"]
        labels = answer["labels"]
        scores = answer["scores"]
        target_txt = ' '.join(list(self.qid2gt_answer[question_id].keys()))

        question_txt = entry["question"]

        if labels is not None:
            target.scatter_(0, labels, scores)

        kg_sents = ' '.join(wiki_sents[:3]) + ' '.join(concept_sents[:3])
        if kg_sents == '':
            print('knowledge error for question ', question_id, question_txt)
            kg_sents = ' '

        # GJR Concept
        if self.use_ConceptNet:
            verif_concepts = self.qid2gt_concept[question_id]
            verif_concept_scores = self.qid2gt_concept_scores[question_id]
        else:
            verif_concepts = [0]

        g = self.graph_list[self.qid2index[question_id]]

        # For classification only
        if 'classification' in self.ablation:
            verif_inds = torch.zeros(len(self.concept2label), dtype=torch.bool)
            for verif_concept_id in verif_concepts:
                verif_inds = verif_inds | (torch.tensor(list(self.concept2label.keys())) == int(verif_concept_id))
            verif_inds = verif_inds * 1.

        # Close-end: candidate if node in 4025
        if 'close_end' in self.ablation:
            node_ids = g.ndata['id'].tolist()
            g.ndata['a'] = torch.tensor([1 if x in self.concept2label else 0 for x in node_ids])
        # transform1 = dgl.AddSelfLoop()
        transform1 = dgl.RemoveSelfLoop()
        g_1 = transform1(g)
        transform2 = dgl.NodeShuffle()
        g_2, new_edges = transform2(g_1)
        subgraph = dgl.graph(new_edges)
        if subgraph.number_of_nodes() != g.number_of_nodes():
            subgraph.add_nodes(g.number_of_nodes() - subgraph.number_of_nodes())
        for ntype in g_2.ntypes:
            for key, feat in g_2.nodes[ntype].data.items():
                subgraph.nodes[ntype].data[key] = feat
        for c_etype in g_2.canonical_etypes:
            for key, feat in g_2.edges[c_etype].data.items():
                subgraph.edges[c_etype].data[key] = feat

        assert subgraph.number_of_nodes() == g.number_of_nodes()
        concept_ids = list(subgraph.ndata['id'])
        max_node_num = len(concept_ids)

        node_mask = [1] * len(concept_ids) + [0] * (max_node_num - len(concept_ids))
        node_scores = [0.] * max_node_num
        train_node_scores = [0.] * max_node_num
        # node_scores[0] = 0.5
        for verif_concept_id, concept_score in zip(verif_concepts, verif_concept_scores):
            if verif_concept_id in concept_ids:
                if node_scores[concept_ids.index(verif_concept_id)] < concept_score:
                    node_scores[concept_ids.index(verif_concept_id)] = concept_score
                    train_node_scores[concept_ids.index(verif_concept_id)] = 1.0


        if sum(train_node_scores) == 0:
            # recompute node_scores
            train_node_scores, verif_concepts = self.gt_update(train_node_scores, verif_concepts, concept_ids)
        
        # Closed Node to Ground Truth
        verif_concepts = [verif_concepts[0]]
        
        close_mask = question_id in self.close_test_qids
        partial_open_mask = question_id in self.partial_open_test_qids
        open_mask = question_id in self.open_test_qids

        return (
            features,
            spatials,
            image_mask,
            question,
            target,
            input_mask,
            segment_ids,
            co_attention_mask,
            question_id,
            verif_inds,
            verif_concepts,
            node_scores,
            train_node_scores,
            node_mask,
            close_mask,
            partial_open_mask,
            open_mask,
            question_txt,
            target_txt,
            kg_sents,
            subgraph,
        )

    def __len__(self):
        return len(self.entries)

    def gt_update(self, scores, gt_codes, graph_codes):    
        gt_nodes = [id2concept[x] for x in gt_codes]
        graph_nodes = [id2concept[x] for x in graph_codes]
        for x in gt_nodes:
            dist = [Levenshtein.distance(x,y) for y in graph_nodes]
            min_socre = min(dist)
            scores[dist.index(min_socre)] = .001
            gt_codes.append(graph_codes[dist.index(min_socre)])
            break
        return scores, gt_codes

    def gather_wiki_features(self, question_id, max_length=22):
        num_wiki_sentences_total = self.num_wiki_sentences
        wiki_tokens = []
        wiki_input_masks = []
        wiki_segment_ids = []

        if self.separate:
            wiki_sentences = self.qid2knowledge[question_id]['separate_wiki_sentences'][:num_wiki_sentences_total]
        else:
            wiki_sentences = self.qid2knowledge[question_id]['non_separate_wiki_sentences'][:num_wiki_sentences_total]

        sents = []
        for wiki in wiki_sentences:
            assert len(wiki.split(' [SEP] ')) == 2
            search_word = wiki.split(' [SEP] ')[0]
            sent = wiki.split(' [SEP] ')[1]
            sents.append(sent)

            token, input_mask, segment_id = self.tokenize_sentence(sent,
                                                                       search_word if self.use_search_word else '',
                                                                       max_length)
            wiki_tokens.append(token)
            wiki_input_masks.append(input_mask)
            wiki_segment_ids.append(segment_id)
        current_num_sentences = len(wiki_tokens)
        for i in range(current_num_sentences, num_wiki_sentences_total):
            wiki_tokens.append([0] * max_length)
            wiki_input_masks.append([0] * max_length)
            wiki_segment_ids.append([0] * max_length)

        wiki_tokens = torch.from_numpy(np.array(wiki_tokens).astype('int'))
        wiki_input_masks = torch.from_numpy(np.array(wiki_input_masks).astype('int'))
        wiki_segment_ids = torch.from_numpy(np.array(wiki_segment_ids).astype('int'))
        return wiki_tokens, wiki_input_masks, wiki_segment_ids, sents

    def gather_concept_features(self, question_id, max_length=18):
        num_concept_sentences_total = self.num_concepts
        concept_tokens = []
        concept_input_masks = []
        concept_segment_ids = []
        sents = []
        if self.separate:
            concept_strs = self.qid2knowledge[question_id]['separate_concept_strs'][:num_concept_sentences_total]
        else:
            concept_strs = self.qid2knowledge[question_id]['non_separate_concept_strs'][:num_concept_sentences_total]
        new_concept_strs = self.qid2newconcept[question_id]
        new_concept_strs.extend(concept_strs)

        for concept_str in concept_strs:
            concept_sent = concept_to_sentence(concept_str.split(' [SEP] ')[1].split('<SEP>'))
            search_word = concept_str.split(' [SEP] ')[0]
            sents.append(concept_sent)
            token, input_mask, segment_id = self.tokenize_sentence(concept_sent,
                                                                       search_word if self.use_search_word else '',
                                                                       max_length)
            concept_tokens.append(token)
            concept_input_masks.append(input_mask)
            concept_segment_ids.append(segment_id)
        current_num_sentences = len(concept_tokens)
        for i in range(current_num_sentences, num_concept_sentences_total):
            concept_tokens.append([0] *max_length )
            concept_input_masks.append([0] *max_length )
            concept_segment_ids.append([0] *max_length )
        concept_tokens = torch.from_numpy(np.array(concept_tokens).astype('int'))
        concept_input_masks = torch.from_numpy(np.array(concept_input_masks).astype('int'))
        concept_segment_ids = torch.from_numpy(np.array(concept_segment_ids).astype('int'))

        return concept_tokens, concept_input_masks, concept_segment_ids, sents, new_concept_strs

    def gather_scene_graph_triples(self, image_id):
        triples = self.sg_prediction[str(image_id)]
        return triples

    def write_graphs(self):
        graphs = []
        nodes_num, edge_num = 0, 0
        qids = []
        for index, entry in tqdm(enumerate(self.entries)):
            entry = self.entries[index]
            image_id = entry["image_id"]
            question_id = entry["question_id"]
            qids.append(question_id)

            # concept_tokens, concept_input_masks, concept_segment_ids, concept_sents, concept_strs = self.gather_concept_features(question_id, self.concept_seq_length)
            concept_strs = self.gather_scene_graph_triples(image_id)
            q_chunks = self.qid2qchunks[question_id]
            cand_answers = self.qid2results[question_id]
            # try:
            if self.close_set == 'open':
                subgraph, node_num = build_graph(question_id, concept_strs, q_chunks, None, max_size=self.graph_size)
            else:
                subgraph, node_num = build_close_graph(question_id, concept_strs, q_chunks, cand_answers, max_size=self.graph_size)
            edge_num += subgraph.number_of_edges()
            nodes_num += node_num
            # except Exception as e:
            #     print("ERROR while building subgraph for ", question_id, concept_strs)
            #     print(e)
            #     subgraph = []
            graphs.append(subgraph)
            # subgraph = nx.read_gpickle('/home/airflowuser/project/graph.gpickle' + self.id2path[image_id])
        
        nodes_num = nodes_num / len(self.entries)
        edge_num = edge_num / len(self.entries)
        print('Average %f number of nodes' % nodes_num)
        print('Average %f edges' % (edge_num))
        graph_labels = {"question_id": torch.tensor(qids)}
        save_graphs("/data/gjr/OK-VQA/myAnnotations/schemagraphs_full/%s_graphs_%s_relsorted_%s.bin" % (self.new_split, self.close_set, self.graph_size), graphs, graph_labels)
        # write_neighbor_json()

        return True


    def find_answer_hops(self):
        graphs = []
        gt_hop_all = 0
        qid_num = len(self.entries) * 1.0
        hop_num = {0:0,1:0,2:0,100:0}
        rel_dict = {}
        flag = {'v':0, 'q':0, 'none':0}
        for index, entry in tqdm(enumerate(self.entries)):
            entry = self.entries[index]
            image_id = entry["image_id"]
            question_id = entry["question_id"]
            gt_codes = entry["answer"]["concept_ids"]

            concept_strs = self.gather_scene_graph_triples(image_id)
            q_chunks = self.qid2qchunks[question_id]
            cand_answers = self.qid2results[question_id]
            # try:
            gt_hop, vq, rel = find_gt(question_id, concept_strs, q_chunks, gt_codes)
            if gt_hop != 100:
                gt_hop_all += gt_hop 
                hop_num[gt_hop] += 1
                flag[vq] += 1
            else:
                print('can\'t find gt_hop ')
                hop_num[100] += 1
            if rel in rel_dict:
                rel_dict[rel] += 1
            else:
                rel_dict[rel] = 1
            # except Exception as e:
            #     print("ERROR can't find answer ", question_id)
            #     print(e)
        
        print('ground truth hops', gt_hop_all / qid_num)
        print('hop_0, hop_1, hop_2,  cant_find ', hop_num[0], hop_num[1], hop_num[2], hop_num[100] )
        print('v  q', flag['v'], flag['q'])
        print('rel_dict', rel_dict)

        return True

    def check_GT(self):

        results = {}
        not_in_graph, all_scores, all_scores_ = 0, 0., 0.
        qid_num = len(self.entries) * 1.0
        nodes_num, max_num, min_num, edge_num = 0, 0, 1000, 0
        q_node_num, v_node_num, a_node_num, n_node_num, q_v_num, q_a_num, v_a_num, q_v_a_num = 0, 0, 0, 0, 0, 0, 0, 0
        gt_in_q, gt_in_a, gt_in_v, gt_in_n = 0,0,0,0
        def nodes_with_feature(nodes, attr):
            return (nodes.data[attr] == 1).squeeze(1)
        for index, entry in enumerate(self.entries):
            entry = self.entries[index]

            question_id = entry["question_id"]
            verif_concepts = self.qid2gt_concept[question_id]
            verif_concept_scores = self.qid2gt_concept_scores[question_id]


            subgraph = self.graph_list[self.qid2index[question_id]]
            concept_ids = list(subgraph.ndata['id'])

            max_num = max(max_num, len(concept_ids))
            min_num = min(min_num, len(concept_ids))
            nodes_num += len(concept_ids)
            edge_num += subgraph.number_of_edges()

            q_nodes = subgraph.filter_nodes(lambda x: (x.data['q'] == 1).squeeze()).tolist()
            a_nodes = subgraph.filter_nodes(lambda x: (x.data['a'] == 1).squeeze()).tolist()
            v_nodes = subgraph.filter_nodes(lambda x: (x.data['v'] == 1).squeeze()).tolist()
            n_nodes = subgraph.filter_nodes(lambda x: (x.data['n'] == 1).squeeze()).tolist()
            q_codes = [concept_ids[i] for i in q_nodes]
            v_codes = [concept_ids[i] for i in v_nodes]
            a_codes = [concept_ids[i] for i in a_nodes]
            n_codes = [concept_ids[i] for i in n_nodes]
            q_node_num += len(q_nodes)
            v_node_num += len(v_nodes)
            a_node_num += len(a_nodes)
            n_node_num += len(n_nodes)
            q_v_num += len(list(set(q_nodes) & set(v_nodes)))
            q_a_num += len(list(set(q_nodes) & set(a_nodes)))
            v_a_num += len(list(set(v_nodes) & set(a_nodes)))
            q_v_a_num += len(list(set(q_nodes) & set(v_nodes) & set(a_nodes)))


            q_flag, v_flag, a_flag, n_flag, gt_score = 0,0,0,0,0
            for verif_concept_id, verif_concept_score in zip(verif_concepts, verif_concept_scores):
                if verif_concept_id in concept_ids:
                    gt_score = max(gt_score, verif_concept_score)
                    if verif_concept_id in q_codes:
                        q_flag = 1
                    if verif_concept_id in v_codes:
                        v_flag = 1
                    if verif_concept_id in a_codes:
                        a_flag = 1
                    if verif_concept_id in n_codes:
                        n_flag = 1
            gt_in_q, gt_in_v, gt_in_a, gt_in_n = gt_in_q+q_flag, gt_in_v+v_flag, gt_in_a+a_flag, gt_in_n+n_flag
            
            all_scores += gt_score

            if gt_score == 0:
                not_in_graph += 1




        def get_avg(x):
            return x / qid_num
        print('Average %f number of nodes' % get_avg(nodes_num))
        print('Average %f edges' % get_avg(edge_num))
        print('Maximum GT Recall %f, %f' %( get_avg(all_scores), all_scores))
        print('Maximum %f and Minimum %f number of nodes' % (max_num, min_num))
        print('No GT in the graph: %d / %d questions' % (not_in_graph, qid_num))
        print('GT in \'q\' nodes: %d / %d questions' % (gt_in_q, qid_num))
        print('GT in \'v\' nodes: %d / %d questions' % (gt_in_v, qid_num))
        print('GT in \'a\' nodes: %d / %d questions' % (gt_in_a, qid_num))
        print('GT in \'n\' nodes: %d / %d questions' % (gt_in_n, qid_num))

        print('%f nodes per graph are labeled q' % get_avg(q_node_num))
        print('%f nodes per graph are labeled v' % get_avg(v_node_num))
        print('%f nodes per graph are labeled a' % get_avg(a_node_num))
        print('%f nodes per graph are labeled n' % get_avg(n_node_num))
        print('%f nodes per graph are labeled q & v' % get_avg(q_v_num))
        print('%f nodes per graph are labeled q & a' % get_avg(q_a_num))
        print('%f nodes per graph are labeled v & a' % get_avg(v_a_num))
        print('%f nodes per graph are labeled v & q & a' % get_avg(q_v_a_num))
        return True
