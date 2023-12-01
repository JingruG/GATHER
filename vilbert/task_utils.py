# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from random import randint
# from allennlp.common.params import Params
from Levenshtein import distance as levenshtein_distance
import dgl
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import allennlp_models.tagging
# from allennlp.predictors.predictor import Predictor
from tqdm import tqdm
import jsonlines
from typing import List
import re
import numpy as np
import pdb
from pytorch_transformers.tokenization_bert import BertTokenizer
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Dataset, RandomSampler
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch
import sys
import os
import logging
import json
from io import open
from tkinter import TRUE
import warnings
from xml.etree.ElementTree import TreeBuilder
from collections import OrderedDict
import time
from .datasets.conceptnet import id2concept



def fxn():
    warnings.warn("deprecated", DeprecationWarning)


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()


# import nltk
# nltk.download('punkt')

logger = logging.getLogger(__name__)

LossMap = {
    "BCEWithLogitLoss": nn.BCEWithLogitsLoss(reduction="none"),
    "CrossEntropyLoss": nn.CrossEntropyLoss(),
    "MultiMarginLoss": nn.MultiMarginLoss(),
}
# predictor = Predictor.from_path("/data/gjr/huggingface/allennlp-public-models/snli-roberta-large-2020.04.30.tar.gz", \
#     predictor_name="textual-entailment", overrides={"dataset_reader.tokenizer.add_special_tokens": False})
hg_model_hub_name = "/data/gjr/huggingface/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"

nli_tokenizer = AutoTokenizer.from_pretrained(hg_model_hub_name)
nli_model = AutoModelForSequenceClassification.from_pretrained(
    hg_model_hub_name)


def predict(premise, hypothesis):
    tokenized_input_seq_pair = nli_tokenizer.encode_plus(premise, hypothesis,
                                                         max_length=256,
                                                         return_token_type_ids=True, truncation=True)

    input_ids = torch.Tensor(
        tokenized_input_seq_pair['input_ids']).long().unsqueeze(0)
    token_type_ids = torch.Tensor(
        tokenized_input_seq_pair['token_type_ids']).long().unsqueeze(0)
    attention_mask = torch.Tensor(
        tokenized_input_seq_pair['attention_mask']).long().unsqueeze(0)

    input_ids = torch.Tensor(
        tokenized_input_seq_pair['input_ids']).long().unsqueeze(0)
    # remember bart doesn't have 'token_type_ids', remove the line below if you are using bart.
    token_type_ids = torch.Tensor(
        tokenized_input_seq_pair['token_type_ids']).long().unsqueeze(0)
    attention_mask = torch.Tensor(
        tokenized_input_seq_pair['attention_mask']).long().unsqueeze(0)

    outputs = nli_model(input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        labels=None)
    # Note:
    # "id2label": {
    #     "0": "entailment",
    #     "1": "neutral",
    #     "2": "contradiction"
    # },
    predicted_probability = torch.softmax(outputs[0], dim=1)[
        0].tolist()  # batch_size only one
    # print('predicted_probability', predicted_probability)
    entail_score = (
        predicted_probability[0] + predicted_probability[1] - predicted_probability[2] + 1) / 3
    return entail_score, predicted_probability[2]


def entailment_score(hypothesis: List, premises: List, thredshold: float) -> float:
    entail_all, contra_all = 0, 0
    assert len(hypothesis) == len(premises), print(
        "the length of hypothesis and premises should be the same")
    for hyp, pre in zip(hypothesis, premises):
        entail, contra = predict(hypothesis=hyp, premise=pre)
    #     semantic_score += score if score > thredshold else 0.0
    #     valid +=1 if score > thredshold else 0
    # return semantic_score/valid if valid else 0.0
        entail_all += entail
        contra_all += contra
    return entail_all, contra_all


def read_json(file):
    with open(file) as f:
        data = json.load(f)
    return data

# def compute_entailment_loss():
#     prediction = read_json("result_output_updated.json")
#     prediction = prediction[:10] # for testing
#     threshold = 0.5 # set the threshold of nli_score be 0.5
#     model_evaluation_score= 0
#     for pred in tqdm(prediction):
#         model_pred = pred['pred']
#         ref = pred['ref']
#         label_score = pred['score']
#         pred_score = 0
#         most_similar_label = ""
#         for label, premises in ref.items():
#             if premises: # if the premises is not empty
#                 hypothesis = [prem.replace(label, model_pred) for prem in premises]
#                 nli_score=entailment_score(hypothesis, premises, threshold)
#                 if nli_score > pred_score:
#                     pred_score = nli_score
#                     most_similar_label = label
#         # print(most_similar_label,pred_score)
#         if pred_score:
#             model_evaluation_score+=pred_score * label_score[most_similar_label] # multiply the nli score with the original score.
#     print("the semantic evaluation score for the model is {:.2f}".format(model_evaluation_score/len(prediction)) )


def compute_entailment_loss(statements, pred_answers, gt_answers):
    # print('gt_answers', gt_answers, pred_answers)
    if not isinstance(pred_answers[0], str):
        pred_answers = [p[0] for p in pred_answers]
    nli_scores, nli_losses = [], []
    for stats, gt_ans, pred in zip(statements, gt_answers, pred_answers):
        gts = gt_ans.split(',')
        premises = [x + '.' for x in stats.split('.')]
        min_length = min(len(gts), len(premises))
        gts = gts[:min_length]
        premises = premises[:min_length]
        hypothesis = [premises[i].replace(gt, pred)
                      for i, gt in enumerate(gts)]

        # print('hypothesis', hypothesis)
        # print('premises', premises)
        # print('gts', gts)
        threshold = 0.5  # set the threshold of nli_score be 0.5
        nli_score, nli_loss = entailment_score(hypothesis, premises, threshold)
        nli_scores.append(nli_score)
        nli_losses.append(nli_loss)
        # print('nli_score', nli_score)

        # prediction = read_json("result_output_updated.json")
        # prediction = prediction[:10] # for testing
        # threshold = 0.5 # set the threshold of nli_score be 0.5
        # model_evaluation_score= 0
        # for pred in tqdm(prediction):
        #     model_pred = pred['pred']
        #     ref = pred['ref']
        #     label_score = pred['score']
        #     pred_score = 0
        #     most_similar_label = ""
        #     for label, premises in ref.items():
        #         if premises: # if the premises is not empty
        #             hypothesis = [prem.replace(label, model_pred) for prem in premises]
        #             nli_score=entailment_score(hypothesis, premises, threshold)
        #             if nli_score > pred_score:
        #                 pred_score = nli_score
        #                 most_similar_label = label
        #     # print(most_similar_label,pred_score)
        #     if pred_score:
        #         model_evaluation_score+=pred_score * label_score[most_similar_label] # multiply the nli score with the original score.
        # print("the semantic evaluation score for the model is {:.2f}".format(model_evaluation_score/len(prediction)) )
    return nli_scores, nli_losses


def mis_match_loss_max(pred, verif_labels, mave_mask, args):
    bce = nn.BCELoss(reduction='none')
    num_answer = pred.size(1)
    num_correct_answers = (verif_labels > 0.).long().sum(1)  # batch
    losses2 = bce(pred[:, np.arange(num_answer), np.arange(
        num_answer)], verif_labels.float()).sum(1).mean() * args.mml2

    diag_mask = np.ones((pred.size(0), 5, 5))
    for i in range(pred.size(0)):
        diag_mask[i, :num_correct_answers[i], :num_correct_answers[i]] = 0.
    diag_mask = torch.from_numpy(diag_mask).float().cuda()
    diag_mask[:, np.arange(5), np.arange(5)] = 0.
    diag_mask = diag_mask * mave_mask.view(pred.size(0), 1, -1)

    diag_loss = bce(pred,  torch.zeros(
        (verif_labels.size(0), num_answer, num_answer)).float().cuda())
    diag_loss = diag_loss * diag_mask
    losses1 = diag_loss.max(1)[0]  # batch 5
    losses1 = (losses1.sum(1) / ((losses1 > 0.).sum(1) + 0.000001)
               ).mean() * args.mml0

    losses0 = diag_loss.max(2)[0]  # batch 5
    losses0 = (losses0.sum(1) / ((losses0 > 0.).sum(1) + 0.000001)
               ).mean() * args.mml0

    losses = losses2 + losses1 + losses0

    return losses

# def bce_concept_loss(pred, verif_labels):
#     bce = nn.BCELoss(reduction='none')
#     num_answer = pred.size(1)
#     num_correct_answers = (verif_labels > 0.).long().sum(1) # batch
#     losses2 = bce(pred[:, np.arange(num_answer), np.arange(num_answer)], verif_labels.float()).sum(1).mean() * args.mml2

#     diag_mask = np.ones((pred.size(0), 5, 5))
#     for i in range(pred.size(0)):
#         diag_mask[i, :num_correct_answers[i], :num_correct_answers[i]] = 0.
#     diag_mask = torch.from_numpy(diag_mask).float().cuda()
#     diag_mask[:, np.arange(5), np.arange(5)] = 0.
#     diag_mask = diag_mask *  mave_mask.view(pred.size(0), 1, -1)

#     diag_loss = bce(pred,  torch.zeros((verif_labels.size(0), num_answer, num_answer)).float().cuda())
#     diag_loss = diag_loss * diag_mask
#     losses1 = diag_loss.max(1)[0]  # batch 5
#     losses1 = (losses1.sum(1) / ((losses1 > 0.).sum(1)+ 0.000001)).mean()* args.mml0

#     losses0 = diag_loss.max(2)[0]  # batch 5
#     losses0 = (losses0.sum(1) / ((losses0 > 0.).sum(1) + 0.000001)).mean()* args.mml0

#     losses = losses2 + losses1 + losses0

#     return losses


def ForwardModelsVal(args, task_cfg, device, task_id, batch, model, task_losses, epochId):
    # GJR
    question_txt, target_txt, kg_sents, graph = batch[-4:]
    graph = graph.to('cuda')
    batch = tuple(t.cuda(device=device, non_blocking=True)
                  for t in batch if torch.is_tensor(t))
    features, spatials, image_mask, question, target, input_mask, segment_ids, co_attention_mask, question_id, \
        verif_inds, verif_concepts, target_scores, train_target_scores, target_mask, close_mask, p_open_mask, open_mask = batch

    task_tokens = question.new().resize_(
        question.size(0), 1).fill_(int(task_id[4:]))
    vil_pred, fullgraph_anchors, pruned_anchors, pred_features, pruned_pred_features, \
        end_point_idxs, pred_prob, keep_mask, node_ids, batch_sampled_paths = model(
            question,
            features,
            spatials,
            segment_ids,
            input_mask,
            question_ids=question_id,
            image_attention_mask=image_mask,
            co_attention_mask=co_attention_mask,
            task_ids=task_tokens,
            verif_inds=verif_inds,
            question_txt=question_txt,
            target_txt=target_txt,
            kg_sents=kg_sents,
            graph=graph,
            target_mask=target_mask,
            epoch_id=epochId,
        )

    batch_size = len(question_id)
    target_scores = target_scores.detach().cpu()
    if pruned_anchors is not None:
        pruned_anchors = pruned_anchors.detach().cpu()
    if fullgraph_anchors is not None:
        fullgraph_anchors = fullgraph_anchors.detach().cpu()
    if pred_features is not None:
        pred_features = pred_features.detach().cpu()
    if pruned_pred_features is not None:
        pruned_pred_features = pruned_pred_features.detach().cpu()

    if 'classification' in args.ablation:
        target_scores = verif_inds.cpu()

    # Compute CrossEntropy for anchor nodes or all nodes
    if keep_mask is not None:
        target_scores = target_scores[keep_mask==1.].reshape(keep_mask.shape[0], -1)
    if ('e2e' in args.ablation and epochId >= 40) or ('path' in args.ablation and 'path_pool' not in args.ablation):
        # indices_topk: Path ranking
        # endnode_indices_topk: Node ranking as result 
        endnode_indices_sorted = torch.sort(pred_prob, descending=True, dim=1).indices
        indices_topk, endnode_indices_topk = [], []
        for i in range(batch_size):
            end_points = end_point_idxs[i][endnode_indices_sorted[i]]
            indices_topk.append(end_points)
            endnode_topk = torch.tensor(list(OrderedDict.fromkeys(end_points.tolist())))
            endnode_topk = torch.cat((endnode_topk[:100], endnode_topk[-1].expand(100-len(endnode_topk))))
            endnode_indices_topk.append(endnode_topk)

        endnode_indices_topk = torch.stack(endnode_indices_topk).detach().cpu()
        indices_topk = torch.stack(indices_topk).detach().cpu()
        pred_nodes_topk = [[[id2concept[node_ids[i]] for i in batch_sampled_paths[row][idx.item()] if i>=0] for idx in idxs[:20]] for row, idxs in enumerate(endnode_indices_sorted)] # node path
        batch_path_acc_score_1, batch_path_acc_score_5, batch_path_acc_score_10, batch_path_acc_score_20, batch_close_acc_score_1, batch_partial_open_acc_score_1, batch_open_acc_score_1 = ComputeAccuracy10(target_scores, indices_topk, close_mask, p_open_mask, open_mask)
        batch_acc_score_1, batch_acc_score_10, batch_acc_score_20, batch_acc_score_50, batch_acc_score_100, batch_close_acc_score_1, batch_partial_open_acc_score_1, batch_open_acc_score_1 = ComputeAccuracy100(target_scores, endnode_indices_topk, close_mask, p_open_mask, open_mask)
    else:                    
        indices_topk = torch.topk(pred_prob, min(100, pred_prob.shape[1])).indices
        indices_topk = indices_topk.detach().cpu()
        pred_idxs_topk = [[pred_prob.shape[1] * i + idx for idx in idxs[:20]] for i, idxs in enumerate(indices_topk)]
        pred_nodes_topk = [[id2concept[node_ids[i]] for i in idxs] for idxs in pred_idxs_topk] # node text
        batch_acc_score_1, batch_acc_score_10, batch_acc_score_20, batch_acc_score_50, batch_acc_score_100, batch_close_acc_score_1, batch_partial_open_acc_score_1, batch_open_acc_score_1 = ComputeAccuracy100(
            target_scores, indices_topk,  close_mask, p_open_mask, open_mask)
        batch_path_acc_score_1, batch_path_acc_score_5, batch_path_acc_score_10, batch_path_acc_score_20 = 0, 0, 0, 0

    results = []
    for i in range(batch_size):
        results.append({
                        # 'vil_pred': vil_pred_np[i], 
                        'question_id': int(question_id[i]),  
                        'indices_topk': indices_topk[i],
                        'verif_concepts': verif_concepts[i],
                        # 'vil_pred_idx_np': vil_pred_idx_np[i],
                        # 'target_prob_np': target_prob_np[i],
                        # 'target_idx_np': target_idx_np[i],
                        # 'nli_entail':nli_score[i],
                        # 'trans_answers':trans_answers[i],
                        'pred_nodes': pred_nodes_topk[i],
                        })
    returned_variables = ['batch_path_acc_score_1', 'batch_path_acc_score_5', 'batch_path_acc_score_10', 'batch_path_acc_score_20',
                        'batch_size', 'results', 'batch_acc_score_1', 'batch_acc_score_10', 'batch_acc_score_20', 'batch_acc_score_50', 'batch_acc_score_100', 'batch_close_acc_score_1', 'batch_partial_open_acc_score_1', 'batch_open_acc_score_1']
 

    return_dict = {}
    for name in returned_variables:
        tmp = eval(name)
        if torch.is_tensor(tmp):
            tmp = tmp.detach().cpu().numpy()
        return_dict[name] = tmp
    return return_dict


def ForwardModelsTrain(
        args,
        task_cfg,
        device,
        task_id,
        task_count,
        task_iter_train,
        task_dataloader_train,
        model,
        task_losses,
        epochId,
):
    # given the current task, decided whether to forward the model and forward with specific loss.

    # reset the task iteration when needed.
    if task_count[task_id] % len(task_dataloader_train[task_id]) == 0:
        task_iter_train[task_id] = iter(task_dataloader_train[task_id])

    task_count[task_id] += 1
    batch = task_iter_train[task_id].next()
    # GJR
    # print('batch', len(batch))
    question_txt, target_txt, kg_sents, graph = batch[-4:]
    graph = graph.to('cuda')

    batch = tuple(t.cuda(device=device, non_blocking=True)
                  for t in batch if torch.is_tensor(t))
    # batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)

    # for t in batch :
    #     print(type(t[0]),  torch.is_tensor(t))
    features, spatials, image_mask, question, target, input_mask, segment_ids, co_attention_mask, question_id, \
        verif_inds, verif_concepts, target_scores, train_target_scores, target_mask, _, _, _ = batch

    task_tokens = question.new().resize_(
        question.size(0), 1).fill_(int(task_id[4:]))
    # bce_with_logits_loss = nn.BCEWithLogitsLoss(reduction="none")


    vil_pred, fullgraph_anchors, pruned_anchors, pred_features, pruned_pred_features, \
        end_point_idxs, pred_prob, keep_mask, node_ids, batch_sampled_paths  = model(
            question,
            features,
            spatials,
            segment_ids,
            input_mask,
            question_ids=question_id,
            image_attention_mask=image_mask,
            co_attention_mask=co_attention_mask,
            task_ids=task_tokens,
            verif_inds=verif_inds,
            question_txt=question_txt,
            target_txt=target_txt,
            kg_sents=kg_sents,
            graph=graph,
            target_mask=target_mask,
            epoch_id=epochId,
        )
    checktime = False
    # GJRTIME
    if checktime:
        torch.cuda.synchronize()
        start = time.time()

    # print('fullgraph_anchors, pred_features', fullgraph_anchors.shape, pred_features.shape)
    
    triplet_loss = TripletLoss(margin=args.margin_prune)
    loss_prune = triplet_loss(fullgraph_anchors, pred_features, train_target_scores, semi=False, woanchor=False, minneg=False)
    
    loss_zero = torch.tensor([0.0], requires_grad=True).cuda()
    loss_zero.backward()
    loss_tri, loss_cls = loss_zero, loss_zero
                  
    # indices_topk = indices_topk.detach().cpu()
    # pred_idxs_topk = [[pred_prob.shape[1] * i + idx for idx in idxs[:20]] for i, idxs in enumerate(indices_topk)]
    # pred_nodes_topk = [[id2concept[node_ids[i]] for i in idxs] for idxs in pred_idxs_topk] # node text
    # batch_acc_score_1, batch_acc_score_10, batch_acc_score_20, batch_acc_score_50, batch_acc_score_100, batch_close_acc_score_1, batch_partial_open_acc_score_1, batch_open_acc_score_1 = ComputeAccuracy100(
    #     target_scores, indices_topk,  close_mask, p_open_mask, open_mask)
    # batch_path_acc_score_1, batch_path_acc_score_5, batch_path_acc_score_10, batch_path_acc_score_20, batch_close_acc_score_1, batch_partial_open_acc_score_1, batch_open_acc_score_1 = 0, 0, 0, 0, 0, 0, 0

    if 'e2e' in args.ablation and epochId < 40:
        pass
    else:
        if 'prune' in args.ablation:
            # triplet_loss = TripletLoss(margin=args.margin_prune)
            # loss_prune = triplet_loss(fullgraph_anchors, pred_features, train_target_scores, semi=True, woanchor=False, minneg=False)
            # a = len((train_target_scores==1.).int().sum(1).nonzero())
            train_target_scores = train_target_scores[keep_mask==1.].reshape(keep_mask.shape[0], -1)
            # print('train_target_scores: %d in  %d / %d retrieved,  %d / %d has gt' % (len((train_target_scores==1.).int().sum(1).nonzero()), len(torch.unique(train_target_scores.nonzero(as_tuple=True)[0])), a,  a, train_target_scores.shape[0]))

        if 'path' in args.ablation and 'path_pool' not in args.ablation:
            path_scores, resampled_path_scores, resampled_pred_prob, resampled_pred_features = [], [], [], []
            # qv node scores
            # qv_node_scores = (0.8+0.1*qv_node_scores).reshape(train_target_scores.shape[0], -1)
            c = 0
            k = int(end_point_idxs.size(1) * 0.5)
            c_pos = 0
            for i in range(train_target_scores.shape[0]):
                t = train_target_scores[i,:]
                path_score = t[end_point_idxs[i,:]] # + t[start_point_idxs[i,:]] / 2.0
                # path_scores *= qv_node_scores[i]
                # pad a ground truth if no gt retrieved
                if sum(path_score) <= 0:
                    c += 1
                #     path_scores[torch.randint(0,path_scores.shape[0], (1,))] = 0.001    
                path_scores.append(path_score)

                neg_idxs = (path_score==0).nonzero(as_tuple=True)[0]  
                c_pos += end_point_idxs.shape[1] - len(neg_idxs)
                # Resample positive and negative samples
                # neg_mask = torch.zeros(len(path_score)).cuda()
                # if len(neg_idxs) > k+1:
                #     perm = torch.randperm(len(neg_idxs))
                #     neg_mask.scatter_(0, neg_idxs[perm[:k]], 1.)
                # else:
                #     perm = torch.randperm(len(path_score)).cuda()
                #     neg_mask.scatter_(0, perm[:k], 1.)
                # neg_mask = neg_mask.bool()
                # new_path_score = path_score[~neg_mask]
                # if sum(path_score) <= 0:
                #     new_path_score[torch.randint(0,new_path_score.shape[0], (1,))] = 1    
                # resampled_path_scores.append(new_path_score)
                # resampled_pred_prob.append(pred_prob[i][~neg_mask])
                # resampled_pred_features.append(pruned_pred_features[i][~neg_mask])

            # create balanced positive negative samples 
            if len(resampled_path_scores) == len(path_scores):
                train_target_scores = torch.stack(resampled_path_scores)
                pred_prob = torch.stack(resampled_pred_prob)
                pruned_pred_features = torch.stack(resampled_pred_features)
            else:
                train_target_scores = torch.stack(path_scores)
            print('no gt path: %d / %d, %d / %d positive' % (c, train_target_scores.shape[0], int(c_pos/train_target_scores.shape[0]), train_target_scores.shape[1]))
            # print('gt_paths: %.2f / %d' % (train_target_scores.sum(), train_target_scores.shape[0]*train_target_scores.shape[1]))
            # print('train_target_scores', train_target_scores.shape)

        if 'tri' in args.ablation:
            # triplet_loss = MultipleNegativesRankingLoss(scale=20.0)
            triplet_loss = TripletLoss(margin=args.margin_tri)
            loss_tri = triplet_loss(pruned_anchors, pruned_pred_features, train_target_scores, semi=True, woanchor=False, minneg=False)
            # triplet_loss = SupContrastiveLoss()
            # loss_tri = triplet_loss(pruned_pred_features, train_target_scores)

            # GJRTIME
            if checktime:
                torch.cuda.synchronize()
                end = time.time()
                print('final', end-start)
                start = end

        if 'bce' in args.ablation:
            # pred_prob = pred_prob / 0.1 
            # print('pred_prob', pred_prob)
            loss_cls = LossMap['BCEWithLogitLoss'](pred_prob, train_target_scores).sum(1).mean()
            # criterion_ce = nn.CrossEntropyLoss()
            # loss_cls = criterion_ce(pred_prob, train_target_scores)
            # if MSELoss
            # criterion_mse = nn.MSELoss()
            # loss_prune = 0
            # for i in range(train_target_scores.shape[0]):
            #     for pos_emb in pruned_pred_features[i][train_target_scores[i] > 0]:
            #         loss_prune += criterion_mse(anchors[i], pos_emb)
            # If WBCE
            # wbce_loss = W_BCEWithLogitsLoss( 1 - 1 / train_target_scores.shape[1])
            # loss_cls = wbce_loss(pred_prob, train_target_scores)
                

        if 'classification' in args.ablation:
            train_target_scores = verif_inds
            criterion_cls = nn.CrossEntropyLoss() # better than bce
            criterion_mse = nn.MSELoss()
            # loss_prune = 0
            # for i in range(train_target_scores.shape[0]):
            #     for pos_emb in pruned_pred_features[i][train_target_scores[i] > 0]:
            #         loss_prune += criterion_mse(anchors[i], pos_emb)
            loss_cls = criterion_cls(pred_prob, train_target_scores)
            # triplet_loss = TripletLoss(margin=0.1)
            # loss_tri = triplet_loss(anchors, pruned_pred_features, train_target_scores, semi=False, woanchor=False, minneg=True)        


    indices_top1 = torch.max(pred_prob, 1).indices
    y = torch.tensor([train_target_scores[i][idx] for i, idx in enumerate(indices_top1)]).cuda()
    train_acc_1 = torch.sum(y) / (y.shape[0] * 1.0)

    loss = loss_prune + loss_tri + loss_cls
    return_dict = {}
    returned_variables = ['loss', 'loss_tri', 'loss_prune', 'loss_cls', 'train_acc_1']
    for name in returned_variables:
        return_dict[name] = eval(name)
    return return_dict

def ranknet_grad(
        score_predict: torch.Tensor,
        score_real: torch.Tensor,
) -> torch.Tensor:
    """
    Get loss from one user's score output
    :param score_predict: 1xN tensor with model output score
    :param score_real: 1xN tensor with real score
    :return: Gradient of ranknet
    """
    sigma = 1.0
    score_predict_diff_mat = score_predict - score_predict.t()
    score_real_diff_mat = score_real - score_real.t()
    tij = (1.0 + torch.sign(score_real_diff_mat)) / 2.0
    lambda_ij = torch.sigmoid(sigma * score_predict_diff_mat) - tij
    return lambda_ij.sum(dim=1, keepdim=True) - lambda_ij.t().sum(dim=1, keepdim=True)

def calc_euclidean(x1, x2):
    return (x1 - x2).pow(2).sum(1)

class W_BCEWithLogitsLoss(torch.nn.Module):
    
    def __init__(self, w_p = None):
        super(W_BCEWithLogitsLoss, self).__init__()
        
        self.w_p = w_p
        self.w_n = 1 - w_p
        
    def forward(self, logits, labels, epsilon = 1e-7):
        
        ps = torch.sigmoid(logits.squeeze()) 
        
        loss_pos = -1 * torch.mean(self.w_p * labels * torch.log(ps + epsilon))
        loss_neg = -1 * torch.mean(self.w_n * (1-labels) * torch.log((1-ps) + epsilon))
        
        loss = loss_pos + loss_neg
        
        return loss

class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    Args:
        margin (float): margin for triplet.
    """

    def __init__(self, margin=0.1, mutual_flag=False):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.mutual = mutual_flag

    def euclidean_dist(self, x, y):
        m,n = x.size(0),y.size(0)
        xx = torch.pow(x,2).sum(1,keepdim=True).expand(m,n)
        yy = torch.pow(y,2).sum(dim=1,keepdim=True).expand(n,m).t()
        dist = xx + yy
        dist.addmm_(1,-2,x,y.t())
        dist = dist.clamp(min=1e-12).sqrt()
        return dist
    
    def forward_row(self, anc, inputs, targets):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        anc = 1. * anc / (torch.norm(anc, 2, dim=-1, keepdim=True).expand_as(anc) + 1e-12)
        inputs = 1. * inputs / (torch.norm(inputs, 2, dim=-1, keepdim=True).expand_as(inputs) + 1e-12)
        dist = self.euclidean_dist(anc, inputs)
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        # hard negative
        # for i in range(n):
        #     dist_ap.append(dist[0][mask[i]].max().unsqueeze(0)) # dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
        #     dist_an.append(dist[0][mask[i]==0].min().unsqueeze(0))
        # semi-hard
        p_dist = dist[0][targets > 0]
        n_dist = dist[0][targets == 0]
        if len(p_dist) <= 0:
            return self.margin
        for i in range(len(p_dist)):
            # for j in torch.topk(n_dist, n_dist.shape[0]//2, largest=False).indices:
            dist_ap.append(p_dist[i].unsqueeze(0)) 
            dist_an.append(n_dist.min().unsqueeze(0))
        # 
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        if self.mutual:
            return loss, dist
        return loss

    def forward_row_hard(self, anc, inputs, targets):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        anc = 1. * anc / (torch.norm(anc, 2, dim=-1, keepdim=True).expand_as(anc) + 1e-12)
        inputs = 1. * inputs / (torch.norm(inputs, 2, dim=-1, keepdim=True).expand_as(inputs) + 1e-12)
        # Compute pairwise distance, replace by the official when merged
        dist = self.euclidean_dist(anc, inputs)
        # dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        # dist = dist + dist.t()
        # dist.addmm_(1, -2, inputs, inputs.t())
        # dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative

        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        # hard negative
        # for i in range(n):
        #     dist_ap.append(dist[0][mask[i]].max().unsqueeze(0)) # dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
        #     dist_an.append(dist[0][mask[i]==0].min().unsqueeze(0))
        # semi-hard
        p_dist = dist[0][targets > 0]
        n_dist = dist[0][targets == 0]
        if len(p_dist) <= 0:
            return self.margin
        for i in range(len(p_dist)):
            # for j in torch.topk(n_dist, n_dist.shape[0]//2, largest=False).indices:
            for j in range(len(n_dist)):
                dist_ap.append(p_dist[i].unsqueeze(0)) 
                dist_an.append(n_dist[j].unsqueeze(0))
        # 
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        if self.mutual:
            return loss, dist
        return loss
        

    def forward_row_hard_semi(self, anc, inputs, targets):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        anc = 1. * anc / (torch.norm(anc, 2, dim=-1, keepdim=True).expand_as(anc) + 1e-12)
        inputs = 1. * inputs / (torch.norm(inputs, 2, dim=-1, keepdim=True).expand_as(inputs) + 1e-12)
        dist = self.euclidean_dist(anc, inputs)
        dist = dist[0]
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_an_ap = dist.unsqueeze(0) - dist.unsqueeze(1)
        dist_ap, dist_an = [], []
        # hard negative
        if targets.sum() == 0:
            return self.margin
        p_idxs = torch.arange(n)[targets > 0]
        hard_mask =  (mask == 0 ) & (dist_an_ap < self.margin)# & (dist_an_ap > 0)
        # all hard 
        for i in p_idxs:
            # & dist_ij > 0 
            hard_idxs = torch.arange(n)[hard_mask[i]]
            if len(hard_idxs):
                for j in hard_idxs:
                    dist_ap.append(dist[i].unsqueeze(0)) 
                    dist_an.append(dist[j].unsqueeze(0))
            dist_ap.append(dist[i].unsqueeze(0)) 
            dist_an.append(dist[mask[i]==0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        if self.mutual:
            return loss, dist
        return loss

    def forward_row_hard_ancless(self, inputs, targets):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        inputs = 1. * inputs / (torch.norm(inputs, 2, dim=-1, keepdim=True).expand_as(inputs) + 1e-12)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        # hard negative
        if targets.sum() <= 1:
            return self.margin
        # n_dist = dist[0][targets == 0]
        p_idxs = targets.nonzero().squeeze()
        
        
        # for i in range(len(p_dist)):
        for i in p_idxs:
            for j in p_idxs:
                if i == j:
                    continue
                # all hard 
                dist_ij = (dist[i][mask[i]==0] - dist[i][j])
                hard_mask = dist_ij < self.margin # & dist_ij > 0 
                if hard_mask.sum() > 0:
                    for dan in dist[i][mask[i]==0][hard_mask]:
                        dist_ap.append(dist[i][j].unsqueeze(0)) 
                        dist_an.append(dan.unsqueeze(0))
                # semi-hard
                # semi_mask = (dist[i][mask[i]==0] > dist[i][j])
                # if semi_mask.sum() > 0:
                #     dist_ap.append(dist[i][j].unsqueeze(0)) 
                #     dist_an.append(dist[i][mask[i]==0][semi_mask].min().unsqueeze(0))
                # # hard
                # dist_ap.append(dist[i][j].unsqueeze(0)) 
                # dist_an.append(dist[i][mask[i]==0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        return self.ranking_loss(dist_an, dist_ap, y)

    def forward(self, anchors, pred_features, target_scores, semi=False, woanchor=False, minneg=False):
        # anchors:       batch x 1024
        # pred_features: batch x 100 x 1024
        # target_scores: batch x 100
        loss = 0
        num = 0
        assert pred_features.shape[1] == target_scores.shape[1]
        for row in range(target_scores.shape[0]):
            scores = target_scores[row,:]
            pred_feature = pred_features[row,:,:]
            anc = anchors[row, :].unsqueeze(0)
            if minneg:
                loss += self.forward_row(anc, pred_feature, scores)
                continue
            if woanchor:
                loss += self.forward_row_hard_ancless(pred_feature, scores)
            else:
                if semi:
                    loss += self.forward_row_hard_semi(anc, pred_feature, scores)
                else:
                    loss += self.forward_row_hard(anc, pred_feature, scores)
        return loss


from torch import Tensor
def cos_sim(a: Tensor, b: Tensor):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))

class MultipleNegativesRankingLoss(nn.Module):
    def __init__(self, scale: float = 20.0, similarity_fct = cos_sim):
        super(MultipleNegativesRankingLoss, self).__init__()
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward_row(self, inputs, targets):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        # inputs = 1. * inputs / (torch.norm(inputs, 2, dim=-1, keepdim=True).expand_as(inputs) + 1e-12)
        # Compute pairwise distance, replace by the official when merged
        embeddings_a = inputs[targets > 0][:]
        embeddings_b = inputs[targets == 0][:]
        if len(embeddings_a) <= 0:
            return self.scale
        scores = self.similarity_fct(embeddings_a, embeddings_b) * self.scale
        labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)  # Example a[i] should match with b[i]
        loss = self.cross_entropy_loss(scores, labels)

        # mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        # dist_ap, dist_an = [], []
        # # hard negative
        # # for i in range(n):
        # #     dist_ap.append(dist[0][mask[i]].max().unsqueeze(0)) # dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
        # #     dist_an.append(dist[0][mask[i]==0].min().unsqueeze(0))
        # # semi-hard
        # if len(p_dist) <= 0:
        #     return self.margin
        # for i in range(len(p_dist)):
        #     # for j in torch.topk(n_dist, n_dist.shape[0]//2, largest=False).indices:
        #     for j in range(len(n_dist)):

        #         dist_ap.append(p_dist[i].unsqueeze(0)) 
        #         dist_an.append(n_dist[j].unsqueeze(0))
        # # 
        # dist_ap = torch.cat(dist_ap)
        # dist_an = torch.cat(dist_an)
        # # Compute ranking hinge loss
        # y = torch.ones_like(dist_an)
        # loss = self.ranking_loss(dist_an, dist_ap, y)
        # if self.mutual:
        #     return loss, dist
        return loss
        

    def forward(self, anchors, pred_features, target_scores, semi=False, woanchor=False):
        # anchors:       batch x 1024
        # pred_features: batch x 100 x 1024
        # target_scores: batch x 100
        loss = 0
        num = 0
        assert pred_features.shape[1] == target_scores.shape[1]
        for row in range(target_scores.shape[0]):
            scores = target_scores[row,:]
            pred_feature = pred_features[row,:,:]
            loss += self.forward_row(pred_feature, scores)
        return loss

from pytorch_metric_learning import losses


class SupContrastiveLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.margin = 1


    def forward_row(self, projections, targets):
        """
        :param projections: torch.Tensor, shape [batch_size, projection_dim]
        :param targets: torch.Tensor, shape [batch_size]
        :return: torch.Tensor, scalar
        """
        device = torch.device("cuda") if projections.is_cuda else torch.device("cpu")

        dot_product_tempered = torch.mm(projections, projections.T) / self.temperature
        # # Minus max for numerical stability with exponential. Same done in cross entropy. Epsilon added to avoid log(0)
        exp_dot_tempered = (
            torch.exp(dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0]) + 1e-5
        )

        mask_similar_class = (targets.unsqueeze(1).repeat(1, targets.shape[0]) == targets).to(device)
        mask_anchor_out = (1 - torch.eye(exp_dot_tempered.shape[0])).to(device)
        mask_combined = mask_similar_class * mask_anchor_out
        cardinality_per_samples = torch.sum(mask_combined, dim=1) + 1e-5

        log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered * mask_anchor_out, dim=1, keepdim=True)))
        supervised_contrastive_loss_per_sample = torch.sum(log_prob * mask_combined, dim=1) / cardinality_per_samples
        supervised_contrastive_loss = torch.mean(supervised_contrastive_loss_per_sample)

        return supervised_contrastive_loss

    def forward_row_1(self, feature_vectors, labels):
        # Normalize feature vectors
        feature_vectors_normalized = F.normalize(feature_vectors, p=2, dim=1)
        # Compute logits
        logits = torch.div(
            torch.matmul(
                feature_vectors_normalized, torch.transpose(feature_vectors_normalized, 0, 1)
            ),
            self.temperature,
        )
        return losses.NTXentLoss(temperature=0.07)(logits, labels)

    def forward_pair(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        pos = (1-label) * torch.pow(euclidean_distance, 2)
        neg = (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        loss_contrastive = torch.mean( pos + neg )
        return loss_contrastive

    def forward_row_2(self, features, labels=None, mask=None):
        loss = 0
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        mask = torch.eq(labels, labels.T).float().to(device)
        for i in range(mask.shape[0]):
            for j in range(i, mask.shape[1]):
                loss += self.forward_pair(features[i], features[j], mask[i][j])
        return loss

    def forward_row_3(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                    if features.is_cuda
                    else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                                'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        # logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        # logits = anchor_dot_contrast - logits_max.detach()
        logits = anchor_dot_contrast

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)


        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

    
    def forward(self, pred_features, target_scores):
        # anchors:       batch x 1024
        # pred_features: batch x 100 x 1024
        # target_scores: batch x 100
        loss = 0
        num = 0
        assert pred_features.shape[1] == target_scores.shape[1]
        for row in range(target_scores.shape[0]):
            scores = target_scores[row,:]
            pred_feature = pred_features[row,:,:]
            loss += self.forward_row(pred_feature, scores)
            # loss += self.forward_row_1(pred_feature, scores)
        return loss


def LoadLosses(args, task_cfg, task_ids):
    losses = {}
    task_types = []
    num_labels = 0
    for i, task_id in enumerate(task_ids):
        task = "TASK" + task_id
        model_type = task_cfg[task]["type"]
        if model_type not in task_types:
            task_types.append(model_type)
        losses[task] = LossMap[task_cfg[task]["loss"]]

    return losses


def collate_batch(batch):
    # subgraphs = [t[-1] for t in batch ]
    batch = list(zip(*batch))
    # features, spatials, image_mask, question, target, input_mask, segment_ids, co_attention_mask, question_id, \
    #     wiki_tokens, concept_tokens, google_images, verif_labels, verif_inds, question_txt, target_txt, kg_sents, subgraphs = zip(*batch)
    batch_data = [torch.stack(t) if torch.is_tensor(
        t[0]) else torch.tensor(t) for t in batch[:-4]]
    batch_data.extend(batch[-4:-1])
    batch_data.append(dgl.batch(batch[-1]))
    return batch_data


def LoadDatasets(args, task_cfg, ids, split="trainval"):
    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=args.do_lower_case
    )
    task_datasets_train = {}
    task_datasets_val = {}
    task_dataloader_train = {}
    task_dataloader_val = {}
    task_ids = []
    task_batch_size = {}
    task_num_iters = {}
    from vilbert.datasets.okvqa_mavex_dataset import VQAClassificationDataset

    for i, task_id in enumerate(ids):
        task = "TASK" + task_id
        task_name = task_cfg[task]["name"]
        task_ids.append(task)
        batch_size = task_cfg[task]["batch_size"] // args.gradient_accumulation_steps
        num_workers = args.num_workers
        if args.local_rank != -1:
            batch_size = int(batch_size / dist.get_world_size())
            num_workers = int(num_workers / dist.get_world_size())

        logger.info(
            "Loading %s Dataset with batch size %d"
            % (task_cfg[task]["name"], batch_size)
        )

        task_datasets_train[task] = None
        if "train" in split:
            task_datasets_train[task] = VQAClassificationDataset(
                task=task_cfg[task]["name"],
                dataroot=task_cfg[task]["dataroot"],
                annotations_jsonpath=task_cfg[task]["train_annotations_jsonpath"],
                split=task_cfg[task]["train_split"],
                image_features_reader={},
                gt_image_features_reader={},
                tokenizer=tokenizer,
                bert_model=args.bert_model,
                clean_datasets=args.clean_train_sets,
                padding_index=0,
                max_seq_length=task_cfg[task]["max_seq_length"],
                max_region_num=task_cfg[task]["max_region_num"], args=args,
            )

        task_datasets_val[task] = None
        if "val" in split:
            task_datasets_val[task] = VQAClassificationDataset(
                task=task_cfg[task]["name"],
                dataroot=task_cfg[task]["dataroot"],
                annotations_jsonpath=task_cfg[task]["val_annotations_jsonpath"],
                split=task_cfg[task]["val_split"],
                image_features_reader={},
                gt_image_features_reader={},
                tokenizer=tokenizer,
                bert_model=args.bert_model,
                clean_datasets=args.clean_train_sets,
                padding_index=0,
                max_seq_length=task_cfg[task]["max_seq_length"],
                max_region_num=task_cfg[task]["max_region_num"], args=args,
            )

        # Schema Graph Generation
        doPreprocess = False
        if doPreprocess:
            print('Preprocessing')
            task_datasets_val[task].write_graphs()
            task_datasets_train[task].write_graphs()
        checkGroundTruths = False
        if checkGroundTruths:
            print('Checking Ground Truths')
            task_datasets_val[task].check_GT()
            task_datasets_train[task].check_GT()
            # task_datasets_val[task].find_answer_hops()
            # task_datasets_train[task].find_answer_hops()

        task_num_iters[task] = 0
        task_batch_size[task] = 0
        if "train" in split:
            if args.local_rank == -1:
                train_sampler = RandomSampler(task_datasets_train[task])
            else:
                train_sampler = DistributedSampler(task_datasets_train[task])

            task_dataloader_train[task] = DataLoader(
                task_datasets_train[task],
                sampler=train_sampler,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=collate_batch,
            )

            task_num_iters[task] = len(task_dataloader_train[task])
            task_batch_size[task] = batch_size

        if "val" in split:
            task_dataloader_val[task] = DataLoader(
                task_datasets_val[task],
                shuffle=False,
                batch_size=16,
                num_workers=1,
                pin_memory=True,
                collate_fn=collate_batch,
            )

    return (
        task_batch_size,
        task_num_iters,
        task_ids,
        task_datasets_train,
        task_datasets_val,
        task_dataloader_train,
        task_dataloader_val,
    )


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros(*labels.size()).to(logits.device)
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = one_hots * labels
    return scores


def compute_score_with_pred_txt(pred_answers, targets):
    if not isinstance(pred_answers[0], str):
        pred_answers = [p[0] for p in pred_answers]
    scores = 0
    for pred, target in zip(pred_answers, targets):
        target = target.split(',')
        pred = pred.replace('_', ' ')
        edit_dists = [levenshtein_distance(pred, t) for t in target]
        scores = sum(map(lambda x: x < 3, edit_dists))
    # scores = scores / len(pred_answers)
    return scores


def compute_score_with_pred_ids(pred_ids, target_ids):
    # scores = 0
    # for pred, target in zip(pred_ids, target_ids):
    #     pred = pred[0]
    #     if pred in target:
    #         scores += 1
    # scores = scores / len(pred_ids)
    from sklearn.metrics import f1_score
    pred_ids = torch.stack([p[0] for p in pred_ids]).cpu()
    target_ids = target_ids.squeeze().cpu()
    scores = f1_score(target_ids, pred_ids, average='micro')
    return scores


def ComputeAccuracy100(targets, indices_topk, close_mask, p_open_mask, open_mask):
    # logits = logits.detach()
    y = torch.stack([torch.tensor([targets[i][idx] for idx in idxs])
                    for i, idxs in enumerate(indices_topk)]).cuda()


    # # get the index of the max log-probability, batch_size
    # pred = logits.max(1, keepdim=False)[1].cpu()
    # pred_top5 = torch.topk(logits, 5)
    # # batch_size
    # correct = sum([t[p] for p,t in zip(pred, y)])
    top1_y = y[:, 0]
    top10_y = y[:, :10]
    top20_y = y[:, :20]
    top50_y = y[:, :50]
    top100_y = y[:, :100]
    # print('top10_y', top10_y)
    # 计算最高的分数
    correct_top1 = top1_y.sum()
    correct_top10 = torch.max(top10_y, 1).values.sum()
    correct_top20 = torch.max(top20_y, 1).values.sum()
    correct_top50 = torch.max(top50_y, 1).values.sum()
    correct_top100 = torch.max(top100_y, 1).values.sum()


    close_y = (y.t() * close_mask.int()).t()
    close_correct_top1 = close_y[:, 0].sum()

    partial_open_y = (y.t() * p_open_mask.int()).t()
    partial_open_correct_top1 = partial_open_y[:, 0].sum()

    open_y = (y.t() * open_mask.int()).t()
    # open_correct_top1 = open_y[:, 0].sum()
    open_correct_top1 = torch.max(open_y[:, :10], 1).values.sum()
    s = len(y)
    # print(correct_top5)
    # for i, doc in enumerate(docs):
    #     e = s + len(doc.mentions)
    #     d_mention_slice = mention_correct[s:e]
    #     s = e
    #     total_mentions.append(doc.total_mentions)
    #     actual_mentions.append(len(doc.mentions))
    #     actual_correct.append(d_mention_slice.sum())

    # return total_mentions, actual_mentions, actual_correct
    return correct_top1, correct_top10, correct_top20, correct_top50, correct_top100, close_correct_top1, partial_open_correct_top1, open_correct_top1

def ComputeAccuracy10(targets, indices_topk, close_mask, p_open_mask, open_mask):
    y = torch.stack([torch.tensor([targets[i][idx] for idx in idxs])
                    for i, idxs in enumerate(indices_topk)]).cuda()

    top1_y = y[:, 0]
    top5_y = y[:, :5]
    top10_y = y[:, :10]
    top20_y = y[:, :20]
    # 计算最高的分数
    correct_top1 = top1_y.sum()
    correct_top5 = torch.max(top5_y, 1).values.sum()
    correct_top10 = torch.max(top10_y, 1).values.sum()
    correct_top20 = torch.max(top20_y, 1).values.sum()

    # zero_y = (y.t() * zeroval_mask.int()).t()
    # zero_correct_top1 = zero_y[:, 0].sum()
    # zero_correct_top10 = torch.max(zero_y[:, :10], 1).values.sum()

    close_y = (y.t() * close_mask.int()).t()
    close_correct_top1 = close_y[:, 0].sum()

    partial_open_y = (y.t() * p_open_mask.int()).t()
    partial_open_correct_top1 = partial_open_y[:, 0].sum()

    open_y = (y.t() * open_mask.int()).t()
    # open_correct_top1 = open_y[:, 0].sum()
    open_correct_top1 = torch.max(open_y[:, :10], 1).values.sum()

    return correct_top1, correct_top5, correct_top10, correct_top20, close_correct_top1, partial_open_correct_top1, open_correct_top1


def pr_loss(scores, targets, lamb=1e-7):
    # scores = scores.resize_(0)
    true_pos = targets.argmax(1)
    scores = torch.tensor(scores, dtype=torch.float)
    loss = F.multi_margin_loss(scores, true_pos, margin=0.5)

    # regularization
    # X = F.normalize(self.rel_embs)
    # diff = (X.view(self.n_rels, 1, -1) - X.view(1, self.n_rels, -1)).pow(2).sum(dim=2).add_(1e-5).sqrt()
    # diff = diff * (diff < 1).float()
    # loss -= torch.sum(diff).mul(lamb)

    # X = F.normalize(self.ew_embs)
    # diff = (X.view(self.n_rels, 1, -1) - X.view(1, self.n_rels, -1)).pow(2).sum(dim=2).add_(1e-5).sqrt()
    # diff = diff * (diff < 1).float()
    # loss -= torch.sum(diff).mul(lamb)
    return loss


def list_mle(y_pred, y_true, eps=1e-12, pad_value_indicator=-1):
    """
    ListMLE loss introduced in "Listwise Approach to Learning to Rank - Theory and Algorithm".
    Args:
        y_pred (torch.FloatTensor): predictions from the model, of shape [batch_size, list_size]
        y_true (torch.FloatTensor): ground truth labels, of shape [batch_size, list_size]
        eps (float): epsilon value, used for numerical stability
        pad_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    Returns:
        torch.Tensor: loss value
    """
    # shuffle for randomised tie resolution
    random_indices = torch.randperm(y_pred.shape[-1])
    y_pred_shuffled = y_pred[:, random_indices]
    y_true_shuffled = y_true[:, random_indices]

    y_true_sorted, indices = y_true_shuffled.sort(descending=True, dim=-1)

    pred_sorted_by_true = y_pred_shuffled.gather(dim=1, index=indices)

    mask = y_true_sorted == pad_value_indicator
    pred_sorted_by_true[mask] = float("-inf")

    max_pred_scores, _ = pred_sorted_by_true.max(dim=1, keepdim=True)

    pred_sorted_by_true_minus_max = pred_sorted_by_true - max_pred_scores

    cumsums = pred_sorted_by_true_minus_max.exp().flip(dims=[1]).cumsum(dim=1).flip(dims=[1])

    observation_loss = torch.log(cumsums + eps) - pred_sorted_by_true_minus_max

    observation_loss[mask] = 0.0

    return observation_loss.sum(dim=1).mean()

from itertools import product
from torch.nn import BCEWithLogitsLoss, MarginRankingLoss
def pairwise_hinge(y_pred, y_true, padded_value_indicator=-1):
    """
    RankNet loss introduced in "Learning to Rank using Gradient Descent".
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param padded_value_indicator:
    :return: loss value, a torch.Tensor
    """
    y_pred = y_pred.clone()
    y_true = y_true.clone()

    mask = y_true == padded_value_indicator
    y_pred[mask] = float('-inf')
    y_true[mask] = float('-inf')

    # here we generate every pair of indices from the range of document length in the batch
    document_pairs_candidates = list(product(range(y_true.shape[1]), repeat=2))

    pairs_true = y_true[:, document_pairs_candidates]
    pairs_pred = y_pred[:, document_pairs_candidates]

    # here we calculate the relative true relevance of every candidate pair
    true_diffs = pairs_true[:, :, 0] - pairs_true[:, :, 1]

    # here we filter just the pairs that are 'positive' and did not involve a padded instance
    # we can do that since in the candidate pairs we had symetric pairs so we can stick with
    # positive ones for a simpler loss function formulation
    the_mask = (true_diffs > 0) & (~torch.isinf(true_diffs))

    s1 = pairs_pred[:, :, 0][the_mask]
    s2 = pairs_pred[:, :, 1][the_mask]
    target = the_mask.float()[the_mask]

    return MarginRankingLoss(margin=1, reduction='mean')(s1, s2, target)
