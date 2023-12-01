# Preprocess

### Question Chunking

```
qid2qchunks = cPickle.load(open('./OK-VQA/myAnnotations/new_val_split/%s_question_chunk_entities.pkl' % (new_split), 'rb'))
```

### Compute Concept Similarities

```

from distutils.log import info
import jieba  
import jieba.posseg as pseg  
import os  
import glob
from tqdm import tqdm
import pandas as pd
import numpy as np
import json
from collections import Counter
import Levenshtein
from text2vec import SentenceModel, EncoderType, semantic_search
import jieba
import heapq
import jieba.analyse
import pandas as pd
from scipy.spatial.distance import cdist

t2v_model = SentenceModel("./huggingface/paraphrase-multilingual-MiniLM-L12-v2", encoder_type=EncoderType.MEAN)

concept_nodes = id2concept
nodes_vec = t2v_model.encode(concept_nodes) 
scores = cdist(nodes_vec, nodes_vec, 'cosine')
scores = np.around(scores)
np.savetxt('./ConceptNet/numberbatch-19.08-dist.txt', scores)

ss = np.loadtxt('./ConceptNet/numberbatch-19.08-dist.txt')
ConceptNet 516782

```





1. Go to `task_utils.py`

   Change line 1022
   
   
   ```
   doPreprocess = True
   ```
   
2. Go to `okvqa_mavex_dataset`

  Change line 145 

  ```
  zeroshot = True
  ```

  Change line 607

  ```
  save_graphs("./OK-VQA/myAnnotations/subgraphs/%s_graphs_relsorted_1000%s.bin" % (self.new_split, self.postfix), graphs, graph_labels)
  ```

3. Go to `conceptnet.py`

   Do nothing

### Restructure Schema Graph

```
import json
import torch
train_questions = sorted( json.load( open( './OK-VQA/myAnnotations/new_val_split_backup/OpenEnded_mscoco_train2014_questions_in_ConceptNet.json'))["questions"], key=lambda x: x["question_id"])
val_questions = sorted( json.load( open( './OK-VQA/myAnnotations/new_val_split_backup/OpenEnded_mscoco_val2014_questions_in_ConceptNet.json'))["questions"], key=lambda x: x["question_id"])
qid2index = {}
qids = []
for i,x in enumerate(train_questions):
	qid2index[x['question_id']] = i 
	qids.append(x['question_id'])
	
for i,x in enumerate(val_questions):
	qid2index[x['question_id']] = i + len(train_questions)
	qids.append(x['question_id'])
	
graph_labels = {'question_id': torch.tensor(qids)}

from dgl.data.utils import load_graphs, save_graphs
train_graph_list, label_dict = load_graphs("./OK-VQA/myAnnotations/subgraphs_old/train_graphs_relsorted_1000_in_ConceptNet.bin")
val_graph_list, label_dict = load_graphs("./OK-VQA/myAnnotations/subgraphs_old/val_graphs_relsorted_1000_in_ConceptNet.bin")
train_graph_list.extend(val_graph_list)
all_graph_list = train_graph_list

new_train_questions = sorted( json.load( open( './OK-VQA/myAnnotations/new_val_split/OpenEnded_mscoco_train2014_questions_in_ConceptNet.json'))["questions"], key=lambda x: x["question_id"])
new_val_questions = sorted( json.load( open( './OK-VQA/myAnnotations/new_val_split/OpenEnded_mscoco_val2014_questions_in_ConceptNet.json'))["questions"], key=lambda x: x["question_id"])

new_train_graphs = [all_graph_list[qid2index[x['question_id']]] for x in new_train_questions]
new_val_graphs = [all_graph_list[qid2index[x['question_id']]] for x in new_val_questions]

train_graph_labels = {'question_id': torch.tensor([x['question_id'] for i,x in enumerate(new_train_questions)])}
val_graph_labels = {'question_id': torch.tensor([x['question_id'] for i,x in enumerate(new_val_questions)])}

save_graphs("./OK-VQA/myAnnotations/schemagraphs/train_graphs_open_relsorted_1000.bin", new_train_graphs, train_graph_labels)
save_graphs("./OK-VQA/myAnnotations/schemagraphs/val_graphs_open_relsorted_1000.bin", new_val_graphs, val_graph_labels)
```





### Close Set label2spaceid

```
num_labels = 100000
answer_space_list = list(range(0, num_labels))
random.shuffle(answer_space_list)
label2spaceid = {int(x):answer_space_list[i] for i,x in enumerate(range(num_labels))} 
json.dump(label2spaceid, open('./OK-VQA/myAnnotations/new_val_split_full/closeset/label2spaceid_%s.json' % (num_labels), 'w'))
```

```
results = pickle.load(open('/home/jingrugan/KBVQA/MAVEX/save/old split/bert_base_6layer_6conect-100-path-qv-tri-lpad-ce-0.2/results.pkl', 'rb'))
count = 0
all = 0
for r in results:
	if r['question_id'] in open_qids:
		all += 1
		qid = q['question_id']
    pred = r['pred_nodes']
    pred = [x[-1] for x in pred]
    pred_id = [concept2id[x] for x in pred]
    if len(set(r['verif_concepts']) & set(pred_id)) > 0:
      count += 1
```

### Close Graph with pre-trained answer candidates

```
import pickle 
train_results = pickle.load(open('/home/jingrugan/KBVQA/MAVEX/save/bert_base_6layer_6conect-classification/evaluate_results_train.pkl', 'rb'))
val_results = pickle.load(open('/home/jingrugan/KBVQA/MAVEX/save/bert_base_6layer_6conect-classification/evaluate_results_val.pkl', 'rb'))
concept2label = pickle.load(open('./OK-VQA/myAnnotations/trainval_concept2label_729.pkl', "rb"))
label2conceptid = list(concept2label.keys())
qid2clscandidates = {x['question_id']: [label2conceptid[y] for y in x['pred_nodes']] for x in val_results}
qid2clscandidates.update({x['question_id']: [label2conceptid[y] for y in x['pred_nodes']] for x in train_results})
json.dump(qid2clscandidates, open('./OK-VQA/myAnnotations/new_val_split_8501/closeset/qid2clscandidates.json', 'w'))
```

### Check PICa Resutls

```
prompts = ['coco14_promptC_gpt3_VinVLTag_n16_repeat5_47.934998_CLIPImagequestion.json', 'coco14_promptC_gpt3_VinVL_n16_repeat5_46.809354_CLIPImagequestion.json', 'coco14_promptC_gpt3_5GT_n16_repeat5_53.246136_CLIPImagequestion.json']

import pickle 
label2ans = pickle.load(open('data/okvqa_new/cache/trainval_label2ans_729.pkl','rb'))
ans2label = pickle.load(open('data/okvqa_new/cache/trainval_ans2label_729.pkl','rb'))
targets = pickle.load(open('data/okvqa_new/cache/val_target_729.pkl', 'rb'))
targets = sorted(targets, key=lambda x:x['question_id'])
qid2targets = {a['question_id']:[a,{label2ans[x]: s for x,s in zip(b['labels'], b['scores'])}] for a,b in zip(questions, targets)}

predictions = json.load(open('/home/jingrugan/KBVQA/PICa/output_saved/prompt_answer/'+prompts[1]))
predictions = {int(k.split('<->')[1]): v[0] for k,v in predictions.items()}


concept_targets = pickle.load(open('./OK-VQA/myAnnotations/new_val_split_8505/val_target_729_in_ConceptNet.pkl','rb'))
concept_qids = [x['question_id'] for x in concept_targets]

new_data_split_dict = json.load(open('./OK-VQA/myAnnotations/new_val_split_8505/new_open_split_qids.json'))
open_test_qids = new_data_split_dict['open_test']
partial_open_test_qids = new_data_split_dict['partial_open_test']
close_test_qids = new_data_split_dict['close_test']

c_all, c_concept = 0, 0
for k,v in predictions.items():
	if v in qid2targets[k][1]:
		c_all += qid2targets[k][1][v]

for k,v in predictions.items():
	if k in partial_open_test_qids:
		if v in qid2targets[k][1]:
			c_concept += qid2targets[k][1][v]

print(c_all/4790, c_concept/len(partial_open_test_qids))
```

5046:   [0.4693222354340025, 0.45869996036464084, 0.5217994451050285]

4790:   [0.4807933194154442, 0.4696450939457158, 0.5342379958246294]

Open-test:  [0.26506024096385544, 0.24096385542168675, 0.3795180722891566]

Partial open-test: [0.3052845528455282, 0.28739837398373963, 0.36788617886178826]

