# GATHER: Path-based KB-VQA

Code and data for *"Open-Set Knowledge-Based Visual Question Answering with Inference Paths"*.

## Installation
Using `conda`, create an environment

```
pip install -r requirements.txt
```

## Data Preparation

Download dataset OK-VQA.

Train from scratch or use pre-trained model parameters [here](https://drive.google.com/file/d/1XB8hfolvroTuDaogpu0lk7YDdM_Ivah8/view?usp=sharing).

Download pre-processed ConceptNet-annotated KBVQA data, or generate the schema graphs with script.

- [Generated Schema Graphs](https://drive.google.com/file/d/1sbtfhPEcpn8rOtV0nXXEtXp1lZCgGcss/view?usp=sharing)

- [KBVQA Annotated with ConceptNet Entities](https://drive.google.com/file/d/1AACUy-hLJaFIneHFeeZtoUp32hnhgRwl/view?usp=sharing)

Please refer to `Preprocess.md` for more implementation details.

## Train a new model

```
python ft_gather.py --save_name test_gather --seed 7777 --evaluate 0 --backbone_model bert --use_concept 0 --use_wiki 0 --use_image 0 --use_split_name 8505 --add_answer_emb 0 --ablation ptm path lstm prune node_type --from_pretrained ./save/bert_base_6layer_6conect-step1-128/pytorch_model.bin --vil_dim 128
```

