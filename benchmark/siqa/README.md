---
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: validation
    path: data/validation-*
dataset_info:
  features:
  - name: context
    dtype: string
  - name: question
    dtype: string
  - name: answerA
    dtype: string
  - name: answerB
    dtype: string
  - name: answerC
    dtype: string
  - name: label
    dtype: string
  splits:
  - name: train
    num_bytes: 6327209
    num_examples: 33410
  - name: validation
    num_bytes: 372815
    num_examples: 1954
  download_size: 3678635
  dataset_size: 6700024
---
# Dataset Card for "siqa"

[More Information needed](https://github.com/huggingface/datasets/blob/main/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)