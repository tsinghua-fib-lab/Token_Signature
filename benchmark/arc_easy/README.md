---
dataset_info:
  features:
  - name: id
    dtype: string
  - name: question
    dtype: string
  - name: choices
    struct:
    - name: label
      sequence: string
    - name: text
      sequence: string
  - name: answerKey
    dtype: string
  splits:
  - name: test
    num_bytes: 657514
    num_examples: 2376
  - name: train
    num_bytes: 619000
    num_examples: 2251
  - name: validation
    num_bytes: 157394
    num_examples: 570
  download_size: 763157
  dataset_size: 1433908
---
# Dataset Card for "arc_easy"

[More Information needed](https://github.com/huggingface/datasets/blob/main/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)