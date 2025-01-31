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
    num_bytes: 375511
    num_examples: 1172
  - name: train
    num_bytes: 349760
    num_examples: 1119
  - name: validation
    num_bytes: 96660
    num_examples: 299
  download_size: 449682
  dataset_size: 821931
---
# Dataset Card for "arc_challenge"

[More Information needed](https://github.com/huggingface/datasets/blob/main/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)