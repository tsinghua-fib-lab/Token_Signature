# Token Signature: Predicting Chain-of-Thought Gains with Token Decoding Feature in Large Language Models

This repository contains the core implementation of our ICML 2025 paper:  
**"Token Signature: Predicting Chain-of-Thought Gains with Token Decoding Feature in Large Language Models."**

## 🧠 Overview

Our work introduces a novel method to **predict Chain-of-Thought (CoT) reasoning gains** using token-level decoding features from large language models (LLMs). This repository includes all code for inference, answer extraction, and evaluation used in the paper.

<img width="1686" height="570" alt="image" src="https://github.com/user-attachments/assets/278a5465-dd8e-43f1-9c5b-fe7536acb3a1" />

<img width="1632" height="682" alt="image" src="https://github.com/user-attachments/assets/a20f02a1-ce67-4f30-b85c-81b54510e340" />


## 📂 File Structure

### 🔍 Core Inference

- `main.py`, `solve.py`, `task1.py`:  
  Main scripts to run inference using LLMs.

- `extract_answer.py`:  
  Extracts answers from model outputs via `vllm` and character-level matching.

### 📊 Evaluation Scripts

- `cal_aggregated_sc.py`: Compute aggregated score.
- `cal_instance_sc.py`: Compute per-instance score.
- `cal_token_use.py`: Calculate token consumption.
- `cal_cot_gain.py`: Compute Chain-of-Thought (CoT) gain.

### 🚀 Execution Scripts

- `run_main_program.sh`: Run full inference pipeline.
- `run_extract.sh`: Extract answers from model output.
- `run_cal.sh`: Run evaluation scripts to compute scores and CoT gain.

## 📁 Directory Overview

- `benchmark/`:  
  Contains question-answer pairs for various benchmarks.

- `dynamic_cot/`:  
  Key implementation of dynamic Chain-of-Thought prompting.

- `model transfer/`:  
  Core code for model transfer experiments.

## 📄 Citation

If you find this code useful for your research, please consider citing our paper:

```bibtex
@article{liu2025token,
  title={Token Signature: Predicting Chain-of-Thought Gains with Token Decoding Feature in Large Language Models},
  author={Liu, Peijie and Xu, Fengli and Li, Yong},
  journal={arXiv preprint arXiv:2506.06008},
  year={2025}
}
