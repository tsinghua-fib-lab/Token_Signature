---
configs:
- config_name: default
  data_files:
  - split: murder_mysteries
    path: murder_mystery.csv
  - split: object_placements
    path: object_placements.csv
  - split: team_allocation
    path: team_allocation.csv
license: cc-by-4.0
task_categories:
- question-answering
language:
- en
tags:
- reasoning
- commonsense
pretty_name: MuSR
size_categories:
- n<1K
---

# MuSR: Testing the Limits of Chain-of-thought with Multistep Soft Reasoning

### Creating murder mysteries that require multi-step reasoning with commonsense using ChatGPT!
By: Zayne Sprague, Xi Ye, Kaj Bostrom, Swarat Chaudhuri, and Greg Durrett.

View the dataset on our custom viewer and [project website](https://zayne-sprague.github.io/MuSR/)!

Check out the [paper](https://arxiv.org/abs/2310.16049). Appeared at ICLR 2024 as a spotlight presentation!

Git Repo with the source data, how to recreate the dataset (and create new ones!) [here](https://github.com/Zayne-sprague/MuSR)