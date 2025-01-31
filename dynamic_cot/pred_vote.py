import json
import os
from matplotlib import pyplot as plt
import numpy as np

from collections import defaultdict
# 定义文件路径和相关变量
benchmarks = ["gsm8k","MultiArith","gpqa","FOLIO","ContextHub_abductive","ContextHub_deductive",
              "arc_challenge","arc_easy","MuSR","lsat","commensenseqa","piqa","siqa","strategyqa"]   #
model_type = ["llama","microsoft","mistralai","llama"]
model_all = ["Llama-3.2-3B-Instruct","Phi-3.5-mini-instruct","Mistral-7B-Instruct-v0.3","Llama-3.1-8B-Instruct"]

index=1
benchmark=benchmarks[3]
# # 遍历基准和模型，提取数据
for benchmark in benchmarks:
    result = defaultdict(lambda: {'Index_number': 0,'sum_pred': 0.0, 'mean_pred': 0.0,'vote_pred':0})
    output_path=f"outputs/{benchmark}/vote_pred.json"
    for index in range(len(model_all)):
    
        # 动态设置文件路径
        # if benchmark=="gsm8k":
        pred_output_file_path=f"outputs/{benchmark}/{model_type[index]}/{model_all[index]}-pred-prob.json"
        with open(pred_output_file_path, 'r') as file:
            data = json.load(file)
        for item in data:
            if 'test_problem_index' in item and 'y_pred_prob' in item:
                key = item['test_problem_index']  # 以 'Array1' 的值作为键
                result[key]['Index_number'] += 1  # 统计出现次数
                result[key]['sum_pred'] += item['y_pred_prob']
        
    for key in result:
        result[key]['mean_pred'] = result[key]['sum_pred']/result[key]['Index_number']
        if result[key]['mean_pred']>0.5:
            result[key]['vote_pred']=1
        else:
            result[key]['vote_pred']=0

    with open(output_path, 'w') as output_file:
        json.dump(result, output_file, indent=4)