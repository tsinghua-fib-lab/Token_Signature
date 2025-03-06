from collections import Counter
import json
import os
import random
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

benchmarks = ["gsm8k","MultiArith","gpqa","FOLIO","ContextHub_abductive","ContextHub_deductive",
              "arc_challenge","arc_easy","MuSR","lsat","commensenseqa","piqa","siqa","strategyqa"]   #
# model_type = ["llama","microsoft","mistralai","llama"]
model_all = ["gpt-4o-mini","gpt-4o"]


def custom_accuracy(test_labels, y_pred,test_cot_token_use,test_da_token_use):
    test_labels = np.array(test_labels)
    y_pred = np.array(y_pred)
    correct_predictions = 0
    token_use_all=0
    for true_label, pred_label,test_cot_token,test_da_token in zip(test_labels, y_pred,test_cot_token_use,test_da_token_use):
        if pred_label == 1:
            token_use_all+=test_cot_token
        elif pred_label == 0:
            token_use_all+=test_da_token

        if true_label == -1:
            continue
        elif true_label == 0 and pred_label == 0:
            correct_predictions += 1
        elif true_label == 1 and pred_label == 1:
            correct_predictions += 1
        elif true_label == 2:
            correct_predictions += 1
    accuracy = correct_predictions / len(test_labels)
    pred_token_use=token_use_all / len(test_labels)
    return accuracy,pred_token_use


def find_best_threshold(X, y):
    thresholds = np.sort(X)  # 将特征值排序
    best_accuracy = 0
    best_threshold = None

    for t in thresholds:
        predictions = (X > t).astype(int)
        accuracy = (predictions == y).mean()
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = t

    return best_threshold, best_accuracy
index=1
benchmark=benchmarks[3]

for index in range(len(model_all)):
    for benchmark in benchmarks:
        # if benchmark=="gsm8k":
        cot_path = f"model_transfer/outputs_openai/{benchmark}/{model_all[index]}/cot/{model_all[index]}.jsonl"
        direct_answer_path = f"model_transfer/outputs_openai/{benchmark}/{model_all[index]}/direct_answer/{model_all[index]}.jsonl"


        cot_file_path = f"model_transfer/outputs_openai/{benchmark}/{model_all[index]}/cot/{model_all[index]}-extract-answer-5.jsonl"
        direct_answer_file_path = f"model_transfer/outputs_openai/{benchmark}/{model_all[index]}/direct_answer/{model_all[index]}-extract-answer-5.jsonl"
        # standard_file_path = f"outputs_openai/{benchmark}/{model_all[index]}/standard/{model_all[index]}-sc.jsonl"

        # vote_file_path = f'outputs/{benchmark}/vote_pred.json'
        vote_file_path = f'outputs/{benchmark}/vote_pred.json'

        labels = []
        problem_index=[]
        cot_token_use=[]
        da_token_use=[]


        with open(cot_file_path, 'r') as cot_file, \
                open(direct_answer_file_path, 'r') as direct_answer_file, \
                open(cot_path, 'r') as cot_file_initial, \
                open(direct_answer_path, 'r') as direct_answer_file_initial:

            cot_lines = [json.loads(line) for line in cot_file]
            direct_answer_lines = [json.loads(line) for line in direct_answer_file]
            cot_lines_initial = [json.loads(line) for line in cot_file_initial]
            direct_answer_lines_initial = [json.loads(line) for line in direct_answer_file_initial]

        print(len(cot_lines_initial),len(direct_answer_lines_initial))

        for i in range(len(cot_lines_initial)):
            cot_token_use.append(cot_lines_initial[i]['len_token'])
            da_token_use.append(direct_answer_lines_initial[i]['len_token'])
            if cot_lines[i]["is_correct"] and direct_answer_lines[i]["is_correct"]:       #both right
                problem_index.append(i)
                labels.append(2)
            elif cot_lines[i]["is_correct"] and not direct_answer_lines[i]["is_correct"]:   #CoT right & da error

                labels.append(1)
                problem_index.append(i)
            elif not cot_lines[i]["is_correct"] and direct_answer_lines[i]["is_correct"]:    #CoT error & da right

                labels.append(0)
                problem_index.append(i)
            elif not cot_lines[i]["is_correct"] and not direct_answer_lines[i]["is_correct"]:   #both error
                labels.append(-1)
                problem_index.append(i)


        data = list(zip(problem_index, labels,cot_token_use,da_token_use))
        test_data =data



        test_problem_index, test_labels,test_cot_token_use,test_da_token_use = zip(*test_data)

        test_problem_index = np.array(test_problem_index)
        test_labels = np.array(test_labels)
        test_cot_token_use=np.array(test_cot_token_use)
        test_da_token_use=np.array(test_da_token_use)

        # vote_file_path = '../outputs/gsm8k/vote_pred.json'

        with open(vote_file_path, 'r') as file:
            data = json.load(file)

        y_pred = [item["vote_pred"] for item in data.values()]

        # if benchmark=='gsm8k':
        #     print(y_pred)
        logit_test_accuracy,logit_pred_token_use = custom_accuracy(test_labels, y_pred,test_cot_token_use,test_da_token_use)



        new_elements = {
                "flexible_cot_transfer":round(logit_test_accuracy,4),
                "transfer_pred_token_use":round(logit_pred_token_use,4)
            }
        file_path='model_transfer/flexible_cot_transfer_result_prob_mean.json'
        if not os.path.exists(file_path):
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump([], file, ensure_ascii=False, indent=4)
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        found = False
        for item in data:
            if item.get("Model") == model_all[index] and item.get("Benchmark") == benchmark:
                item.update(new_elements)
                found = True
                break

        if not found:
            data.append({
                "Model": model_all[index],
                "Benchmark": benchmark,
                **new_elements
            })

        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4) 