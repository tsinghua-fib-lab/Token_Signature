from collections import Counter
import json
import os
import random
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

# 定义文件路径和相关变量
benchmarks = ["gsm8k","MultiArith","gpqa","FOLIO","ContextHub_abductive","ContextHub_deductive",
              "arc_challenge","arc_easy","MuSR","lsat","commensenseqa","piqa","siqa","strategyqa"]   #
model_type = ["llama","microsoft","mistralai","llama"]
model_all = ["Llama-3.2-3B-Instruct","Phi-3.5-mini-instruct","Mistral-7B-Instruct-v0.3","Llama-3.1-8B-Instruct"]

# 计算准确率
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


#最大化分类准确度
def find_best_threshold(X, y):
    thresholds = np.sort(X)  # 将特征值排序
    best_accuracy = 0
    best_threshold = None

    for t in thresholds:
        # 根据阈值进行分类
        predictions = (X > t).astype(int)
        
        # 计算分类准确率
        accuracy = (predictions == y).mean()
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = t

    return best_threshold, best_accuracy
index=1
benchmark=benchmarks[3]
# # 遍历基准和模型，提取数据
for index in range(len(model_all)):
    for benchmark in benchmarks:
        # 动态设置文件路径
        # if benchmark=="gsm8k":
        cot_path = f"outputs/{benchmark}/{model_type[index]}/cot/{model_all[index]}.jsonl"
        direct_answer_path = f"outputs/{benchmark}/{model_type[index]}/direct_answer/{model_all[index]}.jsonl"


        cot_file_path = f"outputs/{benchmark}/{model_type[index]}/cot/{model_all[index]}-extract-answer-5.jsonl"
        direct_answer_file_path = f"outputs/{benchmark}/{model_type[index]}/direct_answer/{model_all[index]}-extract-answer-5.jsonl"
        standard_file_path = f"outputs/{benchmark}/{model_type[index]}/standard/{model_all[index]}-sc.jsonl"

        pred_output_file_path=f"outputs/{benchmark}/{model_type[index]}/{model_all[index]}-pred-prob.json"

        features = []
        labels = []
        problem_index=[]
        cot_token_use=[]
        da_token_use=[]


        with open(cot_file_path, 'r') as cot_file, \
                open(direct_answer_file_path, 'r') as direct_answer_file, \
                open(standard_file_path, 'r') as standard_file, \
                open(cot_path, 'r') as cot_file_initial, \
                open(direct_answer_path, 'r') as direct_answer_file_initial:

            cot_lines = [json.loads(line) for line in cot_file]
            direct_answer_lines = [json.loads(line) for line in direct_answer_file]
            standard_lines = [json.loads(line) for line in standard_file]
            cot_lines_initial = [json.loads(line) for line in cot_file_initial]
            direct_answer_lines_initial = [json.loads(line) for line in direct_answer_file_initial]

        print(len(standard_lines),len(cot_lines_initial),len(direct_answer_lines_initial))
        # 提取特征和标签
        for i in range(len(standard_lines)):
            spearman_correlation = standard_lines[i]["spearman_correlation"]
            cot_token_use.append(cot_lines_initial[i]['len_probs'])
            da_token_use.append(direct_answer_lines_initial[i]['len_probs'])
            if cot_lines[i]["is_correct"] and direct_answer_lines[i]["is_correct"]:       #both right
                problem_index.append(i)
                features.append([spearman_correlation])
                labels.append(2)
            elif cot_lines[i]["is_correct"] and not direct_answer_lines[i]["is_correct"]:   #CoT right & da error
                features.append([spearman_correlation])
                labels.append(1)
                problem_index.append(i)
            elif not cot_lines[i]["is_correct"] and direct_answer_lines[i]["is_correct"]:    #CoT error & da right
                features.append([spearman_correlation])
                labels.append(0)
                problem_index.append(i)
            elif not cot_lines[i]["is_correct"] and not direct_answer_lines[i]["is_correct"]:   #both error
                features.append([spearman_correlation])
                labels.append(-1)
                problem_index.append(i)


        data = list(zip(problem_index, features, labels,cot_token_use,da_token_use))



        # if index==0 and benchmark=="ContextHub_abductive":
        filtered_data = [item for item in data if item[2] in [0, 1]]

        # 随机选取 50 条数据作为训练集
        random.seed(42) 
        random.shuffle(filtered_data)
        train_data = random.sample(filtered_data, 50)

        # 剩余数据作为测试集
        test_data = [item for item in data if item not in train_data]


        # train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)




        train_problem_index, train_features, train_labels,train_cot_token_use,train_da_token_use = zip(*train_data)
        test_problem_index, test_features, test_labels,test_cot_token_use,test_da_token_use = zip(*test_data)

        train_labels = np.array(train_labels)
        mask = np.isin(train_labels, [0, 1])
        train_problem_index_filtered = np.array(train_problem_index)[mask]
        train_features_filtered = np.array(train_features)[mask]
        train_labels_filtered = train_labels[mask]


        test_problem_index = np.array(test_problem_index)
        test_features = np.array(test_features)
        test_labels = np.array(test_labels)
        test_cot_token_use=np.array(test_cot_token_use)
        test_da_token_use=np.array(test_da_token_use)



        # print(f"Filtered training set size: {len(train_labels_filtered)}")
        # print(len(train_features_filtered),len(train_labels_filtered))
        
        label_distribution = Counter(train_labels_filtered)

        # 检查标签类别数量
        if len(label_distribution) < 2:
            continue
        ############ 逻辑回归模型###################
        logreg = LogisticRegression()         #class_weight='balanced'
        logreg.fit(train_features_filtered, train_labels_filtered)  #class_weight='balanced'


        #############
        # param_grid = {
        #     'C': [0.001, 0.01, 0.1, 1, 10, 100],  # 正则化强度
        #     'penalty': ['l2'],  # 惩罚项 (l1: Lasso, l2: Ridge)
        #     'solver': ['liblinear']  # `liblinear` 支持 L1 和 L2 惩罚项
        # }

        # # 初始化逻辑回归模型
        # logreg = LogisticRegression()

        # # 设置 GridSearchCV 进行超参数搜索
        # grid_search = GridSearchCV(logreg, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

        # # 训练 GridSearchCV
        # grid_search.fit(train_features_filtered, train_labels_filtered)

        # # 输出最优参数
        # print("最佳参数:", grid_search.best_params_)
        # print("最佳交叉验证准确率:", grid_search.best_score_)

        # # 获取最优模型
        # best_logreg = grid_search.best_estimator_
        # logreg=best_logreg
        # 在测试集上评估
        # test_accuracy = best_logreg.score(test_features_filtered, test_labels_filtered)
        # print("测试集准确率:", test_accuracy)
        ##############################3


        # L2 正则化，正则化强度 C = 0.1（强正则化）
        # logreg = LogisticRegression(penalty='l2', C=0.1)
        # logreg.fit(train_features_filtered, train_labels_filtered)
        # L1 正则化，需要使用 'liblinear' 或 'saga' solver
        # logreg = LogisticRegression(penalty='l1', solver='liblinear', C=0.1)
        # logreg.fit(train_features_filtered, train_labels_filtered)
        

        y_pred_prob_train = logreg.predict_proba(train_features_filtered)[:, 1]

        threshold = 0.5
        y_pred_train = (y_pred_prob_train >= threshold).astype(int)

        # 训练集准确率
        accuracy = accuracy_score(train_labels_filtered, y_pred_train)
        print(benchmark,model_all[index])
        # print(f"训练集预测准确率: {accuracy:.2f}")
        # print(train_labels_filtered, y_pred_train)
        

        # 预测集准确率 test_features, test_labels
        y_pred_prob = logreg.predict_proba(test_features)[:, 1]
        threshold = 0.5
        y_pred = (y_pred_prob >= threshold).astype(int)

        logit_test_accuracy,logit_pred_token_use = custom_accuracy(test_labels, y_pred,test_cot_token_use,test_da_token_use)
        print(len(test_problem_index),len(y_pred_prob))
        data = [{'test_problem_index': int(a1), 'y_pred_prob': float(a2),'y_pred': 1 if float(a2) > 0.5 else 0}  for a1, a2 in zip(test_problem_index, y_pred_prob)]
        with open(pred_output_file_path, 'w') as file:
            json.dump(data, file, indent=4)

        # 获取权重和截距
        w = logreg.coef_[0][0]  # 逻辑回归的权重
        b = logreg.intercept_[0]  # 截距

        # 计算原始特征对应的阈值
        x_threshold = -b / w
        # 限制阈值在特征范围内
        x_threshold_clipped = np.clip(x_threshold, -1, 1)

        #################最大化分类准确率###########################
        best_threshold, best_accuracy = find_best_threshold(train_features_filtered, train_labels_filtered)
        y_pred = (test_features > best_threshold).astype(int)
        print(benchmark,model_all[index])
        # 计算测试集的准确率
        enumerate_test_accuracy,enumerate_pred_token_use = custom_accuracy(test_labels, y_pred,test_cot_token_use,test_da_token_use)
        # print(f"测试集分类准确率: {accuracy:.3f}")
        print(best_threshold)

        new_elements = {
                "flexible_cot_logit":round(logit_test_accuracy,4),
                "flexible_cot_enumerate":round(enumerate_test_accuracy,4),
                "logit_w":w,
                "logit_threshold":x_threshold,
                "logit_threshold_clipped":x_threshold_clipped,
                "enumerate_threshold":float(best_threshold[0]),
                "logit_pred_token_use":round(logit_pred_token_use,4),
                "enumerate_pred_token_use":round(enumerate_pred_token_use,4)
            }
        file_path='flexible_cot/flexible_cot_result_50_C0_1.json'
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