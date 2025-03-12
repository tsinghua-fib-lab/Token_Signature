import json
import os
import math
import scipy.stats as stats

# model transfer/All_results.json
# dynamic_cot/All_results.json
with open('model transfer/All_results.json', 'r', encoding='utf-8') as f:
    all_results = json.load(f)


def z_test(p1, p2, n1, n2):
    # （pooled proportion）
    p_hat = (p1 * n1 + p2 * n2) / (n1 + n2)
    
    SE = math.sqrt(p_hat * (1 - p_hat) * (1/n1 + 1/n2))
    
    z_score = (p2 - p1) / SE
    
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    
    return z_score, p_value


for element in all_results:
    benchmark = element["Benchmark"]
    data_file_path = f'benchmark/{benchmark}/data.jsonl'
    p1 = element["Direct_Answer_Acc"]
    p2 = element["CoT_Acc"]

    if os.path.exists(data_file_path):
        with open(data_file_path, 'r', encoding='utf-8') as data_file:
            line_count = sum(1 for _ in data_file)
        
        element["Number"] = line_count
        n1 = element["Number"]
        n2 = element["Number"]
        
        z_score, p_value = z_test(p1, p2, n1, n2)
        
        element["z-test"] = z_score
        element["p"] = p_value
        if p_value > 0.05:
            element["Significance"] = "None"
        else:
            if z_score < 0:
                element["Significance"] = "Negative"
            else:
                element["Significance"] = "Positive"
        
    else:
        print(f"Warning: File {data_file_path} does not exist.")
        element["Number"] = 0 
        element["z-test"] = None
        element["p"] = None

with open('model transfer//All_results_z-test.json', 'w', encoding='utf-8') as f:
    json.dump(all_results, f, indent=4)

print("File has been updated successfully.")
