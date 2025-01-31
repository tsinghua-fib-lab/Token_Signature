import argparse
import json
from collections import defaultdict
from scipy.stats import spearmanr

def compute_mean_token_probs(input_file):
    token_probs_sum = defaultdict(float)
    token_probs_count = defaultdict(int)
    max_length = 0

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            token_probs = data.get("token_probs", [])
            for i, prob in enumerate(token_probs):
                token_probs_sum[i] += prob
                token_probs_count[i] += 1
            if len(token_probs) > max_length:
                max_length = len(token_probs)

    mean_token_probs = []
    for i in range(max_length):
        if token_probs_count[i] > 0:
            mean = token_probs_sum[i] / token_probs_count[i]
            mean_token_probs.append(mean)
        else:
            # 如果某个位置没有数据，可以选择跳过或填充为0
            mean_token_probs.append(0.0)
    
    return mean_token_probs

def compute_spearman_correlation(mean_probs, top_n=50):
    if len(mean_probs) < top_n:
        top_n = len(mean_probs)
    subset = mean_probs[:top_n]
    indices = list(range(top_n))
    correlation, _ = spearmanr(indices, subset)
    return correlation

def save_results(output_file, mean_token_probs, spearman_corr, model,benchmark):
    file_path = output_file

    new_elements = {
        "Aggregated_SC":round(spearman_corr,4)
    }
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    found = False
    for item in data:
        if item.get("Model") == model and item.get("Benchmark") == benchmark:
            item.update(new_elements)
            found = True
            break

    if not found:
        data.append({
            "Model": model,
            "Benchmark": benchmark,
            **new_elements
        })

    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def main():
    parser = argparse.ArgumentParser(description="mean-SC-MAI")
    parser.add_argument('--input_file', type=str, required=True, help='输入的JSONL文件路径。')
    parser.add_argument('--output_file', type=str, required=True, help='输出的JSONL文件路径。')
    parser.add_argument('--model', type=str, required=True, help='model')
    parser.add_argument('--benchmark', type=str, required=True, help='benchmark')
    # parser.add_argument('--experiment', type=str, required=True, help='experiment')


    args = parser.parse_args()
    input_fname = args.input_file
    output_fname = args.output_file
    model=args.model
    benchmark=args.benchmark

    # 计算 mean_token_probs
    mean_token_probs = compute_mean_token_probs(input_fname)

    # 计算 Spearman 相关性
    spearman_corr = compute_spearman_correlation(mean_token_probs, top_n=50)

    save_results(output_fname, mean_token_probs, spearman_corr,model,benchmark)

    print("处理完成！")

if __name__ == "__main__":
    main()


