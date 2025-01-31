import json
import argparse
from scipy.stats import spearmanr

def compute_spearman_instance(input_file, output_file, result_path,model,benchmark):
    # 存储所有Spearman相关系数
    spearman_correlations = []
    top50_tokens=0
    Question_num=0
    # 打开输入和输出文件
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        
        for line_num, line in enumerate(fin, 1):
            try:
                data = json.loads(line)
                token_probs = data.get('token_probs', [])
                len_probs = data.get('len_probs', len(token_probs))
                
                Question_num+=1
                # 根据len_probs决定取多少token_probs
                if len_probs >= 50:
                    tokens = token_probs[:50]
                    positions = list(range(1, 51))  # 位置索引从1到50
                    top50_tokens+=50
                else:
                    tokens = token_probs
                    positions = list(range(1, len_probs + 1))  # 位置索引从1到len_probs
                    top50_tokens+=len_probs
                
                # 确保有足够的数据进行相关性计算
                if len(tokens) >= 2:
                    corr, _ = spearmanr(tokens, positions)
                else:
                    corr = None  # 不足以计算相关性
                
                # 添加到相关系数列表
                spearman_correlations.append(corr)
                
                # 将结果写入输出文件
                output_data = {
                    'spearman_correlation': corr
                }
                fout.write(json.dumps(output_data, ensure_ascii=False) + '\n')
            
            except json.JSONDecodeError as e:
                print(f"第 {line_num} 行 JSON 解码错误: {e}")
                spearman_correlations.append(None)
            except Exception as e:
                print(f"第 {line_num} 行处理错误: {e}")
                spearman_correlations.append(None)
    
    # 计算Spearman相关系数的均值，忽略None值
    valid_correlations = [c for c in spearman_correlations if c is not None]
    if valid_correlations:
        mean_corr = sum(valid_correlations) / len(valid_correlations)
        mean_tokens=top50_tokens/Question_num
    else:
        mean_corr = None

    file_path = result_path

    new_elements = {
        "Instance_SC":round(mean_corr,4)
    }
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)


    # 遍历数据，找到目标并更新
    found = False
    for item in data:
        if item.get("Model") == model and item.get("Benchmark") == benchmark:
            item.update(new_elements)
            found = True
            break

    # 如果没有找到目标条目，添加新的记录
    if not found:
        data.append({
            "Model": model,
            "Benchmark": benchmark,
            **new_elements
        })

    # 将更新后的数据写回文件
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

    # # 将均值写入result_path
    # with open(result_path, 'a', encoding='utf-8') as f:
    #     if mean_corr is not None:
    #         f.write(f"Model: {model}\nBenchmark: {benchmark}\n")
    #         f.write(f"Mean Spearman Correlation: {mean_corr}\n")
    #         f.write(f"Mean tokens: {mean_tokens}\n")
    #     else:
    #         f.write("No valid Spearman correlations were calculated.\n")
    
    # print(f"Spearman相关系数已保存到 {output_file}")
    # print(f"平均Spearman相关系数已保存到 {result_path}")

def main():
    parser = argparse.ArgumentParser(description="计算Spearman相关系数并保存结果。")
    parser.add_argument('--input_file', type=str, required=True, help='输入的JSONL文件路径。')
    parser.add_argument('--output_file', type=str, required=True, help='输出的JSONL文件路径。')
    parser.add_argument('--result_path', type=str, required=True, help='保存平均Spearman相关系数的文件路径。')
    parser.add_argument('--model', type=str, required=True, help='model')
    parser.add_argument('--benchmark', type=str, required=True, help='benchmark')

    args = parser.parse_args()
    
    compute_spearman_instance(args.input_file, args.output_file, args.result_path,args.model,args.benchmark)

if __name__ == "__main__":
    main()
