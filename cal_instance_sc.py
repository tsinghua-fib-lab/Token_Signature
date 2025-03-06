import json
import argparse
from scipy.stats import spearmanr

def compute_spearman_instance(input_file, output_file, result_path,model,benchmark):
    spearman_correlations = []
    top50_tokens=0
    Question_num=0

    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        
        for line_num, line in enumerate(fin, 1):
            try:
                data = json.loads(line)
                token_probs = data.get('token_probs', [])
                len_probs = data.get('len_probs', len(token_probs))
                
                Question_num+=1
                if len_probs >= 50:
                    tokens = token_probs[:50]
                    positions = list(range(1, 51))  
                    top50_tokens+=50
                else:
                    tokens = token_probs
                    positions = list(range(1, len_probs + 1))  
                    top50_tokens+=len_probs

                if len(tokens) >= 2:
                    corr, _ = spearmanr(tokens, positions)
                else:
                    corr = None 
                
                spearman_correlations.append(corr)

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
