import json
import argparse

def compute_token_use(input_file,  result_path,model,benchmark):
    total_len_probs = 0
    num_rows = 0
    with open(input_file, 'r', encoding='utf-8') as fin:
        
        for line_num, line in enumerate(fin, 1):
            try:
                data = json.loads(line)
                total_len_probs += data.get("len_probs", 0)
                num_rows += 1
  
            except json.JSONDecodeError as e:
                print(f"第 {line_num} 行 JSON 解码错误: {e}")

            except Exception as e:
                print(f"第 {line_num} 行处理错误: {e}")

    if num_rows > 0:
        avg_len_probs = total_len_probs / num_rows
    else:
        avg_len_probs = 0

    file_path = result_path
     
    if 'cot' in input_file:
        new_elements = {
            "CoT_mean_tokens":round(avg_len_probs,2)
        }
    elif 'direct' in input_file:
        new_elements = {
            "DA_mean_tokens":round(avg_len_probs,2)
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
    parser.add_argument('--result_path', type=str, required=True, help='保存平均Spearman相关系数的文件路径。')
    parser.add_argument('--model', type=str, required=True, help='model')
    parser.add_argument('--benchmark', type=str, required=True, help='benchmark')

    args = parser.parse_args()
    
    compute_token_use(args.input_file, args.result_path,args.model,args.benchmark)

if __name__ == "__main__":
    main()
