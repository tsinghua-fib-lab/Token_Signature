import json

# 假设你的 jsonl 文件名为 'data.jsonl'
input_file = 'benchmark/gsm8k/test.jsonl'
output_file = 'benchmark/gsm8k/test_answeKey.jsonl'

# 打开输入和输出文件
with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    for line in infile:
        # 解析每一行 JSON
        data = json.loads(line.strip())
        
        # 提取 'answer' 字段中的 "####" 后的数字
        answer = data.get('answer', '')
        if '####' in answer:
            answer_key = answer.split('####')[-1].strip()
            # 在字典中添加新的键 'answerKey'
            data['answerKey'] = answer_key
        
        # 将修改后的字典写入输出文件
        json.dump(data, outfile, ensure_ascii=False)
        outfile.write('\n')

print(f"提取后的数据已保存到 {output_file}")
