import json


input_file = "outputs/All_results.json"  # 输入文件名
output_file = "outputs/All_results.json"  # 输出文件名

# 从文件读取 JSON 数据
with open(input_file, "r", encoding="utf-8") as file:
    data = json.load(file)

# 为每个元素计算 cot_gain 并添加到数据中
for element in data:
    if "CoT_Acc" in element and "Direct_Answer_Acc" in element:
        element["cot_gain"] = element["CoT_Acc"] - element["Direct_Answer_Acc"]

# 将更新后的数据保存回文件
with open(output_file, "w", encoding="utf-8") as file:
    json.dump(data, file, ensure_ascii=False, indent=4)

print(f"更新后的数据已保存到 {output_file}")
