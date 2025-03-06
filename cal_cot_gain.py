import json


input_file = "outputs/All_results.json" 
output_file = "outputs/All_results.json"  

with open(input_file, "r", encoding="utf-8") as file:
    data = json.load(file)

for element in data:
    if "CoT_Acc" in element and "Direct_Answer_Acc" in element:
        element["cot_gain"] = element["CoT_Acc"] - element["Direct_Answer_Acc"]

with open(output_file, "w", encoding="utf-8") as file:
    json.dump(data, file, ensure_ascii=False, indent=4)

print(f"更新后的数据已保存到 {output_file}")
