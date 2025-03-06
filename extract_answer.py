import argparse
import json
import os
import re
import sys
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from task1 import AnswerTask, ChoiceTask  # 确保 task.py 在同一目录下
import setproctitle

setproctitle.setproctitle('')

os.environ["CUDA_VISIBLE_DEVICES"] = "4" 

def main():
    parser = argparse.ArgumentParser(description="Extract answers from model outputs.")
    parser.add_argument('--model_name_or_path', type=str, required=True, help='Path to the model.')
    parser.add_argument('--data_file', type=str, required=True, help='Path to the data file (JSONL).')
    parser.add_argument('--encode_format', type=str, choices=['instruct', 'normal'], required=True, help='Encoding format.')
    parser.add_argument('--max_new_tokens', type=int, default=512, help='Maximum number of new tokens.')
    parser.add_argument('--decoding', type=str, choices=['standard','direct_answer','cot'], required=True, help='Decoding method.')
    parser.add_argument('--output_fname', type=str, required=True, help='Path to the model output file (JSONL).')
    parser.add_argument('--batch_size', type=int, default=40, help='Batch size.')
    parser.add_argument('--result_path', type=str, required=True, help='Path to save the result (accuracy).')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use.')
    parser.add_argument('--output_name', type=str, required=True, help='Path to save the extracted answers (JSONL).')
    parser.add_argument('--benchmark', type=str, required=True, help='Benchmark')
    parser.add_argument('--experiment', type=str, required=True, help='Experiment')
    parser.add_argument('--model', type=str, required=True, help='Model')

    args = parser.parse_args()
    print(args.output_name)
    if os.path.exists(args.output_name):
        print(f"{args.output_name} exist!!!!!!!!!!!!!!!!!!!!!!")
        sys.exit(0)

    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

    benchmark = args.benchmark
    if benchmark in ["gsm8k", "MultiArith", "MATH"]:
        task = AnswerTask(encode_format=args.encode_format, decoding=args.decoding, data_file=args.data_file,model=args.model)
        use_answer_task = True

    else:
        task = ChoiceTask(encode_format=args.encode_format, decoding=args.decoding,model=args.model)
        use_answer_task = False
        if args.experiment == "cot":
            sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_new_tokens)
            llm = LLM(model=args.model_name_or_path)
            # llm.to(device)


    total = 0
    correct = 0

    os.makedirs(os.path.dirname(args.output_name), exist_ok=True)
    os.makedirs(os.path.dirname(args.result_path), exist_ok=True)

    results = []

    with open(args.data_file, 'r', encoding='utf-8') as data_f, \
         open(args.output_fname, 'r', encoding='utf-8') as output_f:

        data_lines = data_f.readlines()
        output_lines = output_f.readlines()

        if len(data_lines) != len(output_lines):
            print("警告: data_file 和 output_fname 的行数不匹配！")
        
        for data_line, output_line in tqdm(zip(data_lines, output_lines), total=min(len(data_lines), len(output_lines)), desc="Processing"):
            data_example = json.loads(data_line)
            output_example = json.loads(output_line)

            # 获取 prompt 和 generated_text
            prompt = data_example.get("prompt") or data_example.get("question")  
            generated_text = output_example.get("generated_text")

            if not prompt or not generated_text:
                extracted_answer = "[invalid]"
            else:
                combined_text_prompt = prompt + generated_text #+ "\nBased on the above information, tell me the answer choice. So the best answer letter choice is "   #"So the best answer letter choice is "
                if args.model=="Phi-3.5-mini-instruct":
                    combined_text = "<|user|>"+combined_text_prompt+ "<|end|><|assistant|>So the best answer letter choice is"   #<your answer letter choice>
                elif args.model=="Llama-3.2-3B-Instruct" or  args.model=="Llama-3.1-8B-Instruct": 
                    combined_text = '<|begin_of_text|><|start_header_id|>user<|end_header_id|>'+combined_text_prompt+'<|eot_id|><|start_header_id|>assistant<|end_header_id|>So the best answer letter choice is'
                elif args.model=="Mistral-7B-Instruct-v0.3":
                    combined_text = '[INST]'+combined_text_prompt+'[/INST]So the best answer letter choice is'
                    
                if use_answer_task:
                    extracted_answer, _ = task.extract_model_answer(generated_text)
                    if extracted_answer!='[invalid]':
                        try:
                            extracted_answer=float(extracted_answer.replace(",", ""))
                            if extracted_answer.is_integer():
                                extracted_answer = str(int(extracted_answer))  
                            else:
                                extracted_answer = str(extracted_answer)
                        except:
                            extracted_answer = str(extracted_answer)
                else:
                    if args.experiment == "cot":
                        outputs = llm.generate(combined_text, sampling_params)
                        print(outputs[0].outputs[0].text)
                        completion = outputs[0].outputs[0].text
                        second_part = completion
                        if benchmark == "piqa" or benchmark == "strategyqa" or benchmark == "sst-2" :
                            match = re.search(r"[A-B]", second_part.strip())
                        elif benchmark=="siqa" or benchmark=="crows" or benchmark=="bbh" or benchmark=="FOLIO":
                            match = re.search(r"[A-C]", second_part.strip())
                        elif benchmark=="gpqa" or benchmark=="mmlu" or benchmark=="arc_easy" or benchmark=="arc_challenge":
                            match = re.search(r"[A-D]", second_part.strip())
                        else:                        #"commensenseqa"  "lsat" "MuSR"
                            match = re.search(r"[A-E]", second_part.strip())

                        words = second_part.strip().split()

                        # if words[0].lower() == "not" and benchmark=="gpqa":
                        #     extracted_answer="[invalid]"

                        if match:
                            extracted_answer = match.group(0)
                            if (words[0].lower() == "not" or  words[0].lower() == "none") and benchmark=="gpqa":
                                 extracted_answer="[invalid]"
                        else:
                            match_num = re.search(r'\d', second_part) 
                            if match_num:
                                number = int(match_num.group(0))
                                if 1 <= number <= 5:
                                    extracted_answer = chr(64 + number)  
                                else:
                                    extracted_answer = "[invalid]"
                            else:
                                extracted_answer = "[invalid]"
                            # extracted_answer = "[invalid]"
                        

                    elif args.experiment == "direct_answer":
                        extracted_answer, _ = task.extract_model_answer(generated_text)

            correct_answer = data_example.get("answer", "[invalid]")

            is_correct = (extracted_answer == correct_answer)
            if is_correct:
                correct += 1
            total += 1
            
            if use_answer_task:
                filtered_output_example = {
                    "extracted_answer": extracted_answer,
                    "correct_answer": correct_answer,
                    "is_correct": is_correct
                }
            else:
                if args.experiment == "cot":
                    filtered_output_example = {
                        "so_the_answer": second_part,
                        "extracted_answer": extracted_answer,
                        "correct_answer": correct_answer,
                        "is_correct": is_correct
                    }
                else:
                    filtered_output_example = {
                        "extracted_answer": extracted_answer,
                        "correct_answer": correct_answer,
                        "is_correct": is_correct
                    }
            
            results.append(filtered_output_example)

    accuracy = correct / total if total > 0 else 0.0

    with open(args.output_name, 'w', encoding='utf-8') as out_f:
        for result in results:
            json.dump(result, out_f, ensure_ascii=False)
            out_f.write('\n')

    file_path = args.result_path
    if args.experiment=='direct_answer':
        new_elements = {
            "Direct_Answer_Acc":round(accuracy,4)
        }
    elif args.experiment=='cot':
        new_elements = {
            "CoT_Acc":round(accuracy,4)
        }

    target_model = args.model
    target_benchmark = args.benchmark

    if not os.path.exists(file_path):
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump([], file, ensure_ascii=False, indent=4)

    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("JSON 文件内容不是列表，请检查文件格式！")

    found = False
    for item in data:
        if item.get("Model") == target_model and item.get("Benchmark") == target_benchmark:
            item.update(new_elements)
            found = True
            break

    if not found:
        data.append({
            "Model": target_model,
            "Benchmark": target_benchmark,
            **new_elements
        })

    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
