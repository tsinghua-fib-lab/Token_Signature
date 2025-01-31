# #!/bin/bash

# # 指定输入和输出文件路径
# input_file="outputs/gsm8k/llama/standard/Llama-3.2-3B-Instruct.jsonl"
# output_file="outputs/gsm8k/llama/standard/Llama-3.2-3B-Instruct-with-extracted-answer.jsonl"

# # 执行 Python 脚本提取答案
# python extract_answer.py "$input_file" "$output_file"


#!/bin/bash



for model in  "Phi-3.5-mini-instruct" #"Llama-3.2-3B-Instruct" "Mistral-7B-Instruct-v0.3"  "Llama-3.1-8B-Instruct"  # #  # 
do
    for benchmark in  "strategyqa" #"gsm8k"  "MultiArith" "gpqa" "FOLIO" "ContextHub_abductive" "ContextHub_deductive"   "arc_challenge" "arc_easy" "MuSR" "lsat" "commensenseqa"  "piqa" "siqa" "strategyqa"  #crows "sst-2" 
    do     #"gpqa" "MuSR" "lsat"
        for experiment in     'cot' #'standard' 'direct_answer'
        do 
            # 根据模型和基准测试设置 encode_format 和 max_new_tokens
            if [ "$model" = "Mistral-7B-Instruct-v0.3" ]; then
                encode_format="instruct"; max_new_tokens=16; model_type="mistralai"; batch_size=100
            elif [ "$model" = "Llama-3.1-8B-Instruct" ] ||  [ "$model" = "Llama-3.2-3B-Instruct" ]; then
                encode_format="normal"; max_new_tokens=16; model_type="llama"; batch_size=100
            elif [ "$model" = "Phi-3.5-mini-instruct" ]; then
                encode_format="normal"; max_new_tokens=16; model_type="microsoft"; batch_size=100
            else
                encode_format="normal"; max_new_tokens=16; model_type="none"; batch_size=100
            fi

            # 定义文件路径
            data_file="benchmark/$benchmark/data.jsonl"
            output_fname="outputs/$benchmark/$model_type/$experiment/$model.jsonl"
            output_name="outputs/$benchmark/$model_type/$experiment/$model-extract-answer-5.jsonl"
            result_path="outputs/All_results.json"

            # 确保输出目录存在
            mkdir -p "$(dirname "$output_name")"
            mkdir -p "$(dirname "$result_path")"

            # 执行 Python 脚本提取答案
            python3 extract_answer.py \
                --model_name_or_path "model/$model_type/$model" \
                --data_file "$data_file" \
                --encode_format "$encode_format" \
                --max_new_tokens "$max_new_tokens" \
                --decoding "$experiment" \
                --output_fname "$output_fname" \
                --batch_size "$batch_size" \
                --result_path "$result_path" \
                --gpu_id 4 \
                --output_name "$output_name" \
                --benchmark $benchmark \
                --experiment $experiment \
                --model $model
        done
    done
done



# for experiment in   'cot' 'direct_answer'  #'standard' 
# do
#     for model in "Phi-3.5-mini-instruct" "Llama-3.2-3B-Instruct"   "Llama-3.1-8B-Instruct"     "Mistral-7B-Instruct-v0.3"   # 
#     do
#         for benchmark in "ContextHub_abductive" "ContextHub_deductive" #"piqa" "siqa" "strategyqa" "FOLIO" "lsat" "arc_challenge" "arc_easy"  "gsm8k"  "MultiArith" "commensenseqa"  "crows" "sst-2"  "FOLIO"  "gpqa" "MuSR" "piqa" "siqa"  # # ContextHub_abductive  ContextHub_deductive
#         do     #
             
#             # 根据模型和基准测试设置 encode_format 和 max_new_tokens
#             if [ "$model" = "Mistral-7B-Instruct-v0.3" ]; then
#                 encode_format="instruct"; max_new_tokens=16; model_type="mistralai"; batch_size=100
#             elif [ "$model" = "Llama-3.1-8B-Instruct" ] ||  [ "$model" = "Llama-3.2-3B-Instruct" ]; then
#                 encode_format="normal"; max_new_tokens=16; model_type="llama"; batch_size=100
#             elif [ "$model" = "Phi-3.5-mini-instruct" ]; then
#                 encode_format="normal"; max_new_tokens=16; model_type="microsoft"; batch_size=100
#             else
#                 encode_format="normal"; max_new_tokens=16; model_type="none"; batch_size=100
#             fi

#             # 定义文件路径
#             data_file="benchmark/$benchmark/data.jsonl"
#             output_fname="outputs/$benchmark/$model_type/$experiment/$model.jsonl"
#             output_name="outputs/$benchmark/$model_type/$experiment/$model-extract-answer-2.jsonl"
#             result_path="outputs/result_direct_answer_1.txt"

#             # 确保输出目录存在
#             mkdir -p "$(dirname "$output_name")"
#             mkdir -p "$(dirname "$result_path")"

#             # 执行 Python 脚本提取答案
#             python3 extract_answer2.py \
#                 --model_name_or_path "model/$model_type/$model" \
#                 --data_file "$data_file" \
#                 --encode_format "$encode_format" \
#                 --max_new_tokens "$max_new_tokens" \
#                 --decoding "$experiment" \
#                 --output_fname "$output_fname" \
#                 --batch_size "$batch_size" \
#                 --result_path "$result_path" \
#                 --gpu_id 4 \
#                 --output_name "$output_name" \
#                 --benchmark $benchmark \
#                 --experiment $experiment \
#                 --model $model
#         done
#     done
# done


# for model in   "Llama-3.2-3B-Instruct" #"Mistral-7B-Instruct-v0.3" "Phi-3.5-mini-instruct" "Llama-3.1-8B-Instruct"   #  # 
# do
#     for benchmark in "gpqa" #"gsm8k"  "MultiArith" "gpqa" "FOLIO" "ContextHub_abductive" "ContextHub_deductive"   "arc_challenge" "arc_easy" "MuSR" "lsat" "commensenseqa"  "piqa" "siqa" "strategyqa" "crows" "sst-2" #    "arc_challenge" "arc_easy"     # # ContextHub_abductive  ContextHub_deductive
#     do     #
#         for experiment in   'cot'   #'standard' 'direct_answer'
#         do 
#             # 根据模型和基准测试设置 encode_format 和 max_new_tokens
#             if [ "$model" = "Mistral-7B-Instruct-v0.3" ]; then
#                 encode_format="instruct"; max_new_tokens=16; model_type="mistralai"; batch_size=100
#             elif [ "$model" = "Llama-3.1-8B-Instruct" ] ||  [ "$model" = "Llama-3.2-3B-Instruct" ]; then
#                 encode_format="normal"; max_new_tokens=16; model_type="llama"; batch_size=100
#             elif [ "$model" = "Phi-3.5-mini-instruct" ]; then
#                 encode_format="normal"; max_new_tokens=16; model_type="microsoft"; batch_size=100
#             else
#                 encode_format="normal"; max_new_tokens=16; model_type="none"; batch_size=100
#             fi

#             # 定义文件路径
#             data_file="benchmark/$benchmark/data.jsonl"
#             output_fname="outputs/$benchmark/$model_type/$experiment/$model.jsonl"
#             output_name="outputs/$benchmark/$model_type/$experiment/$model-extract-answer-4.jsonl"
#             result_path="outputs/result.txt"

#             # 确保输出目录存在
#             mkdir -p "$(dirname "$output_name")"
#             mkdir -p "$(dirname "$result_path")"

#             # 执行 Python 脚本提取答案
#             python3 extract_answer2.py \
#                 --model_name_or_path "model/$model_type/$model" \
#                 --data_file "$data_file" \
#                 --encode_format "$encode_format" \
#                 --max_new_tokens "$max_new_tokens" \
#                 --decoding "$experiment" \
#                 --output_fname "$output_fname" \
#                 --batch_size "$batch_size" \
#                 --result_path "$result_path" \
#                 --gpu_id 4 \
#                 --output_name "$output_name" \
#                 --benchmark $benchmark \
#                 --experiment $experiment \
#                 --model $model
#         done
#     done
# done
