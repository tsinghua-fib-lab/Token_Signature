#!/bin/bash

# ##################################################################### extract_answer #######################################################################



for model in  "Phi-3.5-mini-instruct" "Llama-3.2-3B-Instruct" "Mistral-7B-Instruct-v0.3"  "Llama-3.1-8B-Instruct" 
do
    for benchmark in  "gsm8k"  "MultiArith" "gpqa" "FOLIO" "ContextHub_abductive" "ContextHub_deductive"   "arc_challenge" "arc_easy" "MuSR" "lsat" "commensenseqa"  "piqa" "siqa" "strategyqa" 
    do   
        for experiment in     'cot' 'standard' 'direct_answer'
        do 
            if [ "$model" = "Mistral-7B-Instruct-v0.3" ]; then
                encode_format="instruct"; max_new_tokens=16; model_type="mistralai"; batch_size=100
            elif [ "$model" = "Llama-3.1-8B-Instruct" ] ||  [ "$model" = "Llama-3.2-3B-Instruct" ]; then
                encode_format="normal"; max_new_tokens=16; model_type="llama"; batch_size=100
            elif [ "$model" = "Phi-3.5-mini-instruct" ]; then
                encode_format="normal"; max_new_tokens=16; model_type="microsoft"; batch_size=100
            else
                encode_format="normal"; max_new_tokens=16; model_type="none"; batch_size=100
            fi

            data_file="benchmark/$benchmark/data.jsonl"
            output_fname="outputs/$benchmark/$model_type/$experiment/$model.jsonl"
            output_name="outputs/$benchmark/$model_type/$experiment/$model-extract-answer-5.jsonl"
            result_path="outputs/All_results.json"

            mkdir -p "$(dirname "$output_name")"
            mkdir -p "$(dirname "$result_path")"

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


