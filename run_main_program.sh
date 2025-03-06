#!/bin/bash
for model in  "Phi-3.5-mini-instruct"  "Llama-3.2-3B-Instruct"  "Llama-3.1-8B-Instruct" "Mistral-7B-Instruct-v0.3" 
do
    for benchmark in   "gsm8k"  "MultiArith""commensenseqa" "lsat" "crows" "sst-2" "arc_challenge" "arc_easy" "FOLIO"  "gpqa" "MuSR" "piqa" "siqa"  
    do
        for experiment in 'cot' 'direct_answer'  'standard' 
        do
            # Set encode_format and max_new_tokens based on model
            if [ "$model" = "Mistral-7B-Instruct-v0.1" ] || [ "$model" = "Mistral-7B-Instruct-v0.3" ]; then
                encode_format="instruct"; max_new_tokens=512; model_type="mistralai";batch_size=50
                if [ "$experiment" = "direct_answer" ]; then
                    max_new_tokens=32; batch_size=80
                elif [ "$benchmark" = "mmlu" ] || [ "$benchmark" = "lsat" ] || [ "$experiment" = "cot" ]; then
                    max_new_tokens=1024; batch_size=25

                fi
            elif [ "$model" = "Llama-3.1-8B-Instruct" ]; then
                encode_format="normal"; max_new_tokens=512; model_type="llama";batch_size=50
                if [ "$experiment" = "direct_answer" ]; then
                    max_new_tokens=32; batch_size=80
                elif [ "$benchmark" = "mmlu" ] || [ "$benchmark" = "lsat" ] || [ "$experiment" = "cot" ]; then
                    max_new_tokens=1024; batch_size=25
                fi
            elif [ "$model" = "Llama-3.2-3B-Instruct" ]; then
                encode_format="normal"; max_new_tokens=512; model_type="llama";batch_size=50
                if [ "$experiment" = "direct_answer" ]; then
                    max_new_tokens=32; batch_size=80
                elif [ "$benchmark" = "mmlu" ] || [ "$benchmark" = "lsat" ] || [ "$experiment" = "cot" ]; then
                    max_new_tokens=1024; batch_size=25
                fi            
            elif [ "$model" = "Phi-3.5-mini-instruct" ]; then
                encode_format="normal"; max_new_tokens=512; model_type="microsoft";batch_size=50
                if [ "$experiment" = "direct_answer" ]; then
                    max_new_tokens=16; batch_size=80
                elif [ "$benchmark" = "mmlu" ] || [ "$benchmark" = "lsat" ] || [ "$experiment" = "cot" ]; then
                    max_new_tokens=1024; batch_size=25
                fi
            else
                encode_format="normal"; max_new_tokens=512; model_type="none"
            fi

            mkdir -p ./outputs/$benchmark/$model_type
            mkdir -p ./outputs/$benchmark/$model_type/$experiment


            # Run the Python script with the dynamically set parameters
            python main.py \
                --model_name_or_path model/$model_type/$model \
                --data_file benchmark/$benchmark/data.jsonl \
                --encode_format $encode_format \
                --max_new_tokens $max_new_tokens \
                --decoding $experiment \
                --output_fname outputs/$benchmark/$model_type/$experiment/$model.jsonl \
                --batch_size $batch_size \
                --result_path outputs/$benchmark/$model_type/$experiment/result.txt \
                --gpu_id 7 \
                --model $model
        done
    done
done