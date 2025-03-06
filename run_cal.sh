#!/bin/bash
##################################################################### instance_sc #######################################################################
for model in    "Llama-3.2-3B-Instruct" "Phi-3.5-mini-instruct" "Mistral-7B-Instruct-v0.3" "Llama-3.1-8B-Instruct"   #    # 
do
    # benchmark
    for benchmark in   "gsm8k" "MultiArith" "gpqa" "FOLIO" "ContextHub_abductive" "ContextHub_deductive"   "arc_challenge" "arc_easy" "MuSR" "lsat" "commensenseqa"  "piqa" "siqa" "strategyqa" "crows" "sst-2"
    do             
        for experiment in    'standard'  'cot' 'direct_answer'
        do
            if [ "$model" = "Mistral-7B-Instruct-v0.3" ]; then
                model_type="mistralai"
            elif [ "$model" = "Llama-3.1-8B-Instruct" ] ||  [ "$model" = "Llama-3.2-3B-Instruct" ]; then
                model_type="llama"
            elif [ "$model" = "Phi-3.5-mini-instruct" ]; then
                model_type="microsoft"
            else
                model_type="none"
            fi

            data_file="benchmark/$benchmark/data.jsonl"
            output_fname="outputs/$benchmark/$model_type/$experiment/$model.jsonl"
            output_name="outputs/$benchmark/$model_type/$experiment/$model-sc.jsonl"
            result_path="outputs/All_results.json"

            python3 cal_instance_sc.py \
                --input_file "$output_fname"  \
                --output_file "$output_name" \
                --result_path "$result_path" \
                --model "$model" \
                --benchmark "$benchmark"
        done
    done
done

##################################################################### aggregated_sc #######################################################################
for model in   "Llama-3.2-3B-Instruct" "Phi-3.5-mini-instruct" "Mistral-7B-Instruct-v0.3" "Llama-3.1-8B-Instruct" 
do
    for benchmark in "gsm8k" "MultiArith" "gpqa" "FOLIO" "ContextHub_abductive" "ContextHub_deductive"   "arc_challenge" "arc_easy" "MuSR" "lsat" "commensenseqa"  "piqa" "siqa" "strategyqa" "sst-2" #"crows" 
    do
        for experiment in    'standard' 'cot' 'direct_answer'
        do
            if [ "$model" = "Mistral-7B-Instruct-v0.3" ]; then
                model_type="mistralai"
            elif [ "$model" = "Llama-3.1-8B-Instruct" ] ||  [ "$model" = "Llama-3.2-3B-Instruct" ]; then
                model_type="llama"
            elif [ "$model" = "Phi-3.5-mini-instruct" ]; then
                model_type="microsoft"
            else
                model_type="none"
            fi

            data_file="benchmark/$benchmark/data.jsonl"
            output_fname="outputs/$benchmark/$model_type/$experiment/$model.jsonl"
            output_name="outputs/All_results.json"

            python3 cal_aggregated_sc.py \
                --input_file "$output_fname"  \
                --output_file "$output_name" \
                --model "$model" \
                --benchmark "$benchmark" 
        done
    done
done


#####################################################################token use#######################################################################
for model in   "Llama-3.2-3B-Instruct" "Phi-3.5-mini-instruct" "Mistral-7B-Instruct-v0.3" "Llama-3.1-8B-Instruct" 
do
    for benchmark in "gsm8k" "MultiArith" "gpqa" "FOLIO" "ContextHub_abductive" "ContextHub_deductive"   "arc_challenge" "arc_easy" "MuSR" "lsat" "commensenseqa"  "piqa" "siqa" "strategyqa" "sst-2" #"crows" 
    do
        for experiment in     'cot' 'direct_answer'   #'standard'
        do
            if [ "$model" = "Mistral-7B-Instruct-v0.3" ]; then
                model_type="mistralai"
            elif [ "$model" = "Llama-3.1-8B-Instruct" ] ||  [ "$model" = "Llama-3.2-3B-Instruct" ]; then
                model_type="llama"
            elif [ "$model" = "Phi-3.5-mini-instruct" ]; then
                model_type="microsoft"
            else
                model_type="none"
            fi

            data_file="benchmark/$benchmark/data.jsonl"
            output_fname="outputs/$benchmark/$model_type/$experiment/$model.jsonl"
            result_path="outputs/All_results.json"

            python3 cal_token_use.py \
                --input_file "$output_fname"  \
                --result_path "$result_path" \
                --model "$model" \
                --benchmark "$benchmark"

        done
    done
done
