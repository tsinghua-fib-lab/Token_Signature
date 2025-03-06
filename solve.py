import collections
from dataclasses import dataclass, field
import torch
import torch.nn.functional as F

@dataclass
class DecodingArguments:
    encode_format: str = field(default="instruct")  
    max_new_tokens: int = field(default=512)  
    decoding: str = field(default="greedy")  
    cot_n_branches: int = field(default=10)
    cot_aggregate: str = field(default="max") 
    benchmark: str = field(default="gsm8k")


def greedy_decoding_solve(model, tokenizer, task, batch, args: DecodingArguments, device):
    # Ensure inputs are on the correct device
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    
    # Perform greedy decoding
    gen_ids = model.generate(
        input_ids=input_ids, attention_mask=attention_mask,
        do_sample=False, max_new_tokens=args.max_new_tokens,
        output_scores=True, return_dict_in_generate=True, pad_token_id=tokenizer.pad_token_id,return_legacy_cache=True  # Keep legacy format if needed
    )
    
    # Extract the log-probabilities from the scores
    gen_probs = torch.stack(gen_ids['scores'], dim=1).softmax(-1)
    n_vocab = gen_probs.shape[-1]
    gen_probs = gen_probs.reshape(len(gen_ids.sequences), 1, -1, n_vocab)

    ret = []
    for i, sequence in enumerate(gen_ids.sequences):
        # Decode the generated tokens (skip input part)
        input_text = tokenizer.decode(input_ids[i], skip_special_tokens=True)
        generated_text = tokenizer.decode(sequence[batch['input_ids'].shape[-1]:], skip_special_tokens=True)

        # Get token probabilities
        logits = gen_probs[i]
        token_probabilities = []
        for j in range(logits.shape[1]):
            token_logits = logits[0, j, :]
            max_token_logit = token_logits.max(dim=-1).values
            token_probabilities.append(max_token_logit)
        
        # Convert to CPU values and filter out padding tokens (token_id == 2)
        token_probs = [tensor.item() for tensor in token_probabilities]
        token_ids = sequence[batch['input_ids'].shape[-1]:].tolist()

        filtered_token_ids = []
        filtered_token_probs = []

        for token_id, token_prob in zip(token_ids, token_probs):
            if token_id != tokenizer.pad_token_id:  # Use tokenizer.pad_token_id instead of hardcoded 2
                filtered_token_ids.append(token_id)
                filtered_token_probs.append(token_prob)

        # Append the result
        ret.append({
            'prompt':input_text,
            'generated_text': generated_text,
            'token_ids': filtered_token_ids,
            'token_probs': filtered_token_probs,
            'len_token': len(filtered_token_ids),
            'len_probs': len(filtered_token_probs),
        })
    
    return ret


def solve(model, tokenizer, task, batch, args: DecodingArguments, device):
    return greedy_decoding_solve(model, tokenizer, task, batch, args, device)
   