## yphi.py ##
# Updates: 2024.10.05
# Author : Seongho Kim, Kunmo Jeong, Sungwoo Yun
# Email  : seongho-kim@yonsei.ac.kr
# Github : seongho-git
# Comment: 
# 1. Apply accelerate lib and pipeline optimization
# 2. Calculate batch size with memory capacity, especially in kv caching
# 3. Split the dataset into batches of 12
# 4. Dynamic split dataset
# 5. Adjust batch size dynamically
# 6. Add token length to the dataset and sort by ascending order
# 7. Add heurístic rule for not evoking CUDA out of memory error

import time
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers.pipelines.pt_utils import KeyDataset
# added
from accelerate import Accelerator
import psutil
import gc

####### Section 0. Custom functions for adjusting batch size #######
def check_memory_capacity():
    memory_info = psutil.virtual_memory()
    if torch.cuda.is_available():
        total_memory = int(memory_info.total) // (1024 ** 2)  # total memory (MB)
        allocated_memory = int(memory_info.used) // (1024 ** 2)  # allocated memory (MB)
        free_memory = int(memory_info.available) // (1024 ** 2)  # free memory (MB)
        print(f"Total memory: {total_memory:.2f} MB / Allocated memory: {allocated_memory:.2f} MB / Free memory: {free_memory:.2f} MB")
    else:
        print("CUDA device is not available.")

    return total_memory, free_memory

def estimate_memory_usage(model_config, input_texts, max_new_tokens, init_free_memory):
    # caluclate token length for each input
    print("===== Estimate memory usage =====")
    total_memory, free_memory = check_memory_capacity()
    MB = 1024 ** 2  # Bytes to MB conversion
    
    # tokenized_inputs = [tokenizer(input_text, return_tensors="pt") for input_text in input_texts]
    # input_lengths = [len(tokenized_input['input_ids'][0]) for tokenized_input in tokenized_inputs]
    # print(f"input_lengths: {input_lengths}")
    print(f"input_texts: {input_texts}")

    # Model configuration values
    batch_size = 1 # initial batch size
    param_size = 2  # bfloat16 = 2 bytes
    embedding_size = model_config.hidden_size  # Hidden size of the model
    # max_sequence_length = max(input_lengths)  # Maximum token length of the input
    max_sequence_length = max(input_texts)  # Maximum token length of the input
    max_sequence_length_add1 = max_sequence_length + 1  # Maximum token length of the input + 1
    max_context_length = min(max_sequence_length + max_new_tokens, model_config.max_position_embeddings)  # Maximum context length
    # num_attention_heads = model_config.num_attention_heads  # Number of attention heads
    # num_key_value_heads = model_config.num_key_value_heads  # Number of key-value heads
    # num_groups = num_attention_heads // num_key_value_heads  # Number of groups in attention
    # num_layers = model_config.num_hidden_layers  # Number of layers in the model
    # multiplicative_factor = num_layers // num_groups  # Multiplicative factor for memory calculation
    vocab_size = model_config.vocab_size  # Vocabulary size for logits
    intermediate_size = model_config.intermediate_size  # Intermediate size of the model

    # Memory calculation
    # input_memory = param_size * batch_size * max_sequence_length_add1 * vocab_size // MB
    # logits_memory = param_size * batch_size * max_sequence_length_add1 * vocab_size // MB
    vocab_memory = param_size * batch_size * max_sequence_length_add1 * vocab_size * 2 // MB
    
    # query_memory = param_size * batch_size * max_sequence_length_add1 * embedding_size // MB
    # scores_memory = param_size * batch_size * max_sequence_length_add1 * max_context_length // MB
    # context_memory = param_size * batch_size * max_sequence_length_add1 * embedding_size // MB
    # output_memory = param_size * batch_size * max_sequence_length_add1 * embedding_size // MB
    # ff_memory = param_size * batch_size * max_sequence_length_add1 * (intermediate_size * 2 + embedding_size) // MB

    # layer_memory = query_memory + scores_memory + context_memory + output_memory + ff_memory # Decoder layer memory
    decoder_memory = param_size * batch_size * max_sequence_length_add1 * (embedding_size * 4 + max_context_length + intermediate_size * 2) // MB
    kv_memory = param_size * batch_size * max_context_length * embedding_size * 2  // MB # Key and Value memory
    inference_memory = decoder_memory + kv_memory * model_config.num_hidden_layers + vocab_memory
    
    batch_size1 = int(init_free_memory // inference_memory)
    batch_size2 = int(free_memory // inference_memory)
    batch_size = max(batch_size1, batch_size2)
    # batch_size = max((batch_size1 + batch_size2) // 2, 1)
    
    print(f"decoder_memory : {decoder_memory:.2f} MB / logits_memory: {vocab_memory:.2f} MB")
    print(f"kv_memory: {kv_memory:.2f} MB / inference_memory: {inference_memory:.2f} MB / max_sequence_length: {max_sequence_length}")
    print(f"batch_size1: {batch_size1} = {init_free_memory} // {inference_memory} / batch_size2: {batch_size2} = {free_memory} // {inference_memory}")
    
    return batch_size, batch_size2, max_sequence_length

def dynamic_batch(model_config, inputs, max_new_tokens, init_free_memory):
    batch_size, batch_size2, max_sequence_length = estimate_memory_usage(model_config, inputs, max_new_tokens, init_free_memory)
    
    # for not evoking CUDA out of memory error
    # heurístic rule
    if max_sequence_length > 412:
        if batch_size > 1:
            batch_size = 1
    elif max_sequence_length > 260:
        if batch_size > 2:
            batch_size = 2
    elif max_sequence_length > 167:
        if batch_size > 3:
            batch_size = 3
    elif max_sequence_length > 132:
        if batch_size > 4:
            batch_size = 4
    elif max_sequence_length > 37:
        if batch_size > 6:
            batch_size = 6
    else:
        batch_size = 12
    
    if batch_size > len(inputs):
        batch_size = len(inputs)

    return batch_size, batch_size2

def add_token_length(dataset):
    dataset['token_length'] = tokenizer(dataset['message'][0]['content'], return_tensors="pt")['input_ids'].shape[1]
    return dataset

# Function to split the dataset into batches of 12
def data_split(data, split_size):
    return [data.select(range(i, min(i + split_size, len(data)))) for i in range(0, len(data), split_size)]

if __name__ == "__main__":
    ####### Section 1. Set up #######
    gc.collect()
    torch.cuda.empty_cache()
    
    torch.random.manual_seed(0)
    accelerator = Accelerator()

    model_id = "/mnt/usb/Phi-3-medium-4k-instruct" # Local model path
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cuda",
        torch_dtype=torch.bfloat16,  # maintain bfloat16
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model.config.max_position_embeddings = 1024  # 4096 > 1024 (1k) revised by notice
    model.eval() # added
    model_config = model.config

    generation_args = {
        "max_new_tokens": 500,
        "return_full_text": False,
        #"temperature": 0.0,
        "do_sample": False,
    }
    
    ####### Section 2. GPU Warm up #######
    messages = [
        {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
        {"role": "assistant", "content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."},
        {"role": "user", "content": "What about solving an 2x + 3 = 7 equation?"},
    ]

    # adjust the batch size dynamically
    init_total_memory, init_free_memory = check_memory_capacity()
    # batch_size, batch_size2 = dynamic_batch(model_config, [input["content"] for input in messages], generation_args["max_new_tokens"], init_free_memory)
    pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    # batch_size=batch_size,
    batch_size=3,
    )
    pipe = accelerator.prepare(pipe)

    with torch.no_grad():
        output = pipe(messages, **generation_args)
    print(output[0]['generated_text'])

    ####### Section 3. Load data and Inference part ####### 
    start = time.time()
    data = load_dataset("json", data_files="/mnt/usb/test_dataset.jsonl")['train']
    
    # sort data by length of content in message by ascending order
    data = data.map(add_token_length)
    data = data.sort("token_length")
    print(data['token_length'])
        
    # Split the dataset by 12
    data_splits = data_split(data, 12)

    with torch.no_grad():
        outs = []
        print("===== Start Inference =====")
        total_memory, free_memory = check_memory_capacity()
        
        for split in data_splits:
            start_split = time.time()
            split_outs = []          
            # batch_size, batch_size2 = dynamic_batch(model_config, [input["message"][0]["content"] for input in split], generation_args["max_new_tokens"], init_free_memory)
            batch_size, batch_size2 = dynamic_batch(model_config, [input["token_length"] for input in split], generation_args["max_new_tokens"], init_free_memory)
            print(f"len(inputs): {len(split)} / adjusted batch size: {batch_size}")
            
            if batch_size < 3 and batch_size2 < 1:
                under_split = data_split(split, 2)
                for block in under_split:
                    batch_size, batch_size2 = dynamic_batch(model_config, [input["token_length"] for input in block], generation_args["max_new_tokens"], init_free_memory)
                    print(f"block len(inputs): {len(block)} / adjusted block batch size: {batch_size}")
                    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, batch_size=batch_size,)
                    pipe = accelerator.prepare(pipe)
                    split_outs = pipe(KeyDataset(block, 'message'), **generation_args)
                    outs.extend(split_outs)
                    torch.cuda.empty_cache()
                end_split = time.time()
                print(f"total block Elapsed_time (sec/item): {(end_split-start_split)/len(split)}")
                continue
            
            pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, batch_size=batch_size,)
            pipe = accelerator.prepare(pipe)
            split_outs = pipe(KeyDataset(split, 'message'), **generation_args)
            outs.extend(split_outs)
            end_split = time.time()
            print(f"split Elapsed_time (sec/item): {(end_split-start_split)/len(split)}")
            torch.cuda.empty_cache()
    end = time.time()

    ####### Section 4. Accuracy (Just for leaderboard) #######
    print("===== Answers =====")
    correct = 0
    # if need, change original from rp
    for i, out in enumerate(outs):
        correct_answer = data[i]["answer"]
        answer = out[0]["generated_text"].lstrip().replace("\n","")
        if answer == correct_answer:
            correct += 1
        print(answer)

    print("===== Perf result =====")
    print(f"last batch_size: {batch_size}")
    print("Elapsed_time: ", end-start)
    print("Sec / Item: ", (end-start)/len(data))
    print(f"Correctness: {correct}/{len(data)}")