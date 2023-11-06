import transformers, torch
import json
import random
import fire
import sys
import numpy as np
sys.path.append("../src")
from transformers import AutoModelForCausalLM, AutoTokenizer
from myutils import map_gpu_memory,get_args,code_clean,code_clean2,get_unit_test,prompt_for_64,get_UTfeedback_prompt,filter_fix_ans
from collections import defaultdict

def main(
    model_path: str,
    output_file: str, 
):
    #为模型的多卡运行分配显存，默认使用了一个服务器上的所有显卡，也就是4张。这里直接从fastchat中的源码摘取了部分
    max_memory_mapping = map_gpu_memory(used_gpu=[])

    #加载模型
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True,legacy=False)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True, max_memory=max_memory_mapping, torch_dtype=torch.float16, low_cpu_mem_usage=True)#, use_safetensors=True
    model.tie_weights()
    # model.config.pad_token_id = model.config.eos_token_id
    sample_num = 10
    
    prompt = "Q:Write a sentence about spring.\nA:"
    inputs = tokenizer(prompt,return_tensors='pt', return_token_type_ids=False)
    inputs = inputs.to('cuda')
    input_length = inputs.input_ids.shape[1]
    print(f"input length {input_length}")
    with torch.no_grad():
        preds = model.generate(**inputs, max_new_tokens=30,temperature=1.0,top_p=0.95, do_sample=True,num_return_sequences=sample_num,return_dict_in_generate=True,output_scores=True) 
        transition_scores = model.compute_transition_scores(
        preds.sequences, preds.scores, normalize_logits=True
        ).cpu()
        print(type(transition_scores))
        print(len(transition_scores))
        print(len(transition_scores[0]))
        print(transition_scores)
        print(preds.keys())
        # print()
        print(type(preds["sequences"]))
        print(type(preds["scores"]))
        print(len(preds["sequences"]))
        print(len(preds["scores"]))
        print(preds["scores"][0])
        print(preds["scores"][-1])
        print(type(preds["scores"][0]))
        print(preds["scores"][0].size())
        print(preds["scores"][-1].size())
        print(tokenizer.decode(tokenizer.eos_token_id))
        print("pad_token_id",model.config.pad_token_id)
        print(tokenizer.decode(model.config.pad_token_id))
        seq_to_score = {}
        seq_to_score2 = {}
    for pred,transition_score in zip(preds["sequences"],transition_scores.numpy()):
        gen_tokens = pred[input_length:].cpu()
        valid = np.isfinite(transition_score)
        tsc = transition_score[valid]
        output_length = input_length + tsc.shape[-1]
        sc = np.sum(tsc,axis=-1)/output_length
        true_sc = np.exp(sc)
        true_sc2 = np.exp(np.sum(tsc,axis=-1)/tsc.shape[-1])
        print(f"gen_length: {tsc.shape[-1]},output_length: {output_length}, tsc: {sc}, true_sc: {true_sc}\n{tsc}")
        print(gen_tokens)
        for tok,score in zip(gen_tokens, transition_score):
            print(f"| {tok:5d} | {tokenizer.decode(tok):8s} | {score:.3f} | {np.exp(score):.2%}")
        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        print(gen_text)
        print(transition_score)
        seq_to_score[gen_text] = true_sc
        seq_to_score2[gen_text] = true_sc2
        print("--------------------")
    for k,v in sorted(seq_to_score.items(),key=lambda x: x[1],reverse=True):
        print(f"seq: {k}  score: {v}")
    for k,v in sorted(seq_to_score2.items(),key=lambda x: x[1],reverse=True):
        print(f"seq: {k}  score: {v}")
    return

class people:
    def __init__(self,age=0,weight=0):
        self.age = age
        self.weight = weight


if __name__=="__main__":
    # fire.Fire(main)
    
    # ages = [random.randint(0,100) for i in range(10)]
    ages = [10, 16, 10, 16, 19, 17, 30, 32, 10, 16]
    age_set = list(set(ages))
    print(age_set)
    weights = [random.randint(0,300) for i in range(10)]
    peoples = []
    for a,w in zip(ages,weights):
        peoples.append(people(a,w))
    for p in peoples:
        print(f"age: {p.age} weight: {p.weight}")
    peoples = sorted(peoples,key=lambda x: (x.age,x.weight))
    print("============================================")
    for p in peoples:
        print(f"age: {p.age} weight: {p.weight}")
    # print("==========tsc====================")
    #     print(transition_scores.numpy())
    #     valid = np.isfinite(transition_scores.numpy())
    #     print(valid)
    #     print(type(valid))
    #     print(np.array(valid))
    #     tsc = transition_scores.numpy()[np.array(valid)]
    #     print(tsc)
    #     print("==========tsc====================")
    #     output_length = input_length + np.sum(tsc < 0,axis=1)
    #     length_penalty = model.generation_config.length_penalty
    #     reconstructed_scores = tsc.sum(axis=1) / (output_length**length_penalty)