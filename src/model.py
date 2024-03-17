from transformers import AutoModelForCausalLM, AutoTokenizer
from myutils import map_gpu_memory,filter_fix_ans
import torch
import time
import numpy as np

class Model():
    def __init__(self,model_path):
        #提取模型名字
        if model_path[-1] == '/':
            model_path = model_path[:-1]
        self.model_name = model_path.split('/')[-1]
        
        #为模型的多卡运行分配显存，默认使用了一个服务器上的所有显卡，也就是4张。这里直接从fastchat中的源码摘取了部分
        max_memory_mapping = map_gpu_memory(used_gpu=[])

        #加载模型
        print("load model from ",model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True,legacy=False)#, max_memory=max_memory_mapping
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16, low_cpu_mem_usage=True)#, use_safetensors=True
        self.model.eval()
        
    