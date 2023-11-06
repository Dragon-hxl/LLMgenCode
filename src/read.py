import transformers, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from human_eval.data import write_jsonl, read_problems

def get_one_complication(s):#为模型的输入添加前缀串，看能否提高准确度。目前直接返回原串。
    preflex = "hello,can you complete the code bleow: \n\n"
    problem = preflex + s
    return s + "\n"

#为模型的多卡运行分配显存，默认使用了一个服务器上的所有显卡，也就是4张。这里直接从fastchat中的源码摘取了部分
gpu_memory = []
num_gpus = torch.cuda.device_count()
for gpu_id in range(num_gpus):
    with torch.cuda.device(gpu_id):
        device = torch.cuda.current_device()
        gpu_properties = torch.cuda.get_device_properties(device)
        total_memory = gpu_properties.total_memory / (1024**3)
        allocated_memory = torch.cuda.memory_allocated() / (1024**3)
        available_memory = total_memory - allocated_memory
        gpu_memory.append(available_memory)
max_memory_mapping = {
                    i: str(int(gpu_memory[i] * 0.85)) + "GiB"
                    for i in range(num_gpus)
                }
print(max_memory_mapping)

#加载vicuna模型
model_path = "/lustre/S/hexiaolong/vicuna-7b-v1.1"
print("load model from ",model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="sequential", trust_remote_code=True, max_memory=max_memory_mapping, torch_dtype=torch.float16, low_cpu_mem_usage=True)
#获取模型输入
print("load inputs")
# inputs = tokenizer('hello, i will go to beijing 7 days, could you please give me a journey list?', return_tensors='pt', return_token_type_ids=False)#\n->
problems = read_problems()
inputs = tokenizer(get_one_complication(problems['HumanEval/0']["prompt"]), return_tensors='pt', return_token_type_ids=False)
input_len = len(get_one_complication(problems['HumanEval/0']["prompt"]))
inputs = inputs.to('cuda')
print(problems['HumanEval/0'].keys())
print(problems['HumanEval/0'])

print("get answer")
pred = model.generate(**inputs, max_new_tokens=512, repetition_penalty=1.1)
ans = tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)
print("====================================================")
print(ans)
print("====================================================")
solution = ans.split("```")[0].replace("->>","")
print(solution)