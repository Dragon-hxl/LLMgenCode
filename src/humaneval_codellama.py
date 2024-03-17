import transformers, torch
import json
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from human_eval.data import write_jsonl, read_problems
from itertools import product

prompt_file = "prompt_codellama.txt"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="human_eval test")
    parser.add_argument('-mf','--model_path',default='/lustre/S/hexiaolong/vicuna-7b-v1.1',required=True, help="file path where the model used locates")
    parser.add_argument('-o','--output',default="ouput.json",required=True, help="file where the output store")

    args = parser.parse_args()
    # result_root = "./"
    output_file = args.output
    model_path = args.model_path
    
    with open(prompt_file,"r") as pf:
        shot = pf.read()

    def get_one_complication(s):#为模型的输入添加前缀串，看能否提高准确度。目前直接返回原串。
        preflex = "[INST]\nComplete the function:\n"
        preflex2 = f"You are an expert Python programmer, and your task is to complete a Python funtion.Here is the function you should complete:\n{s}\nYour code should start with a [PYTHON] tag and end with a [/PYTHON] tag."
        preflex3 = f"Given the function definition:\n{s}\nPlease provide the best implementation for it.\n# your answer.\n"
        p = preflex + s +"\n[\INST]\n[ANSWER]\n"
        # start_idx = s.find("\"\"\"")
        # if ">>>" in s:
        #     end_idx = s.find(">>>")
        # elif "Example" in s:
        #     end_idx = s.find("For example")
        # elif "For example" in s:
        #     end_idx = s.find("For example")
        # task = s[start_idx:end_idx].replace("\n","").replace("    ","") + "\"\"\""
        # func_signature = [x for x in problems[id]["prompt"].split("\n") if "def" in x][0]
        # res = f"You are an expert Python programmer,and here is your task : \n{task}\nPlease implete the funtion to solve the task.\n{func_signature}"
        # preflex4="\\begin{code}\n"#[INST]Complete the following function[\INST]\n
        # res = preflex4 + s # shot + s + "\n### result ###\n"
        return 

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
    used_gpu = []
    memory_mapping ={}
    if used_gpu!=[]:
        for i in used_gpu:
            memory_mapping[i] = max_memory_mapping[i]
        max_memory_mapping = memory_mapping
    print(max_memory_mapping)

    #加载模型
    print("load model from ",model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="sequential", trust_remote_code=True, max_memory=max_memory_mapping, torch_dtype=torch.float16, low_cpu_mem_usage=True)#, use_safetensors=True

    temperature_list = [0.1]#, 0.4, 0.7, 1.0
    top_k_list = [20]#0,50
    top_p_list = [0.95]
    do_sample_list = [True]#False, 
    repetition_penalty_list = [1.0]#, 1.1

    #获取solution
    problems = read_problems()
    taskids = list(problems.keys())
    num_task = len(taskids)
    print("task num: ",num_task )
    f = open(output_file,"w+",encoding='utf-8')

    for t,top_k,top_p,do_sample,repetition_penalty in product(temperature_list,top_k_list,top_p_list,do_sample_list,repetition_penalty_list):
        print(t,top_k,top_p,do_sample,repetition_penalty)
        for id in taskids:
            # print("get solution for task :",id)
            problem = get_one_complication(problems[id]["prompt"])# problem = problems[id]["prompt"]
            # print("+"*20)
            # print(problem)
            # print("+"*20)
            input_len = len(problem)
            inputs = tokenizer(problem, return_tensors='pt', return_token_type_ids=False)
            inputs = inputs.to('cuda')
            pred = model.generate(**inputs, max_new_tokens=512,top_k=top_k,temperature=t,top_p=top_p,do_sample=do_sample,repetition_penalty=repetition_penalty)#,temperature=0.4  do_sample=True,,repetition_penalty=1.1
            ans = tokenizer.decode(pred.cpu()[0], skip_special_tokens=True).strip()#[input_len-2:]
            # solution = ans.split("```")[0]#.replace("->>","")
            print(f"=================={id} origin code==========================")
            print(ans)
            # print("==================final code==========================")
            # print(solution)
            print("============================================")
            output = {"task_id": id,"completion":ans}
            f.write(json.dumps(output)+"\n")
            if "10" in id:
                break
    f.close()
