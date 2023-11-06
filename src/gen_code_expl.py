import transformers, torch
import json
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from human_eval.data import write_jsonl, read_problems

prompt_file = "prompt.txt"
prompt_code_expl_file = "prompt_code_expl3.txt"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="human_eval test")
    parser.add_argument('-mf','--model_path',default='/lustre/S/hexiaolong/vicuna-7b-v1.1',required=True, help="file path where the model used locates")
    parser.add_argument('-o','--output',default="ouput.json",required=True, help="file where the output store")

    args = parser.parse_args()
    output_file = args.output
    model_path = args.model_path
    unit_tests = {}

    with open(prompt_file,"r") as f:
        preflex = f.read()
        print(preflex)
    with open(prompt_code_expl_file,"r") as f:
        pompt_code_expl = f.read()
        # print(pompt_code_expl)
    def get_one_complication(problem,unit_test):#为模型的输入添加前缀串，看能否提高准确度。目前直接返回原串。
        res = preflex + unit_test + "\n\n# Complete the Python funtion:\n" + problem["prompt"]
        # print("=================prompt==================")
        # print(res)
        # print("=================end==================")
        return res

    def get_unit_test(problems):
        taskids = list(problems.keys())
        for id in taskids:
            entry_point = problems[id]["entry_point"]
            test = problems[id]["test"].replace("candidate",entry_point).split("\n")
            test = [x.strip() for x in test if (entry_point in x and "assert" in x)]
            if "HumanEval/32" in id:
                test = ["assert find_zero([1, 2])==-0.5"]
            elif id == "HumanEval/1":
                test = ["assert candidate('(()()) ((())) () ((())()())') == ['(()())', '((()))', '()', '((())()())']"]
            unit_tests[id] = test[0]

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

    #加载vicuna模型
    print("load model from ",model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="sequential", trust_remote_code=True, max_memory=max_memory_mapping, torch_dtype=torch.float16, low_cpu_mem_usage=True, use_safetensors=True)#, use_safetensors=True

    #获取solution
    problems = read_problems()
    get_unit_test(problems)
    taskids = list(problems.keys())
    num_task = len(taskids)
    print("task num: ",num_task )
    f = open(output_file,"w+",encoding='utf-8')
    for id in taskids:
        print("get solution for task :",id)
        problem = get_one_complication(problems[id],unit_tests[id])
        input_len = len(problem)
        inputs = tokenizer(problem, return_tensors='pt', return_token_type_ids=False)
        inputs = inputs.to('cuda')
        pred = model.generate(**inputs, max_new_tokens=512, temperature=0,repetition_penalty=1.1)#,temperature=0.4
        ans = tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)[input_len-1:]
        solution = ans.strip('\n')#.split("```")[0]#.replace("->>","")
        print("=========origin solution====================")
        print(solution)
        print("============================================")
        idx1 = solution.find("### result ###")
        idx2 = solution.find("### Task End ###")
        if idx1 != -1:
            solution = solution[idx1:idx2-1].replace("### result ###\n","")
        else:
            solution = solution[:idx2-1]
        codes = solution.split("\n")
        if not solution.startswith("def"):
            res = solution
        else:
            res = "\n".join(codes[1:])
        solution = res
        print(solution)
        print("============================================")
        entry_point = problems[id]["entry_point"]
        start_code = ""
        for line in problems[id]["prompt"].split('\"\"\"')[0].split("\n"):
            if entry_point in line and "def" in line:
                start_code += line + "\n"
                break
            start_code += line + "\n"
        print("=====start code===========")
        print(start_code)
        # f.write(json.dumps(output)+"\n")
        prompt = pompt_code_expl + "\n" + start_code + solution + "\n"
        if "0" in id:
            print("########################################")
            print(prompt)
            print("########################################")
        input_len = len(prompt)
        inputs = tokenizer(prompt, return_tensors='pt', return_token_type_ids=False)
        inputs = inputs.to('cuda')
        pred = model.generate(**inputs, max_new_tokens=2048, temperature=0,repetition_penalty=1.1)#,temperature=0.4
        ans = tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)[input_len:]
        print("--------------------code expl-------------------")
        print(ans)
        print("------------------------final expl------------------------")
        code_linen = len([x for x in (start_code + "\n" + solution).split("\n") if x!=""]) + 1
        ans = "\n".join(ans.split("\n")[:code_linen])
        print(ans)
        print("-----------------------------------------------")
        # if "20" in id:
        #     break
    f.close()


