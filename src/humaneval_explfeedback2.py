import transformers, torch
import json
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from human_eval.data import read_problems
from human_eval.execution import check_one_correctness, run_code_with_output
from concurrent.futures import ThreadPoolExecutor
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

prompt_file = "prompt.txt"
UTfeedback_file = "prompt_UTfeedback.txt"
explfeedback_file = "prompt_expl2.txt"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="human_eval test")
    parser.add_argument('-mf','--model_path',default='/lustre/S/hexiaolong/vicuna-7b-v1.1',required=True, help="file path where the model used locates")
    parser.add_argument('-o','--output',default="ouput.json",required=True, help="file where the output store")

    args = parser.parse_args()
    output_file = args.output
    model_path = args.model_path
    unit_tests = {}
    ut_prompt = {}

    with open(prompt_file,"r") as f:
        preflex = f.read()
        print(preflex)

    with open(explfeedback_file,"r") as af:
        explfeedback_prompt = af.read()

    def get_one_complication(problem,unit_test):#为模型的输入添加前缀串，看能否提高准确度。目前直接返回原串。
        res = preflex + unit_test + "\n\n#Complete the Python funtion:\n" + problem["prompt"]
        # print("=================unit test in prompt==================")
        # print(unit_test)
        # print("=================end==================")
        return res

    def get_unit_test():
        with open("unitTest.jsonl","r") as f:
            lines = f.readlines()
            for line in lines:
                data = json.loads(line)
                tid = data["task_id"]
                ut = data["unit_tests"]
                unit_tests[tid] = ut
                utp = "assert " + str(ut[0]["in"]) + str(ut[0]["op"]) + str(ut[0]["out"])
                ut_prompt[tid] = utp


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
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="sequential", trust_remote_code=True, max_memory=max_memory_mapping, torch_dtype=torch.float16, low_cpu_mem_usage=True, use_safetensors=True)#

    #获取solution
    problems = read_problems()
    get_unit_test()
    taskids = list(problems.keys())
    num_task = len(taskids)
    print("task num: ",num_task )
    f = open(output_file,"w+",encoding='utf-8')
    for id in taskids:
        problem = get_one_complication(problems[id],ut_prompt[id])
        input_len = len(problem)
        inputs = tokenizer(problem, return_tensors='pt', return_token_type_ids=False)
        inputs = inputs.to('cuda')
        pred = model.generate(**inputs, max_new_tokens=512, temperature=0,repetition_penalty=1.1)#,temperature=0.4
        ans = tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)[input_len:]
        solution = ans.strip("\n")#.split("```")[0]#.replace("->>","")
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
        print("+++++++++++filter solution++++++++++++++++++")
        print(solution)
        print("++++++++++++++++++++++++++++++++++++++++++++")
        #self-debug step
        entry_point = problems[id]["entry_point"]
        start_code = ""
        for line in problems[id]["prompt"].split('\"\"\"')[0].split("\n"):
            if entry_point in line and "def" in line:
                start_code += line + "\n"
                break
            start_code += line + "\n"
        print("=====start code===========")
        print(start_code)
        cir = 0 
        while cir < 2:
            completion = solution
            result = {}
            unit_result = {}
            with ThreadPoolExecutor(max_workers=1) as executor:
                args = (problems[id], completion, 3.0)
                future = executor.submit(check_one_correctness, *args)
                result = future.result()#check_one_correctness(problems[id],completion,3.0)
            print(result)
            if result["passed"]:
                cir = 1000
            else:
                print("###### start self-debug process ######")
                prompt = explfeedback_prompt + ut_prompt[id]+ "\n# Complete the Python funtion:\n" + problems[id]["prompt"]+"### result ###\n" + start_code + "\n" +solution
                print("---------------------feeedback prompt---------------------------")
                print(prompt)
                print("----------------------------------------------------------------")
                input_len = len(prompt)
                prompt = prompt[:8000]
                inputs = tokenizer(prompt, return_tensors='pt', return_token_type_ids=False)
                print("++++++++++++++++++++++++++++")
                print(type(input_prompt),input_prompt)
                print(type(input_prompt["input_ids"]),input_prompt["input_ids"].size())
                print(type(input_prompt["attention_mask"]),input_prompt["attention_mask"].size())
                inputs = inputs.to('cuda')
                pred = model.generate(**inputs, max_new_tokens=512, temperature=0,repetition_penalty=1.1)#,temperature=0.4
                ans = tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)[input_len:]
                if "```python" in ans:
                    solution = ans.strip().split("```")[1][6:]
                elif "```" in ans:
                    solution = ans.strip().split("```")[1]
                codes = solution.split("\n")
                solution = ""
                for code in codes:
                    if code=="":
                        continue
                    if code[0]!="\t" and code[0]!=" ":
                        continue
                    solution +=code + "\n"
                print("==================fix ans======================")
                print(ans)
                print("------------------------------------")
                print(solution)
                print("============fix end===============")
            cir += 1

        print("id:",id)
        output = {"task_id": id,"completion":solution}
        f.write(json.dumps(output)+"\n")
        if "5" in id:
            print("Will finish")
            break
    f.close()


