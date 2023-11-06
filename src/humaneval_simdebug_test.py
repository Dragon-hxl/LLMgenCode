import transformers, torch
import json
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from human_eval.data import write_jsonl, read_problems
from human_eval.execution import check_one_correctness
from concurrent.futures import ThreadPoolExecutor
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

prompt_file = "prompt.txt"

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
    simfeedback_file = "prompt_simfeedback2.txt"
    with open(simfeedback_file,"r") as af:
        simde_promt = af.read()
    def get_one_complication(problem,unit_test):#为模型的输入添加前缀串，看能否提高准确度。目前直接返回原串。
        res = preflex + unit_test + "\n\n#Complete the Python funtion:\n" + problem["prompt"]
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
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="sequential", trust_remote_code=True, max_memory=max_memory_mapping, torch_dtype=torch.float16, low_cpu_mem_usage=True)#, use_safetensors=True

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
        solution = ans.strip()#.split("```")[0]#.replace("->>","")
        # print("=========origin solution====================")
        # print(solution)
        idx = solution.find("### Task End ###")
        solution = solution[:idx].replace("### result ###","")
        codes = solution.split("\n")
        print(solution)
        print("============================================")
        if "def" not in solution:
            res = solution
        else:
            res = "\n".join(codes[2:])
        solution = res
        print(solution)
        print("============================================")
        #self-debug step
        cir = 0 
        print("###### start self-debug process ######")
        while cir < 10:
            completion = solution
            # print("run code ......")
            result = {}
            with ThreadPoolExecutor(max_workers=1) as executor:
                args = (problems[id], completion, 3.0)
                future = executor.submit(check_one_correctness, *args)
                result = future.result()#check_one_correctness(problems[id],completion,3.0)
            # result = check_one_correctness(problems[id],completion,3.0)
            print(result)
            if result["passed"]:
                cir = 1000
            else:
                prompt = simde_promt + unit_tests[id] + "\n\n#Complete the Python funtion:\n" + problems[id]["prompt"]+"### result ###\n" + problems[id]["prompt"]+solution
                # print("---------------------feeedback prompt---------------------------")
                # print(prompt)
                # print("----------------------------------------------------------------")
                input_len = len(prompt)
                inputs = tokenizer(prompt, return_tensors='pt', return_token_type_ids=False)
                inputs = inputs.to('cuda')
                pred = model.generate(**inputs, max_new_tokens=512, temperature=0,repetition_penalty=1.1)#,temperature=0.4
                ans = tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)[input_len:]
                # idx = ans.find("Feedback")
                # if idx!=-1:
                #     solution = ans[:idx]
                codes = ans.split("\n")
                solution = ""
                flag = 0
                for code in codes:
                    if code=="":
                        continue
                    if code[0]!="\t" and code[0]!=" ":
                        if "def" not in code or flag==1:
                            break
                        if "def" in code and flag==0:
                            flag=1
                        continue
                    solution +=code + "\n"
                # solution = ans
                print("==================fix ans======================")
                print(ans)
                print("------------------------------------")
                print(solution)
                print("============fix end===============")
            cir += 1

        print("id:",id)
        output = {"task_id": id,"completion":solution}
        f.write(json.dumps(output)+"\n")
        # if "11" in id:
        #     print("Will finish")
        #     break
    f.close()
    # def self_debug():
    #     from human_eval.execution import check_one_correctness
    #     simde_promt = ""
    #     cir = 0
    #     while cir < 10:
    #         completion = solution
    #         result = check_one_correctness(problems[id],completion,50)
    #         if result["passed"]:
    #             break
    #         else:
    #             prompt = simde_promt + solution + "Feedback"


