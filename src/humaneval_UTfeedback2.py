import transformers, torch
import json
import argparse
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from human_eval.data import write_jsonl, read_problems
from human_eval.execution import check_one_correctness, run_code_with_output
from concurrent.futures import ThreadPoolExecutor
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

prompt_file = "prompt.txt"
UTfeedback_file = "prompt_UTfeedback3.txt"

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

    with open(UTfeedback_file,"r") as af:
        UTfeedback_promt = af.read()

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
    # tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # model = AutoModelForCausalLM.from_pretrained(model_path, device_map="sequential", trust_remote_code=True, max_memory=max_memory_mapping, torch_dtype=torch.float16, low_cpu_mem_usage=True, use_safetensors=True)#
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    pipeline = transformers.pipeline("text-generation",model=model_path, torch_dtype=torch.float16,device_map="auto", max_memory=max_memory_mapping)
    
    #获取solution
    problems = read_problems()
    get_unit_test()
    taskids = list(problems.keys())
    num_task = len(taskids)
    print("task num: ",num_task )
    f = open(output_file,"w+",encoding='utf-8')
    for id in taskids:
        print("get solution for task :",id)
        problem = get_one_complication(problems[id],ut_prompt[id])
        input_len = len(problem)
        inputs = tokenizer(problem, return_tensors='pt', return_token_type_ids=False)
        inputs = inputs.to('cuda')
        # pred = model.generate(**inputs, max_new_tokens=512, temperature=0,repetition_penalty=1.1)#,temperature=0.4
        pred = pipeline(**input,do_sample=True,top_k=50,temperature=0.1,top_p=0.95,num_return_sequences=1,eos_token_id=tokenizer.eos_token_id,max_length=256)
        ans = tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)[input_len:]
        solution = ans.strip()#.split("```")[0]#.replace("->>","")
        print("=========origin solution====================")
        print(solution)
        print("============================================")
        idx = solution.find("### Task End ###")
        solution = solution[:idx].replace("### result ###","")
        codes = solution.split("\n")
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
                with ThreadPoolExecutor(max_workers=1) as executor:
                    ut = unit_tests[id]
                    print("check unit_test:",ut[0])
                    run_test = [ut[0]["in"]]
                    # test_res = json.loads([ut[0]["out"]])
                    test_res = ut[0]["out"]
                    args = (problems[id], completion, run_test, 3.0)
                    future = executor.submit(run_code_with_output, *args)
                    result2 = future.result()#check_one_correctness(problems[id],completion,3.0)
                    code_res = result2["result"]
                    print("code_res: ",code_res)
                if type(code_res) is str:
                    prompt = UTfeedback_promt + ut_prompt[id]+ "\n# Funtion to complete:\n" + problems[id]["prompt"]+"### result ###\n```\n" + problems[id]["prompt"].split('\"\"\"')[0] + "\n```\n" \
                      +solution + "\nFeedback: With the above function, " + unit_tests[id][0]["in"] +" returns the following error:\n\"\"\"\n"+code_res+ "\n\"\"\"\nSo the code does not pass the assertion. Please fix it."
                else:
                    # real_res = json.loads(code_res)
                    real_res = str(code_res)
                    if real_res == test_res:
                        prompt = UTfeedback_promt + ut_prompt[id]+ "\n# Funtion to complete:\n" + problems[id]["prompt"]+"### result ###\n```\n" + problems[id]["prompt"].split('\"\"\"')[0] + "\n" \
                      +solution + "\n```\nFeedback: With the above function, " + unit_tests[id][0]["in"] +" == "+ str(code_res) +". The assertion is '"+ut_prompt[id] +"'.\nSo the code pass the assertion. The code above is wrong. Please fix it."
                    else:
                        print(f"real_res: #{real_res}# does not equal to test_res:#{test_res}#")
                        prompt = UTfeedback_promt + ut_prompt[id]+ "\n# Funtion to complete:\n" + problems[id]["prompt"]+"### result ###\n```\n" + problems[id]["prompt"].split('\"\"\"')[0] + "\n" \
                      +solution + "\n```\nFeedback: With the above function, " + unit_tests[id][0]["in"] +" == "+ str(code_res) +". The assertion is \""+ut_prompt[id] +"\".\nSo the code does not pass the assertion. The code above is wrong. Please fix it."
                print("---------------------feeedback prompt---------------------------")
                print(prompt)
                print("----------------------------------------------------------------")
                input_len = len(prompt)
                inputs = tokenizer(prompt, return_tensors='pt', return_token_type_ids=False)
                inputs = inputs.to('cuda')
                pred = model.generate(**inputs, max_new_tokens=512, temperature=0,repetition_penalty=1.1)#,temperature=0.4
                ans = tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)[input_len:].strip()
                if ans.strip().startswith("```python"):
                    solution = ans.strip().split("```")[1][6:]
                elif ans.startswith("```"):
                    solution = ans.split("```")[1]
                else:
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
                print("==================fix ans======================")
                print(ans)
                print("------------------------------------")
                print(solution)
                print("============fix end===============")
            cir += 1

        print("id:",id)
        output = {"task_id": id,"completion":solution}
        f.write(json.dumps(output)+"\n")
        if "9" in id:
            print("Will finish")
            break
    f.close()


