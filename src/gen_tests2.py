import transformers, torch
import json
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from human_eval.data import read_problems
# from human_eval.execution import check_test_correctness
# from concurrent.futures import ThreadPoolExecutor
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# prompt_file = "prompt.txt"
shot_file = "gen_test_shot.txt"
def get_one_shot():
    with open(shot_file,"r") as f:
        shot = f.read()
    return shot

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="human_eval test")
    parser.add_argument('-mf','--model_path',default='/lustre/S/hexiaolong/vicuna-7b-v1.1',required=True, help="file path where the model used locates")
    parser.add_argument('-o','--output',default="ouput.json",required=True, help="file where the output store")

    args = parser.parse_args()
    output_file = args.output
    model_path = args.model_path
    unit_tests = {}

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
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True,legacy=False)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="sequential", trust_remote_code=True, max_memory=max_memory_mapping, torch_dtype=torch.float16, low_cpu_mem_usage=True)#, use_safetensors=True

    #获取solution
    problems = read_problems()
    taskids = list(problems.keys())
    num_task = len(taskids)
    print("task num: ",num_task )
    f = open(output_file,"w+",encoding='utf-8')
    # lack_id = [85, 100, 113, 79, 119, 127, 130, 39]
    shot = get_one_shot()
    for id in taskids:
        # tid = int(id.split("/")[1])
        # if tid not in lack_id:
        #     continue
        # prompt = get_one_shot(problems)
        cir =0
        tests = []
        test_in_set = set()
        test_get = set()
        already_gen =""
        temperature = 1.0
        top_k = 50
        end_cir = 10
        while cir < end_cir:
            print(f"start cir : {cir}")
            prompt = ""
            for line in problems[id]["prompt"].split("\n"):
                if ">>>" in line or "Example" in line or "For example" in line:
                    prompt += "    \"\"\"\n"
                    break
                prompt += line + "\n"
            # for t in test_get:
            #     already_gen += t + "\n"
            # problem = prompt + "\tpass\n\n" + "# Check the correctness of " + problems[id]["entry_point"] +" with 50 tests:\n" + already_gen +"\nassert"
            problem = shot + prompt + "\tpass\n\n" + "# Check the correctness of " + problems[id]["entry_point"] +" with 15 tests:\nassert"
            # problem = prompt + "\tpass\n\n" + "# Check the correctness of " + problems[id]["entry_point"] +"\n" + already_gen +"\nassert"
            input_len = len(problem)
            inputs = tokenizer(problem, return_tensors='pt', return_token_type_ids=False)
            inputs = inputs.to('cuda')
            # pred = model.generate(**inputs, max_new_tokens=2048,top_p=0.95,temperature=temperature,repetition_penalty=1.1)#,temperature=0.4
            pred = model.generate(**inputs, max_new_tokens=512,top_k=top_k,top_p=0.95,do_sample=True,temperature=temperature,num_return_sequences=10)#,repetition_penalty=1.1
            ans = ""
            for p in pred:
                ans += tokenizer.decode(p, skip_special_tokens=True)[input_len-7:].strip() + "\nnext ans :\n"
            print(ans)
            print("============================================")
            entry_point = "assert " + problems[id]["entry_point"] + "("
            # print(f"--------{entry_point}----------------")
            for line in ans.split("\n"):
                if entry_point in line and "==" in line and "# assert" not in line:
                    test_in = line.split("==")[0]
                    test_in = line.split("==")[0][test_in.index("assert ")+7:].strip()
                    test_out = line.split("==")[1].strip()
                    if test_in in test_in_set:
                        continue
                    print(f"gen testcase :  {test_in} == {test_out}")
                    tests.append({"tin":test_in,"tout":test_out})
                    test_in_set.add(test_in)
                    test_get.add(line)
                    print("++++++++++++++++++++++++++++++++++++++++")
            if len(tests) > 300:
                cir = 1024
                f.write(json.dumps({id:tests})+"\n")
                print(f"for task {id} gen tests num: {len(tests)}\n")
            else:
                cir += 1
                # temperature -=0.05
                top_k += 5
                if cir == end_cir:
                    f.write(json.dumps({id:tests})+"\n")
                    print(f"for task {id} gen tests failed with num: {len(tests)}\n")
    f.close()

