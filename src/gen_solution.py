import transformers, torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from human_eval.data import  read_problems
from myutils import map_gpu_memory,code_clean2,get_args
from collections import defaultdict
import time
prompt_file = "prompt_base2.txt"
ut_file = "tests_from_prompt.jsonl"

if __name__ == "__main__":

    args = get_args()
    output_file = args.output
    model_path = args.model_path
    unit_tests = {}
    assertions = defaultdict(list)
    assertion_strings = {}

    with open(prompt_file,"r") as f:
        preflex = f.read()
    
    def get_one_complication(problem,unit_test):#为模型的输入添加前缀串，看能否提高准确度。目前直接返回原串。
        res = preflex + unit_test + "\n\n# Complete the Python funtion:\n" + problem["prompt"] + "### result ###\n"
        return res

    def get_unit_test():
        with open(ut_file,"r") as f:
            for line in f.readlines():
                data = json.loads(line)
                tid = data["task_id"]
                ios = data["ios"]
                unit_tests[tid]=ios
                for io in ios:
                    assertions[tid].append("assert "+io["tin"] + " == " + io["tout"])
                assertion_strings[tid] = "\n".join(assertions[tid])

    #为模型的多卡运行分配显存，默认使用了一个服务器上的所有显卡，也就是8张。这里直接从fastchat中的源码摘取了部分
    max_memory_mapping = map_gpu_memory(used_gpu=[])

    #加载模型
    print("load model from ",model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True,legacy=False)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="sequential", trust_remote_code=True, max_memory=max_memory_mapping, torch_dtype=torch.float16, low_cpu_mem_usage=True)#, use_safetensors=True
    model.tie_weights()
    
    #获取solution
    problems = read_problems()
    get_unit_test()
    taskids = list(problems.keys())
    num_task = len(taskids)
    print("task num: ",num_task )
    f = open(output_file,"w+",encoding='utf-8')
    for tid in taskids:
        print("get solutions for task :",tid)
        num_id = int(tid.split("/")[1])
        if num_id < 72:
            continue
        problem = get_one_complication(problems[tid],assertion_strings[tid])
        input_len = len(problem)
        inputs = tokenizer(problem, return_tensors='pt', return_token_type_ids=False)
        inputs = inputs.to('cuda')
        solutions = set()
        cir = 0
        start_time = time.time()
        while cir < 20:
            pred = model.generate(**inputs, max_new_tokens=512, temperature=1.0,top_k=20,top_p=0.95,do_sample=True,num_return_sequences=10)#,temperature=0.4,repetition_penalty=1.1
            for i,p in enumerate(pred):
                ans = tokenizer.decode(p, skip_special_tokens=True)[input_len:]
                solution = ans.strip("\n")
                # print("+++++++++++origin solution++++++++++++++++++")
                # print(solution)
                # 截取程序
                idx2 = solution.find("### Task End ###")
                if idx2 != -1:
                    solution = solution[:idx2-1] #这里的减1是为了去掉前面的换行
                tmp = solution.split("```")
                if len(tmp)>1:
                    solution = tmp[1]
                else:
                    print("code not wraped by  ```")
                if solution.startswith("python"):
                    solution = solution[6:]
                # solution = solution.strip("\n")
                # 去除函数头和注释
                entry_point = "def " + problems[tid]["entry_point"]
                solution = code_clean2(code=solution,entry_point=entry_point)
                if solution not in solutions:
                    solutions.add(solution)
                    print("+++++++++++filter solution++++++++++++++++++")
                    print(solution)
                    # print("++++++++++++++++++++++++++++++++++++++++++++")
            if len(solutions) >= 100:
                break
        end_time = time.time()
        print(f"gen {len(solutions)} soluitons spends {(end_time-start_time)/60} mins.")
        output = {"task_id": tid,"completion":list(solutions)}
        f.write(json.dumps(output)+"\n")
        f.flush()
    f.close()


