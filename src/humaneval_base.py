import transformers, torch
import json
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from human_eval.data import  read_problems
from myutils import map_gpu_memory,code_clean2,get_args
from collections import defaultdict

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
        print("=================prompt==================")
        print(unit_test + "\n\n# Complete the Python funtion:\n" + problem["prompt"] + "### result ###\n")
        print("=================end==================")
        return res

    def get_unit_test():
        with open(ut_file,"r") as f:
            for line in f.readlines():
                data = json.loads(line)
                tid = data["tid"]
                ios = data["ios"]
                unit_tests[tid]=ios
                for io in ios:
                    assertions[tid].append("assert "+io["tin"] + " == " + io["tout"])
                assertion_strings[tid] = "\n".join(assertions[tid])

    #为模型的多卡运行分配显存，默认使用了一个服务器上的所有显卡，也就是4张。这里直接从fastchat中的源码摘取了部分
    max_memory_mapping = map_gpu_memory(used_gpu=[])

    #加载模型
    print("load model from ",model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True,legacy=False)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="sequential", trust_remote_code=True, max_memory=max_memory_mapping, torch_dtype=torch.float16, low_cpu_mem_usage=True, use_safetensors=True)#, use_safetensors=True
    model.tie_weights()
    
    #获取solution
    problems = read_problems()
    get_unit_test()
    taskids = list(problems.keys())
    num_task = len(taskids)
    print("task num: ",num_task )
    f = open(output_file,"w+",encoding='utf-8')
    for tid in taskids:
        print("get solution for task :",tid)
        problem = get_one_complication(problems[tid],assertion_strings[tid])
        input_len = len(problem)
        inputs = tokenizer(problem, return_tensors='pt', return_token_type_ids=False)
        inputs = inputs.to('cuda')
        pred = model.generate(**inputs, max_new_tokens=512, temperature=0)#,temperature=0.4,repetition_penalty=1.1
        ans = tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)[input_len:]
        solution = ans.strip("\n")
        print("+++++++++++origin solution++++++++++++++++++")
        print(solution)
        # 截取程序
        idx2 = solution.find("### Task End ###")
        if idx2 != -1:
            solution = solution[:idx2-1] #这里的减1是为了去掉前面的换行
        if len(solution.split("```"))>1:
            solution = solution.split("```")[1]
        else:
            print("code not wraped by  ```")
            # print(solution.split("```"))
        if solution.startswith("python"):
            solution = solution[6:]
        solution = solution.strip("\n")
        # 去除函数头和注释
        entry_point = "def " + problems[tid]["entry_point"]
        solution = code_clean2(code=solution,entry_point=entry_point)
        print("+++++++++++filter solution++++++++++++++++++")
        print(solution)
        print("++++++++++++++++++++++++++++++++++++++++++++")
        output = {"task_id": tid,"completion":solution}
        f.write(json.dumps(output)+"\n")
    f.close()


