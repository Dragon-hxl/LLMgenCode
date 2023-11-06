import torch
import json
import argparse
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

# get args for humaneval test on LLM
def get_args():
    parser = argparse.ArgumentParser(description="human_eval simplefeedback")
    parser.add_argument('-mf','--model_path',default='/lustre/S/hexiaolong/vicuna-7b-v1.1',required=True, help="file path where the model used locates")
    parser.add_argument('-o','--output',default="ouput.json",required=True, help="file where the output store")
    parser.add_argument("-v","--verbose", default=False, action="store_true", help="show log")

    args = parser.parse_args()
    
    return args

#为模型的多卡运行分配显存，默认使用了一个服务器上的所有显卡，也就是4张。这里直接从fastchat中的源码摘取了部分
def map_gpu_memory(used_gpu):
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
    # used_gpu = []
    memory_mapping ={}
    if used_gpu!=[]:
        for i in used_gpu:
            memory_mapping[i] = max_memory_mapping[i]
        max_memory_mapping = memory_mapping
    print(max_memory_mapping)
    return max_memory_mapping

def get_unit_test(ut_file, chosen_num=1000,verbose=False):
    unit_tests = {}
    assertions = defaultdict(list)
    assertion_string = {}
    with open(ut_file,"r") as f:
        for line in f.readlines():
            data = json.loads(line)
            tid = data["task_id"]
            ios = data["ios"][:chosen_num]
            unit_tests[tid]=ios
            for io in ios:
                assertions[tid].append("assert "+io["tin"] + " == " + io["tout"])
            assertion_string[tid] = "\n".join(assertions[tid])
            if verbose:
                print(f"Read {len(assertions[tid])} tests from file for task {tid}")
    return unit_tests,assertions,assertion_string



# 接受一个完整的函数代码，去除其中的函数头和注释（humaneval检测需要）
def code_clean(code,entry_point,start_code=""):
    # regex = "\"\"\"[.\n]*\"\"\""
    # 使用正则表达式匹配和移除单行注释
    code = re.sub(r'#.*', '', code)

    # 使用正则表达式匹配和移除多行注释
    code = re.sub(r'(\'\'\'(.*?)\'\'\'|\"\"\"(.*?)\"\"\")', '', code, flags=re.DOTALL)

    # code = re.sub(regex,"",code)
    if entry_point in code:
        res = ""
        prefix = "    "
        for line in code.split("\n"):
            if line=="" or line=="\t" or line=="    ":
                continue
            if entry_point in line:
                prefix = ""
                continue
            if line in start_code:
                continue
            if line[0]!=" " and line[0]!="\t" and not line.startswith("def") and not line.startswith("import") and not line.startswith("from"):
                continue
            res += prefix + line + "\n"
        return res
    else:
        res = ""
        prefix = "    "
        for line in code.split("\n"):
            res += prefix + line + "\n"
        return res
    return code

def code_clean2(code,entry_point,start_code=""):
    """去除code中的注释和空行，去除code中和start_code重复的部分。entry_point是一个函数名，code中在entry_point这一行之前的行要加上缩进，之后的行中的函数体要保留。
    """
    # regex = "\"\"\"[.\n]*\"\"\""
    # 使用正则表达式匹配和移除单行注释
    code = re.sub(r'#.*', '', code)

    # 使用正则表达式匹配和移除多行注释
    code = re.sub(r'(\'\'\'(.*?)\'\'\'|\"\"\"(.*?)\"\"\")', '', code, flags=re.DOTALL)

    # code = re.sub(regex,"",code)
    if entry_point in code:
        res = ""
        prefix = "    "
        for line in code.split("\n"):
            if line=="" or line=="\t" or line=="    ":
                continue
            if entry_point in line:
                prefix = ""
                continue
            if line in start_code:# "import" in line and l
                continue
            if line[0]!=" " and line[0]!="\t" and not line.startswith("def") and not line.startswith("import") and not line.startswith("from"):
                continue
            res += prefix + line + "\n"
        return res
    else:
        res = ""
        prefix = "    "
        for line in code.split("\n"):
            res += prefix + line + "\n"
        return res
    return code

prompt_for_64 = '''def vowels_count(s):
    """Write a function vowels_count which takes a string representing
    a word as input and returns the number of vowels in the string.
    Vowels in this case are 'a', 'e', 'i', 'o', 'u'. Here, 'y' is also a
    vowel, but only when it is at the end of the given word.

    Example:
    >>> vowels_count("abcde")
    2
    >>> vowels_count("ACEDY")
    3
    """ '''
    
def get_UTfeedback_prompt(feedback_prompt, solution, code_res, run_test, test_res, assertions):
    total_tests = len(run_test)
    pass_tests = 0
    passn = 0.0
    if type(code_res) is str:
        prompt =  feedback_prompt +solution + "\n```\nFeedback: With the above function, " + run_test[0] +" returns the following error:\n\"\"\"\n"+code_res+ "\n\"\"\"\nSo the code does not pass the assertion. Please fix it.\n\n### fix result ###\n"
    else:
        utFeedback = "\n```\nFeedback: With the above function,"
        for i,cres in enumerate(code_res):
            try:
                cres = str(cres)
            except:
                continue
            if len(cres) >1024:
                print("Too long feedback, maybe the result of code is too wired!")
                continue
            # print(f"code return res is {cres}. real res is {test_res[i]}")
            if cres == test_res[i]:
                utFeedback += f" {run_test[i]} == {cres} while the assertion is \"{assertions[i]}\".The code pass this aasertion."
                pass_tests += 1
            else:
                utFeedback += f" {run_test[i]} == {cres} while the assertion is \"{assertions[i]}\".The code does not pass this aasertion."
        utFeedback += "\nSo the code is wrong. Please fix it.\n\n### fix result ###\n"
        prompt = feedback_prompt +solution + utFeedback
        passn = (1.0*pass_tests)/total_tests
    return prompt,passn
def filter_fix_ans(ans, entry_point, start_code,verbose=False):
    # print("=================fix ans=====================")
    # print(ans)
    if "```" in ans: # 过滤掉生成的无用信息
        tmp = ans.split("```")
        if len(tmp) > 1:
            sol = tmp[1].strip()
        else:
            sol = ans
    else:
        sol = ans
    if sol.startswith("python"):
        sol = sol[6:]
    sol = sol.strip("\n")
    # 去除函数头和注释
    sol = code_clean2(code=sol,entry_point=entry_point,start_code=start_code)
    if verbose:
        print("-------------filter fix ans----------------")
        print(sol)
        print("============fix end===============")
    return sol