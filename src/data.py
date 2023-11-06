import json
from human_eval.data import write_jsonl, read_problems

def main():
    problems = read_problems()
    taskids = list(problems.keys())
    num_task = len(taskids)
    p_test = []
    start_codes = []
    print("task num: ",num_task )
    for id in taskids:
        print("get solution for task :",id)
        print("===================================================")
        print(problems[id].keys())
        print("===================================================")
        print("prompt:")
        print(problems[id]["prompt"])
        s = [x for x in problems[id]["prompt"].split("\n") if "def" in x]
        print(s[0])
        start_codes.append(s[0])
        print("===================================================")
        print("canonical_solution:\n")
        print(problems[id]["canonical_solution"])
        print("===================================================")
        print("entry_point:\n")
        entry_point = problems[id]["entry_point"]
        print(problems[id]["entry_point"])
        print("===================================================")
        print("test:\n")
        test = problems[id]["test"].replace("candidate",entry_point).split("\n")
        test = [x.strip() for x in test if (entry_point in x and "assert" in x)]
        if "HumanEval/32" in id:
            test = ["assert find_zero([1, 2])==-0.5"]
        elif id == "HumanEval/1":
            test = ["assert candidate('(()()) ((())) () ((())()())') == ['(()())', '((()))', '()', '((())()())']"]
        print(test[0])
        p_test.append(test[0])
        print("===================================================")
    preflex = "# Write Python function to complete the task and pass the assertion tests.\n" + "### Task Start ###\n# These are the assertions for your function:\n" + p_test[0] + "\n\n#Complete the Python funtion:\n" + problems["HumanEval/0"]["prompt"] \
+ "\n### result ###\n" +  start_codes[0] + "\n" + problems["HumanEval/0"]["canonical_solution"] + "### Task End ###\n\n"+ "### Task Start ###\n# These are the assertions for your function:\n" + p_test[1] + "\n\n#Complete the Python funtion:\n" + problems["HumanEval/1"]["prompt"] \
+ "\n### result ###\n" +  start_codes[1] + "\n" + problems["HumanEval/1"]["canonical_solution"] + "### Task End ###\n\n"+ "### Task Start ###\n# These are the assertions for your function:\n" + p_test[2] + "\n\n#Complete the Python funtion:\n" + problems["HumanEval/2"]["prompt"] \
+ "\n### result ###\n" +  start_codes[2] + "\n" + problems["HumanEval/2"]["canonical_solution"] + "### Task End ###\n\n"
    print(preflex)

def get_unit_test():
    #遍历所有task，获取其check程序中的test的io，并写入文件
    problems = read_problems()
    taskids = list(problems.keys())
    num_task = len(taskids)
    p_test = []
    start_codes = []
    print("task num: ",num_task )
    uts = []
    c = 0
    for id in taskids:
        print("task id:",id)
        print("===================================================")
        entry_point = problems[id]["entry_point"]
        test = problems[id]["test"].replace("candidate",entry_point).split("\n")
        if id=="HumanEval/72":
            test = [t.replace("is","==") for t in test]
        test = [x.strip() for x in test if (entry_point in x and "assert" in x)]
        testd = []
        for t in test:
            idx = t.rfind(",")
            idx2 = t[idx:].find("]")
            idx3 = t[idx:].find(")")
            if idx!=-1 and idx2== -1 and idx3==-1 and "==" not in t[idx:]:
                t = t[:idx]
            testd.append(t)
        print(problems[id]["test"])
        print("-------------------------------------------------")
        print(testd)
        print("===================================================")
        ios = []
        
        for t in testd:
            if "==" not in t:
                # print("test not in == format,id:",id)
                c+= 1
                if id=="HumanEval/2" or id=="HumanEval/4":
                    if "<" in t:
                        io = [x.strip() for x in t.split("<")]
                        ios.append({"in":io[0].replace("assert ",""),"out":io[1],"op":"<"})
                        continue
                elif id =="HumanEval/52" or id=="HumanEval/56" or id =="HumanEval/61":
                    testin = t.strip().replace("assert ","")
                    testout = "True"
                    if "assert not" in t:
                        testout = "False"
                    ios.append({"in":testin,"out":testout,"op":"=="})
                    continue
                elif id=="HumanEval/87":
                    ios.append({"in":"get_row([[1,2,3,4,5,6],[1,2,3,4,1,6],[1,2,3,4,5,1]],1)","out":"[(0, 0), (1, 4), (1, 0), (2, 5), (2, 0)]","op":"=="})
                    ios.append({"in":"get_row([[1,2,3,4,5,6],[1,2,3,4,5,6],[1,2,3,4,5,6],[1,2,3,4,5,6],[1,2,3,4,5,6],[1,2,3,4,5,6]],2)","out":"[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1),(5, 1)]","op":"=="})
                    ios.append({"in":"get_row([[1,2,3,4,5,6],[1,2,3,4,5,6],[1,1,3,4,5,6],[1,2,1,4,5,6],[1,2,3,1,5,6],[1,2,3,4,1,6],[1,2,3,4,5,1]],1)","out":"[(0, 0), (1, 0), (2, 1), (2, 0), (3, 2), (3, 0), (4, 3), (4, 0), (5, 4), (5, 0), (6, 5), (6, 0)]","op":"=="})
                    ios.append({"in":"get_row([],1)","out":"[]","op":"=="})
                    ios.append({"in":"get_row([[1]],2)","out":"[]","op":"=="})
                    ios.append({"in":"get_row([[],[1],[1,2,3]],3)","out":"[(2, 2)]","op":"=="})
                    break
                # elif id=="HumanEval/72":
                #     t.replace("is","==")
                #     print(t)
                #     print(problems[id]["test"])
                #     print("-------------------------------------------------")
                #     break
            io = [x.strip() for x in t.split("==")]
            # print("io",io)
            ios.append({"in":io[0].replace("assert ",""),"out":io[1],"op":"=="})
        if id=="HumanEval/32":
            ios = []
            ios.append({"in":"round(find_zero([1, 2]), 2)","out":"-0.5","op":"=="})
            ios.append({"in":"round(find_zero([1, -2]), 2)","out":"0.5","op":"=="})
            ios.append({"in":"round(find_zero([-6, 11, -6, 1]), 2)","out":"1.0","op":"=="})
            ios.append({"in":"round(find_zero([1, 2, 1]), 2)","out":"-1.0","op":"=="})
        if id=="HumanEval/1":
            ios = []
            ios.append({"in":"separate_paren_groups('(()()) ((())) () ((())()())')","out":"['(()())', '((()))', '()', '((())()())']","op":"=="})
            ios.append({"in":"separate_paren_groups('() (()) ((())) (((())))')","out":"['()', '(())', '((()))', '(((())))']","op":"=="})
            ios.append({"in":"separate_paren_groups('(()(())((())))'","out":"['(()(())((())))']","op":"=="})
            ios.append({"in":"separate_paren_groups('( ) (( )) (( )( ))')","out":"['()', '(())', '(()())']","op":"=="})
        if id=="HumanEval/113":
            ios = []
            ios.append({'in': "odd_count(['1234567'])", 'out': '["the number of odd elements 4n the str4ng 4 of the 4nput."]', 'op': '=='})
            ios.append({'in': 'odd_count(["3","11111111"])', 'out': '["the number of odd elements 1n the str1ng 1 of the 1nput.", "the number of odd elements 8n the str8ng 8 of the 8nput."]', 'op': '=='})
            ios.append({'in': "odd_count(['271', '137', '314'])", 'out': "['the number of odd elements 2n the str2ng 2 of the 2nput.','the number of odd elements 3n the str3ng 3 of the 3nput.','the number of odd elements 2n the str2ng 2 of the 2nput.']", 'op': '=='})
        print(ios)
        if ios==[]:
            print(id)
        uts.append({"task_id":id,"unit_tests":ios})
        with open("unitTest.jsonl","w+") as f:
            for ut in uts:
                f.write(json.dumps(ut) + "\n")
    print("c",c)


def handle_prompt(prompt):
    s = prompt.split("\n")
    test = ""
    for i,line in enumerate(s):
        if ">>>" in line:
            test += line[line.index(">>>")+4] + " == "
            test += s[i+1].strip(" ")
            return test
    return "no test in prompt"




def data_detail():
    #打印每个task的所有数据组成
    problems = read_problems()
    taskids = list(problems.keys())
    num_task = len(taskids)
    print("task num: ",num_task )
    for id in taskids:
        print(f"+++++++++++++++++++++++{id}++++++++++++++++++++++++++++")
        print("task keys : ",problems[id].keys())
        print("---------------------------------------------------")
        prompt = ""
        for line in problems[id]["prompt"].split("\n"):
            if ">>>" in line or "Example" in line or "For example" in line:
                prompt += "    \"\"\"\n"
                break
            prompt += line + "\n"
        print("prompt:",problems[id]["prompt"])
        print("new prompt:\n",prompt)
        s = [x for x in problems[id]["prompt"].split("\n") if "def" in x]
        print("start code:",s[0])
        # test_in_prompt = handle_prompt(problems[id]["prompt"])
        # print(test_in_prompt)
        print("---------------------------------------------------")
        print("canonical_solution:\n")
        print(problems[id]["canonical_solution"])
        print("---------------------------------------------------")
        print("entry_point:",problems[id]["entry_point"])
        entry_point = problems[id]["entry_point"]
        print("---------------------------------------------------")
        test = problems[id]["test"].replace("candidate",entry_point)
        print("test:",test)

def extract_test():
    problems = read_problems()
    taskids = list(problems.keys())
    num_task = len(taskids)
    print("task num: ",num_task )
    c1=c2=c3=c4=c5=0
    for id in taskids:
        prompt = problems[id]["prompt"]
        entry_point = problems[id]["entry_point"]
        if ">>>" in prompt:
            # print("+++++++++++++++++")
            # print(prompt)
            # print("+++++++++++++++++")
            c1 +=1
            # lines = prompt.split("\n")
            # print("[TEST]")
            # print("task_id:",id)
            # for i,line in enumerate(lines):
            #     if ">>>" in line:
            #         if "==" in line:
            #             tin = line[line.index(entry_point):].split("==")[0]
            #             tout = line[line.index(entry_point):].split("==")[1]
            #         else:
            #             tin = line[line.index(entry_point):]
            #             tout = lines[i+1].replace("    ","")
            #         print(f"{tin}=={tout}")
            # print("[/TEST]")
        elif "Examples" in prompt:
            c2 +=1
            # print("+++++++++++++++++")
            # print(prompt)
            # print("+++++++++++++++++")
            # seq = prompt[prompt.index("Examples"):].replace("    ","")
            # print("[TEST]")
            # print("task_id:",id)
            # print(seq)
            # print("[/TEST]")
        elif "Example" in prompt:
            c3 +=1
            # print("+++++++++++++++++")
            # print(prompt)
            # print("+++++++++++++++++")
            # seq = prompt[prompt.index("Example"):].replace("    ","")
            # print("[TEST]")
            # print("task_id:",id)
            # print(seq)
            # print("[/TEST]")
        elif "For example" in prompt:
            c4 +=1
            # print("+++++++++++++++++")
            # print(prompt)
            # print("+++++++++++++++++")
            # seq = prompt[prompt.index("For example"):].replace("    ","")
            # print("[TEST]")
            # print("task_id:",id)
            # print(seq)
            # print("[/TEST]")
        else:
            print("+++++++++++++++++")
            print(prompt)
            print("+++++++++++++++++")
            print("[TEST]")
            print("task_id:",id)
            print(prompt)
            print("[/TEST]")
            c5 +=1
            
    print(c1,c2,c3,c4,c5)

def extract_tests_from_file(filename, output_file):
    of = open(output_file,"w+")
    with open(filename,"r") as f:
        flag=False
        tid=-1
        tios = []
        for line in f.readlines():
            if not flag:
                if "[TEST]" in line:
                    flag=True
                    tios=[]
            else:
                if "[/TEST]" in line:
                    of.write(json.dumps({"tid":tid,"ios":tios})+"\n")
                    flag=False
                elif "task_id" in line:
                    tid = line.split(":")[1].strip()
                else:
                    print(line)
                    tin = line.split("==")[0]
                    tout = line.split("==")[1].strip("\n")
                    tios.append({"tin":tin,"tout":tout})
    of.close()
    return True

def read_tests_from_file(filename):
    with open(filename,"r+") as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            task_id = data["task_id"]
            unit_tests = data["unit_tests"]
            print(task_id)
            for ut in unit_tests:
                s = ut["in"] + ut["op"] + ut["out"]
                print(s)
            print("===============================")
    return

if __name__=="__main__":
    # main()
    # data_detail()
    # prompt_file = "prompt.txt"
    # with open(prompt_file,"r") as f:
    #     preflex = f.read()
    #     print(preflex)
    # extract_test()
    # extract_tests_from_file("data_extract.txt","tests_from_prompt.jsonl")
    read_tests_from_file(filename="unitTest.jsonl")
