import json
import argparse
from collections import defaultdict
import random
import json
import argparse
from human_eval.data import write_jsonl, read_problems
from human_eval.execution import check_test_correctness
from concurrent.futures import ThreadPoolExecutor

random.seed(1024)

def gen_new_check(task_file,check_file,test_num = 5):
    wf = open(check_file,"w+")
    c = defaultdict(int)
    problems = read_problems()
    with open(task_file,"r") as f:
        for line in f.readlines():
            data = json.loads(line)
            for key,value in data.items():
                task_id = key
                tests = value
                print(f"For task {task_id} the tests num is {len(tests)}")
                c[len(tests)] += 1
                # entry_point = tests[0]["in"][:tests[0]["in"].index("(")]
                cir = 0
                while True:
                    check_program = ""#f"def check():"
                    if len(tests) >= test_num:
                        ts = random.choices(tests,k= test_num)
                        print(f"tests:{len(ts)}")
                    else:
                        ts = tests
                        print(f"task {task_id} gen less than {test_num} tests")
                    for t in ts:
                        tout = t["out"]
                        tin = t["in"]
                        entry_point = tin[:tin.index("(")]
                        idx1 = tout.find(", \""+entry_point+"(")
                        if idx1!=-1:
                            tout = tout[:idx1]
                        test = "assert " + t["in"] + " == " + tout# + ",\"Test" + str(check_tests_num) +"\""
                        test = test.replace("assert ssert ","assert ").replace("assert sert ","assert ").replace("assert ert ","assert ").replace("assert rt ","assert ").replace("assert t ","assert ")
                        check_program += test + "\n"
                    check_program = check_program.strip()
                    p = problems[task_id]
                    prompt = p["prompt"] + p["canonical_solution"] + "\n" + check_program + "\n"
                    print("#"*30)
                    print(f"For task {task_id} the check program is\n{prompt}")
                    print("#"*30)
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        args = (prompt, 30.0)
                        future = executor.submit(check_test_correctness, *args)
                        result2 = future.result()#check_one_correctness(problems[id],completion,3.0)
                        passed = result2["passed"]
                        r = result2["result"]
                        # print("passed: ",passed)
                        if not passed:
                            print(f"task {task_id} new_check failed {r}")
                            cir += 1
                            if cir > 20:
                                break
                        else:
                            break
                wf.write(json.dumps({"task_id":task_id,"check_program":check_program})+"\n")
        wf.close()
        c = dict(sorted(c.items(),key=lambda x: x[0]))
        print(c)
        print("Finish!")
    return True

def check_new():
    check_file = "new_check.jsonl"
    problems = read_problems()
    checks = {}
    with open(check_file,"r") as f:
        for line in f.readlines():
            data = json.loads(line)
            checks[data["task_id"]] = data["check_program"]
    tasks = list(problems.keys())
    for tid in tasks:
        p = problems[tid]
        check_program = p["prompt"] + p["canonical_solution"] + "\n" + checks[tid] + "\n"
        with ThreadPoolExecutor(max_workers=1) as executor:
            args = (check_program, 30.0)
            future = executor.submit(check_test_correctness, *args)
            result2 = future.result()#check_one_correctness(problems[id],completion,3.0)
            passed = result2["passed"]
            r = result2["result"]
            # print("passed: ",passed)
            if not passed:
                print("="*30)
                print(check_program)
                print("="*30)
                print(f"task {tid} new_check failed {r}")
    return True

if __name__=="__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-tf","--task_file",default="gen_tests.jsonl",required=True)
    # parser.add_argument("-of","--output_file",default="check.jsonl",required=True)
    # args = parser.parse_args()

    # task_file = args.task_file
    # check_file = args.output_file

    # gen_new_check(task_file,check_file,test_num=5)
    check_new()