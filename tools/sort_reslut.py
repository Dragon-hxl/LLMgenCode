import json
import argparse
from collections import defaultdict,Counter
import re

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f","--file",default="result.txt",required=True)
    args = parser.parse_args()

    result_file = args.file
    output_file = result_file[:-4]+ "_sorted.txt"
    r = {}
    with open(result_file,"r") as f:
        lines = f.readlines()
        for line in lines:
            idx = line.find("Result for")
            if idx!=-1:
                line = line[idx:]
                for s in line.split(" "):
                    if "HumanEval" in s:
                        tid = s.split("/")[1]
                        r[tid] = line
            else:
                continue
    sorted_r = sorted(r.items(),key=lambda x:int(x[0]))
    r = dict(sorted_r)
    with open(output_file,"w+") as f:
        for it in sorted_r:
            f.write(it[1]+"\n")
    print("sorted result fininsh!")
    return r

def dict_counter(d):
    counter = defaultdict(int)
    for key,value in d.items():
        counter[value] += 1
    return counter

def res_classification(r):
    errlist = defaultdict(list)
    failed = set()
    wrong_code = set()
    name_err = set()
    passed = set()
    for key,v in r.items():
        result = v[v.index(":")+2:].strip()
        reg = "\(.+\)"
        line_num = re.search(reg,result)
        if line_num:
            result = result[:result.index(line_num.group())]
        errlist[result].append(key)
        if "passed" in result:
            passed.add(key)
        if "failed" in result:
            failed.add(key)
        if "EOF" in result or "invalid syntax" in result or "unexpected character" in result or "EOL" in result or "indent" in result or "unmatched" in result:
            wrong_code.add(key)
        if "name" in result:
            name_err.add(key)
    errlist = dict(sorted(errlist.items(),key=lambda x: x[0]))
    assert_error = 0
    for key,v in errlist.items():
        print(f"{key} ( appears {len(v)} times )\n")
        if "Assertion" in key:
            assert_error = len(v)
        print(v)
    print("falied nums :",len(failed))
    print("assert error :", assert_error)
    print("other=failed-assert_error :",len(failed)-assert_error)
    print("wrong code nums:",len(wrong_code))
    print("name_err:",len(name_err))
    print(f"acc point: {round(float(len(passed))/len(r.keys()),4)}")
    return failed,wrong_code,name_err

if __name__=="__main__":
    r = main()
    res_classification(r)