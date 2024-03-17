# transform the mbpp dataformat to humaneval dataformat
import json
import re

# 寻找函数名
def find_function_name(code):
    lines = code.split('\n')
    for line in lines:
        if line.strip().startswith("def "):
            return line.split()[1].split('(')[0]
    return None

def find_function_name2(test):
    tin = test.replace("assert","").split("==")[0].strip()
    idx= tin.find("(")
    return tin[:idx].strip()

# 寻找函数签名
def find_function_signature(code,function_name):
    entry_point = f"def {function_name}("
    entry_point2 = f"def {function_name} ("
    lines = code.split('\n')
    for line in lines:
        if line.strip().startswith(entry_point):
            return line.strip()
        if line.strip().startswith(entry_point2):
            return line.strip()
    return None

# 寻找函数签名及其前面的代码
def find_function_and_code(code,function_name):
    entry_point = f"def {function_name}("
    entry_point2 = f"def {function_name} ("
    lines = code.split('\n')
    import_lines = ""
    for i, line in enumerate(lines):
        if "import" in line:
            import_lines += line + "\n"
        if line.strip().startswith(entry_point):
            return import_lines
        if line.strip().startswith(entry_point2):
            return import_lines
    return None, None




def get_test(test_list,function_name):
    test = f"def check({function_name}):\n"
    for i in range(len(test_list)):
        test += "    " + test_list[i] + "\n"
    return test

def main():
    # 读取jsonl文件到列表
    mbpp_data = []
    with open("mbpp.jsonl", "r") as fp:
        for line in fp:
            if any(not x.isspace() for x in line):
                mbpp_data.append(json.loads(line))

    mbpp_humaneval_format = []

    for i in range(len(mbpp_data)):
        num_tid = int(mbpp_data[i]["task_id"])
        if num_tid < 11 or num_tid > 511:
            continue
        line = {}
        line["task_id"] = f"MBPP/{num_tid}"
        test_list = mbpp_data[i]["test_list"]
        test = test_list[0]
        # function_name = find_function_name(mbpp_data[i]["code"])
        function_name2 = find_function_name2(test)
        # if function_name != function_name2:
        #     print(f"{line['task_id']}")
        #     print(f"function_name : {function_name}, function_name2 : {function_name2}")
        function_signature = find_function_signature(mbpp_data[i]["code"],function_name2)
        code_before_function = find_function_and_code(mbpp_data[i]["code"],function_name2)
        prompt = "# {}".format(mbpp_data[i]["text"]) + "\n"  +code_before_function+ function_signature + "\n"
        line["prompt"] = prompt
        line["entry_point"] = function_name2
        line["canonical_solution"] = mbpp_data[i]["code"]
        
        line["test"] = get_test(test_list,function_name2)
        print("+++++++++++++++++++++++++++++++++++++++++++++")
        print(line["task_id"])
        print(line["prompt"])
        print(line["entry_point"])
        print(line["canonical_solution"])
        print(line["test"])
        mbpp_humaneval_format.append(line)


    #输出到jsonl文件
    with open("mbpp_humaneval_format.jsonl", "w+") as fp:
        for line in mbpp_humaneval_format:
            fp.write(json.dumps(line) + '\n')

    print("done")




if __name__ == '__main__':

    main()