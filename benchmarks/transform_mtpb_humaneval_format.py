import json

def read_jsonl(file_name):
    with open(file_name, 'r') as f:
        return [json.loads(l) for l in f]
    return None
    
def get_func_name(name):
    func_name = "_".join(name.split(" "))
    return func_name

def get_args(input):
    args = list(input.keys())
    print(args)
    return args

def get_func_sig(func_name, args):
    func_sig = "def " + func_name + "("
    n = len(args)
    for i,arg in enumerate(args):
        if i == n - 1:
            func_sig += arg + "):"
        else:
            func_sig += arg + ", "
    return func_sig

def get_tests(func_name, inputs, outputs):
    test = f"def check({func_name}):\n"
    tins = []
    for tinput in inputs:
        print(f"tinput:{tinput}")
        tin = []
        for key,value in tinput.items():
            tin.append(json.dumps(value))
        print(tin)
        tin = ",".join(tin)
        tins.append(tin)
    for tin,output in zip(tins, outputs):
        test += f"    assert {func_name}({tin}) == {output}\n"
    return test

def tranform_to_humaneval(input_file, output_file):
    data = read_jsonl(input_file)
    formatdata = []
    for task in data:
        problem = {}
        task_id = "mtpb/" + str(task["id"])
        args = get_args(task["inputs"][0])
        func_name = get_func_name(task["name"])
        func_sig = get_func_sig(func_name, args)
        tests = get_tests(func_name, task["inputs"], task["outputs"])
        prompt = func_sig + "\n    \"\"\"\n    " + "\n    ".join(task["prompts"]) + "\n    \"\"\""
        
        problem["task_id"] = task_id
        problem["entry_point"] = func_name
        problem["prompt"] = prompt
        problem["test"] = tests
        problem["func_sig"] = func_sig
        
        print("+---------------------------------+")
        print(f"task_id: {task_id}")
        print(f"entry_point: {func_name}")
        print(f"prompt:\n{prompt}")
        print(f"test:\n{tests}")
        print(f"func_sig: {func_sig}")
        print("+---------------------------------+")
        
        formatdata.append(problem)
    with open(output_file, 'w') as f:
        for problem in formatdata:
            f.write(json.dumps(problem) + "\n")
    return formatdata

if __name__ == "__main__":
    input_file = "mtpb.jsonl"
    output_file = "mtpb_humaneval_format.jsonl"
    tranform_to_humaneval(input_file, output_file)
        
        
        