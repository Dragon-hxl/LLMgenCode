import json
from collections import Counter
# from bigbench.benchmark_tasks.python_programming.challenge import get_challenges
# import bigbench

def challenge_string(challenge):
    return (
        f"```"
        f"\nName: {challenge.name}"
        f"\nDifficulty: {challenge.difficulty}"
        f"\n\nExample outputs:\n"
        + "\n".join(
            [f"{args} => {result}" for args, result in challenge.test_generator()]
        )
        + f"\n\nPrompt:\n{challenge.prompt}"
        f"\n```"
    )

def get_func_name(name):
    func_name = "_".join(name.split(" "))
    return func_name

def get_func_sig(prompt):
    lines = prompt.split("\n")
    func_sig = ""
    for line in lines:
        if "def" in line:
            func_sig = line
            break
    return func_sig


def get_entry_point(func_sig):
    entry_point = func_sig.replace("def", "").split("(")[0].strip()
    return entry_point

def get_tests(func_name, ios):
    test = f"def check({func_name}):\n"
    for io in ios:
        tin = io[0]
        tout = io[1]
        print(f"tin: {tin}, tout: {tout}")
        if tin != "" and tin[0]=="(" and tin[-1]==")":
            tin = tin[1:-1]
        test += f"    assert {func_name}({tin}) == {tout}\n"
    return test

def load_bigbench_from_txt(path):
    name_str = 'Name:'
    difficulty_str = 'Difficulty:'
    example_str = '=>'#'Example outputs:'
    prompt_str = 'Prompt:'
    problems = []
    example_start = False
    tests_str = ""
    task = {}
    with open(path, 'r') as f:
        lines = f.readlines()
        n = len(lines)
        print(f"n: {n}")
        i = 0
        while i < n:
            if name_str in lines[i]:
                name = lines[i][len(name_str):].strip()
                if task:
                    problems.append(task)
                task = {}
                task_id = len(problems)
                task['task_id'] = f"bigbench/{task_id}"
            elif difficulty_str in lines[i]:
                difficulty = lines[i][len(difficulty_str):].strip()
                task['difficulty'] = difficulty
            elif example_str in lines[i]:
                tests_str = lines[i]
                while i+1<n and (example_str not in lines[i+1]) and (lines[i+1]!="\n"):
                    i += 1
                    tests_str += lines[i]
                tin = tests_str.split("=>")[0].strip()
                tout = tests_str.split("=>")[1].strip()
                # if "\n    " in tout:
                #     print(task_id)
                #     tout = tout.replace("\n    ", "\n")
                task["test"] = task.get("test", []) + [(tin, tout)]
            elif prompt_str in lines[i]:
                prompt = ""
                while i+1 < n and (name_str not in lines[i+1]):
                    i += 1
                    prompt += lines[i]
                task['prompt'] = prompt
                func_sig = get_func_sig(prompt)
                task['func_sig'] = func_sig
                entry_point = get_entry_point(func_sig)
                task['entry_point'] = entry_point
            i += 1
    problems.append(task)
    for task in problems:
        print("+-------------------+")
        print(f"Task {task['task_id']}: {task['entry_point']}")
        print(f"Difficulty: {task['difficulty']}")
        print(f"Prompt:\n{task['prompt']}")
        print(f"Tests: {task['test']}")
        test = get_tests(task['entry_point'], task['test'])
        print(f"Test:\n{test}")
        task['test'] = test
    with open("bigbench_humaneval_format.jsonl","w+") as f:
        for task in problems:
            f.write(json.dumps(task) + "\n")
    return problems
                    
if __name__=="__main__":
    load_bigbench_from_txt("bigbench_detial_32.txt")
    # for co in challenges:
    #     print(co)