from typing import Iterable, Dict
import gzip
import json
import os


ROOT = os.path.dirname(os.path.abspath(__file__))
HUMAN_EVAL = os.path.join(ROOT, "..", "data", "HumanEval.jsonl.gz")


def read_problems(evalset_file: str = HUMAN_EVAL) -> Dict[str, Dict]:
    return {task["task_id"]: task for task in stream_jsonl(evalset_file)}


def stream_jsonl(filename: str) -> Iterable[Dict]:
    """
    Parses each jsonl line and yields it as a dictionary
    """
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, 'rt') as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with open(filename, "r") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)


def write_jsonl(filename: str, data: Iterable[Dict], append: bool = False):
    """
    Writes an iterable of dictionaries to jsonl
    """
    if append:
        mode = 'ab'
    else:
        mode = 'wb'
    filename = os.path.expanduser(filename)
    if filename.endswith(".gz"):
        with open(filename, mode) as fp:
            with gzip.GzipFile(fileobj=fp, mode='wb') as gzfp:
                for x in data:
                    gzfp.write((json.dumps(x) + "\n").encode('utf-8'))
    else:
        with open(filename, mode) as fp:
            for x in data:
                fp.write((json.dumps(x) + "\n").encode('utf-8'))


def print_data_detail(data_file):
    problems = read_problems(data_file)
    for tid,task in problems.items():
        print(f"============{tid}============")
        print(f"prompt:\n{task['prompt']}")
        print("-------------------------------")
        print(f"entry_point:\n{task['entry_point']}")
        print("-------------------------------")
        print(f"canonical_solution:\n{task['canonical_solution']}")
        print("-------------------------------")
        print(f"test:\n{task['test']}")
    return problems

def improve_mbpp_data(data_file,res_file):
    problems = read_problems(data_file)
    new_problems = []
    for tid,task in problems.items():
        task["prompt"] = task["prompt"] + "\n"
        new_problems.append(task)
        
    for task in new_problems:
        tid = task["task_id"]
        assert task["test"] == problems[tid]["test"]
        assert task["canonical_solution"] == problems[tid]["canonical_solution"]
        assert task["entry_point"] == problems[tid]["entry_point"]
    write_jsonl(res_file,new_problems)
    return problems

def compare_data(data_file1,data_file2):
    problems1 = read_problems(data_file1)
    problems2 = read_problems(data_file2)
    different_task = []
    for tid,task in problems1.items():
        if problems1[tid]["entry_point"] != problems2[tid]["entry_point"]:
            print("===========================")
            print(tid)
            print(problems1[tid]["prompt"])
            print(problems2[tid]["prompt"])
            different_task.append(tid)
    print(different_task)
    return different_task

if __name__ == "__main__":
    # print_data_detail("./mbpp_humaneval_format_11_510.jsonl")
    # improve_mbpp_data("./mbpp_humaneval_format_11_510.jsonl","./mbpp_humaneval_format_11_510_2.jsonl")
    compare_data("mbpp_humaneval_format_11_510.jsonl","mbpp_humaneval_format.jsonl")