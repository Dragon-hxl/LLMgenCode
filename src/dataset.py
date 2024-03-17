from typing import Iterable, Dict
import gzip
import json
import os

class taskBase:
    def __init__(self, task_id:str="", prompt:str="", entry_point:str="", canonical_solution:str="", prompt_tests:list=[], test:"str"=""):
        self.task_id = task_id
        self.prompt = prompt
        self.entry_point = entry_point
        self.canonical_solution = canonical_solution
        self.prompt_tests = prompt_tests
        self.test = test
    
    @classmethod
    def trans_from_HumanEval(cls, problem):
        return cls(
            task_id=problem.get('task_id', ''),
            prompt=problem.get('prompt', ''),
            entry_point=problem.get('entry_point', ''),
            canonical_solution=problem.get('canonical_solution', ''),
            prompt_tests=problem.get('prompt_tests', []),
            test=problem.get('test', '')
        )
        
    @classmethod
    def trans_from_MBPP(cls, problem):
        return cls(
            task_id=problem.get('task_id', ''),
            prompt=problem.get('prompt', ''),
            entry_point=problem.get('entry_point', ''),
            canonical_solution=problem.get('canonical_solution', ''),
            prompt_tests=problem.get('prompt_tests', []),
            test=problem.get('test', '')
        )
        

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
