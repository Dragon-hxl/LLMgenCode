import json 

def read_jsonl(file_name):
    with open(file_name, 'r') as f:
        return [json.loads(l) for l in f]
    return None

def count_testcases(tasks):
    testcases = []
    for t in tasks:
        