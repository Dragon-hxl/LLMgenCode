import json
from src.resfiles_record3 import *

def load_time_record(res_file):
    time_record = {}
    with open(res_file, 'r') as f:
        res = f.readlines()
        for line in res:
            data = json.loads(line)
            tid = data["task_id"]
            time_record[tid] = data["time_record"]
    return time_record

def time_analysis(time_record):
    choose_solution_time = []
    model_inference_time = []
    run_solutions_time = []
    
    for tid, records in time_record.items():
        for record in records:
            choose_solution_time.append(record["choose_solution_time"])
            model_inference_time.append(record["model_inference_time"])
            run_solutions_time.append(record["run_solutions_time"])
            
    choose_solution_time_total = sum(choose_solution_time)
    model_inference_time_total = sum(model_inference_time)
    run_solutions_time_total = sum(run_solutions_time)
    total_time = choose_solution_time_total + model_inference_time_total + run_solutions_time_total
    
    choose_solution_time_avg = choose_solution_time_total / len(choose_solution_time)
    model_inference_time_avg = model_inference_time_total / len(model_inference_time)
    run_solutions_time_avg = run_solutions_time_total / len(run_solutions_time)
    
    print("Choose solution time: total: {:.6f}mins, avg: {:.6f}mins".format(choose_solution_time_total, choose_solution_time_avg))
    print("Model inference time: total: {:.6f}mins, avg: {:.6f}mins".format(model_inference_time_total, model_inference_time_avg))
    print("Run solutions time: total: {:.6f}mins, avg: {:.6f}mins".format(run_solutions_time_total, run_solutions_time_avg))
    
    print("Total time: {:.2f}mins".format(total_time))
    print("time percentage")
    print("Choose solution time: {:.6f}%".format(choose_solution_time_total / total_time * 100))
    print("Model inference time: {:.6f}%".format(model_inference_time_total / total_time * 100))
    print("Run solutions time: {:.6f}%".format(run_solutions_time_total / total_time * 100))
    
    return choose_solution_time, model_inference_time, run_solutions_time

if __name__ == "__main__":
    res_file = res_root + res_cola7bpy[5]
    time_record = load_time_record(res_file)
    choose_solution_time, model_inference_time, run_solutions_time = time_analysis(time_record)
        