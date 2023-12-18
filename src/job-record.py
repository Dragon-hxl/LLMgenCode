import time
import fire

job_record_file = "job-record.txt"
def main(
    jobID,
    comment:str="",
    output_file: str="",
):
    job_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) 
    with open(job_record_file,"a+") as f:
        s = f"ID : {jobID} , time : {job_time} , job comment : {comment} , output file : {output_file}\n"
        f.write(s)
    return

if __name__=="__main__":
    fire.Fire(main)