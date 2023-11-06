import argparse

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f1","--file1",required=True,help="path of file 1 to compare")
    parser.add_argument("-f2","--file2",required=True,help="path of file 2 to compare")
    args = parser.parse_args()
    f1 = args.file1
    f2 = args.file2

    p1 = set()
    p2 = set()
    with open(f1,"r") as f:
        for line in f.readlines():
            if "Result for" in line:
                for s in line.split(" "):
                    if "HumanEval" in s:
                        tid = s.split("/")[1]
                        if "passed" in line:
                            p1.add(tid)

    with open(f2,"r") as f:
        for line in f.readlines():
            if "Result for" in line:
                for s in line.split(" "):
                    if "HumanEval" in s:
                        tid = s.split("/")[1]
                        if "passed" in line:
                            p2.add(tid)
    print(p1-p2)
    print(p2-p1)
    print("finish!")
