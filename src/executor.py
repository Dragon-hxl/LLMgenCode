from typing import Optional, Callable, Dict, List
import ast
import contextlib
import faulthandler
import io
import os
import multiprocessing
import platform
import signal
import tempfile
import time
from multiprocessing import Process, Manager
from concurrent.futures import ThreadPoolExecutor
from myutils import get_UTfeedback_prompt_v1
from executor_utils import run_code_with_test_result

class py_executor:
    def __init__(self):
        pass
    
    def excute(self,solution: str, tests: List[str], feedback_prompt,timeout: float=0.1,feedback_type="UT"):
        run_test = [t.replace("assert ","").split("==")[0].strip() for t in tests]
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                args = ( solution, run_test, tests, timeout)
                future = executor.submit(run_code_with_test_result, *args)
                result = future.result()
                passed = result["passed"]
                final_res = result["result"]
            feedback,passn,pass_tests = get_UTfeedback_prompt_v1(feedback_prompt, solution, passed, final_res, run_test, tests,feedback_type)
        except BaseException as e:
            print(f"run solution failed with {e}")
            
        return feedback,passn,pass_tests
    
    