from myutils import map_gpu_memory,filter_fix_ans,print_with_tag,make_printv,setup_seed
import time
import numpy as np
import torch
import ast

# prompt file 
prompt_root = "/home/S/hexiaolong/codex/self-debug/data/prompt/"
prompt_file = prompt_root + "prompt_base2.txt"
UTfeedback_file = prompt_root + "prompt_UTfeedback.txt"
mbpp_base_prompt = prompt_root + "prompt_base_mbpp.txt"


#读取prompt
with open(prompt_file,"r") as f:
    preflex = f.read()
with open(UTfeedback_file,"r") as f:
    UTfeedback_promt = f.read()

class PyGenerator():
        
    def code_extract(self,solution):
        # 截取程序
        idx2 = solution.find("### Task End ###")
        if idx2 != -1:
            solution = solution[:idx2-1] #这里的减1是为了去掉前面的换行
        if len(solution.split("```"))>1:
            solution = solution.split("```")[1]
        else:
            print(solution.split("```"))
        if solution.startswith("python"):
            solution = solution[6:]
        solution = solution.strip("\n")
        return solution
    
    # 构成生成初始代码的prompt
    def get_one_complication(self,tprompt,unit_test):#为模型的输入添加前缀串，看能否提高准确度。目前直接返回原串。
        res = preflex + unit_test + "\n\n# Complete the Python funtion:\n" + tprompt + "### result ###\n"
        return res
    
    def generate_base_complication(self, model, prompt, unit_tests, record_time = False, verbose = False):
        setup_seed(2024)
        #prepare the prompt
        prompt = self.get_one_complication(prompt,unit_tests)
        # print_with_tag("base complication prompt",prompt,verbose=verbose)
        
        input_len = len(prompt)
        inputs = model.tokenizer(prompt, return_tensors='pt', return_token_type_ids=False)
        input_tokens_len = inputs.input_ids.shape[1]
        inputs = inputs.to('cuda')
        
        #generate the solution
        st = time.time()
        pred = model.model.generate(**inputs, max_new_tokens=512, temperature=0,pad_token_id=model.tokenizer.eos_token_id)#,temperature=0.4,repetition_penalty=1.1
        model_inference_time = (time.time()-st)/60
        output_tokens_len = pred.shape[1]
        ans = model.tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)[input_len:].strip("\n")
        print_with_tag("base complication origin output",ans,verbose=verbose)
        solution = self.code_extract(ans)
        
        #log the time and length
        if verbose:
            print(f"Model inference time is {model_inference_time} minutes")
            print(f"In generate step, the input tokens shape is {input_tokens_len}, the output tokens shape is {output_tokens_len}")
        
        #return the solution
        if record_time:
            return prompt,solution, model_inference_time, input_tokens_len, output_tokens_len
        else:
            return prompt,solution
        
    def generate_with_feedback(self, model, feedabck_prompt, return_sequences:int=10 ,record_length = False, record_time = False, verbose = False):
        setup_seed(2024)
        printv = make_printv(verbose)
        debug_maxLen = 512
        debug_temp = 1.0
        debug_top_p = 0.95
        
        solutions = []
        
        inputs = model.tokenizer(feedabck_prompt, return_tensors='pt', return_token_type_ids=False)
        input_length = inputs.input_ids.shape[1]
        printv(f"total input length is {inputs.input_ids.shape}")
        inputs = inputs.to('cuda')
        try:
            with torch.inference_mode():
                preds = model.model.generate(**inputs, max_new_tokens=debug_maxLen, temperature=debug_temp,top_p=debug_top_p, do_sample=True,return_dict_in_generate=True,output_scores=True,num_return_sequences=return_sequences,pad_token_id=model.tokenizer.eos_token_id)#,temperature=0.4,repetition_penalty=1.1
                transition_scores = model.model.compute_transition_scores(
                    preds.sequences, preds.scores, normalize_logits=True
                ).cpu().numpy()
            for pred,transition_score in zip(preds["sequences"],transition_scores):
                # 计算生成概率
                gen_tokens = pred[input_length:].cpu()
                valid = np.isfinite(transition_score)
                tsc = transition_score[valid]
                output_length = input_length + tsc.shape[-1]
                sc = np.sum(tsc,axis=-1)/output_length
                true_sc = np.exp(sc)
                
                #记录生成的solution
                ans = model.tokenizer.decode(gen_tokens, skip_special_tokens=True)
                # printv(f"ans is \n{ans}")
                
                solutions.append((ans,true_sc,output_length))
        except BaseException as e:
            printv(f"model inference failed with {e}")
        except:
            printv("model inference failed!")
        return solutions
    
    
    def gen_tests(self,model,problem,num,verbose = False):
        setup_seed(2024)
        printv = make_printv(verbose)
        gentests_fewshot = "/home/S/hexiaolong/codex/self-debug/src/gen_test_shot_20240313.txt"
        with open(gentests_fewshot,"r") as f:
            gentests_prompt = f.read()
        temperature = 0.8
        top_k = 50
        end_cir = 30
        test_get = set()
        gened_tests = []
        # start gen tests
        while len(gened_tests) < num and end_cir > 0:
            st = time.time()
            prompt = f"{gentests_prompt}\n\nfunc signature:\n{problem['prompt']}\nunit tests:\n"
            inputs = model.tokenizer(prompt, return_tensors='pt', return_token_type_ids=False).to('cuda')
            pred = model.model.generate(**inputs, max_new_tokens=512,top_k=top_k,do_sample=True,temperature=temperature,num_return_sequences=8,pad_token_id=model.tokenizer.eos_token_id,repetition_penalty=1.1)
            interval = time.time() - st
            printv(f"gen time : {interval}")
            ans = ""
            for p in pred:
                ans += model.tokenizer.decode(p, skip_special_tokens=True) + "\nnext ans :\n"
            entry_point = "assert " + problem["entry_point"] + "("
            for line in ans.split("\n"):
                line = line.replace("\\_","_")
                if entry_point in line and "==" in line and "# assert" not in line:
                    test_in = line.split("==")[0]
                    try:
                        test_in = line.split("==")[0][test_in.index("assert ")+7:].strip()
                    except:
                        continue
                    test_out = line.split("==")[1].strip()
                    if (test_in,test_out) in test_get:
                        continue
                    testcase = f"assert {test_in} == {test_out}"
                    flag = py_is_syntax_valid(testcase)
                    if not flag:
                        printv(f"gen wrong testcase :  {test_in} == {test_out}")
                        continue
                    else:
                        printv(f"gen testcase :  {test_in} == {test_out}")
                    gened_tests.append(testcase)
                    # test_in_set.add(test_in)
                    test_get.add((test_in,test_out))
                    printv("++++++++++++++++++++++++++++++++++++++++")
            if len(gened_tests) > num:
                printv(f"gen tests num: {len(gened_tests)}\n")
                break
            else:
                end_cir -= 1
        if len(gened_tests) < num:
            printv(f"can not gen enough tests, gened tests num: {len(gened_tests)}\n")
        return gened_tests
    
    def gen_tests_sort_by_prob(self,model,problem,num,verbose = False):
        setup_seed(2024)
        printv = make_printv(verbose)
        gentests_fewshot = "/home/S/hexiaolong/codex/self-debug/src/gen_test_shot_20240313.txt"
        with open(gentests_fewshot,"r") as f:
            gentests_prompt = f.read()
        temperature = 1.0
        top_k = 50
        end_cir = 30
        test_get = set()
        gened_tests = []
        # start gen tests
        while len(gened_tests) < num and end_cir > 0:
            st = time.time()
            prompt = f"{gentests_prompt}\n\nfunc signature:\n{problem['prompt']}\nunit tests:\n"
            inputs = model.tokenizer(prompt, return_tensors='pt', return_token_type_ids=False).to('cuda')
            input_length = inputs.input_ids.shape[1]
            # pred = model.model.generate(**inputs, max_new_tokens=512,top_k=top_k,do_sample=True,temperature=temperature,num_return_sequences=8,pad_token_id=model.tokenizer.eos_token_id,repetition_penalty=1.1)
            with torch.inference_mode():
                preds = model.model.generate(**inputs, max_new_tokens=512, temperature=temperature,top_k=top_k, do_sample=True,return_dict_in_generate=True,output_scores=True,num_return_sequences=8,pad_token_id=model.tokenizer.eos_token_id,repetition_penalty=1.1)#,temperature=0.4,repetition_penalty=1.1
                transition_scores = model.model.compute_transition_scores(
                    preds.sequences, preds.scores, normalize_logits=True
                ).cpu().numpy()
            for pred,transition_score in zip(preds["sequences"],transition_scores):
                # 计算生成概率
                gen_tokens = pred[input_length:].cpu()
                valid = np.isfinite(transition_score)
                tsc = transition_score[valid]
                output_length = input_length + tsc.shape[-1]
                sc = np.sum(tsc,axis=-1)/output_length
                true_sc = np.exp(sc)
                
                #记录生成的tests
                ans = model.tokenizer.decode(gen_tokens, skip_special_tokens=True)
                entry_point = "assert " + problem["entry_point"] + "("
                for line in ans.split("\n"):
                    line = line.replace("\\_","_")
                    if entry_point in line and "==" in line and "# assert" not in line:
                        test_in = line.split("==")[0]
                        try:
                            test_in = line.split("==")[0][test_in.index("assert ")+7:].strip()
                        except:
                            continue
                        test_out = line.split("==")[1].strip()
                        if test_in in test_get:
                            continue
                        testcase = f"assert {test_in} == {test_out}"
                        flag = py_is_syntax_valid(testcase)
                        if not flag:
                            continue
                        else:
                            printv(f"gen testcase :  {test_in} == {test_out}")
                        gened_tests.append((testcase,true_sc))
                        # test_in_set.add(test_in)
                        test_get.add(test_in)
                        printv("++++++++++++++++++++++++++++++++++++++++")
            if len(gened_tests) > num:
                printv(f"gen tests num: {len(gened_tests)}\n")
                break
            else:
                end_cir -= 1
        if len(gened_tests) < num:
            printv(f"can not gen enough tests, gened tests num: {len(gened_tests)}\n")
        gened_tests = sorted(gened_tests,key=lambda x: x[1],reverse=True)
        total_tests = [t[0] for t in gened_tests]
        print_num = min(20,len(total_tests))
        for i in range(print_num):
            printv(f"gened test {i} : {total_tests[i]}")
        return total_tests
    
    def generate_base_complication_mbpp(self, model, prompt, unit_tests, record_time = False, verbose = False):
        setup_seed(2024)
        #prepare the prompt
        with open(mbpp_base_prompt,"r") as f:
            preflex = f.read()
        prompt = preflex + prompt + "### result ###\n"
        # prompt = self.get_one_complication(prompt,unit_tests)
        print_with_tag("base complication prompt",prompt,verbose=verbose)
        
        input_len = len(prompt)
        inputs = model.tokenizer(prompt, return_tensors='pt', return_token_type_ids=False)
        input_tokens_len = inputs.input_ids.shape[1]
        inputs = inputs.to('cuda')
        
        #generate the solution
        st = time.time()
        pred = model.model.generate(**inputs, max_new_tokens=512, temperature=0,pad_token_id=model.tokenizer.eos_token_id)#,temperature=0.4,repetition_penalty=1.1
        model_inference_time = (time.time()-st)/60
        output_tokens_len = pred.shape[1]
        ans = model.tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)[input_len:].strip("\n")
        print_with_tag("base complication origin output",ans,verbose=verbose)
        solution = self.code_extract(ans)
        
        #log the time and length
        if verbose:
            print(f"Model inference time is {model_inference_time} minutes")
            print(f"In generate step, the input tokens shape is {input_tokens_len}, the output tokens shape is {output_tokens_len}")
        
        #return the solution
        if record_time:
            return prompt,solution, model_inference_time, input_tokens_len, output_tokens_len
        else:
            return prompt,solution
                
def py_is_syntax_valid(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except Exception:
        return False