import json

class Node:
    def __init__(self,solution:str,parent=None,prompt:str="",passT_rate:float=-1.0,prob:float=-1.0,depth:int=0,feedbackprompt:str="") -> None:
        self.solution = solution
        self.parent = parent
        self.children = []
        self.prompt = prompt
        self.passT_rate = passT_rate
        self.prob = prob
        self.depth = depth
        self.reflect = ""
        self.test_feedback = ""
        self.feedbackprompt = feedbackprompt #由这个node的solution产生的feedbackprompt
        self.CODET_point = 0.0
        self.CODET_pass_testcase = set()
        self.already_CODET = False
        self.idx = 0
    def __repr__(self) -> str:
        return f"solution:\n{self.solution}\npassT_rate:{self.passT_rate}\nprob:{self.prob}\nCODET_point:{self.CODET_point}"
    
    def __eq__(self, other: object) -> bool:
        return self.solution == self.solution
    
    def __hash__(self) -> int:
        return hash(self.solution)
    
    def show_parents(self) -> None:
        print("++++++show parents of the node++++++")
        print(self.__repr__())
        print("************************")
        if self.parent!=None:
            self.parent.show_parents()
        # while node!=None:
        #     print(node)
        #     node = node.parent
        #     print("************************")
        return None
            
    

        