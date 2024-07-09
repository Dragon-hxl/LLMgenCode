# 毕业论文-基于大语言模型的代码生成自反馈技术优化-源码-贺小龙 20240701

### 文件说明
#### 主程序
main.py 为主程序的文件，通过接收命令行参数，准备数据并调用相应的自反馈策略进行代码生成

#### 模块文件
dasetset.py 加载代码生成任务，根据命令行指定的任务加载对应的数据，humaneval，mbpp，mbtp等
executor.py 为代码执行模块，在代码生成自反馈时会调用该模块执行生成的代码并获得反馈
executor_utils.py 中有executor.py需要的一些函数
model.py为模型模块负责加载模型
generator.py 该模块控制模型文本生成过程
#### 使用不同策略的自反馈过程
base_generate.py 为基础的自反馈实现
not_tree_search.py 增加了生成的候选代码数量的非树搜索
tree_search.py 为使用树搜索的自反馈实现
tree_search_cached.py 为使用树搜索的自反馈实现，使用缓存策略
testcase_filter_first.py 为使用测试用例筛选的树搜索策略
testcase_filter_cached.py 为使用测试用例筛选的树搜索策略，并使用缓存策略
fastdebug_test.py 运行缓存了固定部分后的自反馈过程，比较使用缓存和不使用缓存的加速效果
code_expl_gen.py 对根据代码生成解释过程的测试
testcase_generate.py 根据任务描述生成测试用例

#### 其它工具文件
evaluate_cache.py 对模型输入中固定部分占比，以及使用cache后的加速比进行计算，得到论文第四章的对应结果
evaluate.py 对代码生成的结果进行分析，其中的get_pass_k用于计算pass@k准确率。
time_measure.py 对整个自反馈过程的时间统计
myutils.py包含了许多自反馈过程中需要使用的辅助函数