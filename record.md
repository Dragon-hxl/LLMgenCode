# 日志——记录开发过程
## self-debug每轮生成多个
self-debug第一轮sample 10个，之后每轮由这10个每个生成10个，共计100个，从这100个中选取10个作为下一轮的debug对象。如果这10个中谁对了就结束迭代。
2023.11.1
正常实验结果：
实验分为4类：test分别使用了从prompt和check中提取的， 选取的这10个一是按通过的test数量，二是按生成的概率排序选择前10.
目前仅在7b16k模型上进行实验。
7b16k   test_from_prompt    sort_by_passTestNum     69/164
7b16k   test_from_prompt    sort_by_scores          51/164
一些思考：
1.这些结果的差距是因为什么。
2.修改正确的程序是如何变化的。
3.没有修改对的程序出现了什么问题。