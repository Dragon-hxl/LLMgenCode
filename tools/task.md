



一直未解决的[1, 6, 10, 17, 19, 32, 36, 38, 64, 70, 71, 74, 75, 81, 83, 93, 102, 103, 104, 106, 107, 113, 114, 118, 119, 123, 125, 129, 130, 131, 141, 144, 145, 146, 147, 148, 151, 154, 156, 160, 161, 163]

more_test_not_passed = [1, 3, 6, 9, 10, 16, 17, 19, 21, 32, 36, 38, 39, 46, 50, 62, 64, 65, 68, 69, 70, 71, 74, 75, 76, 77, 81, 83, 87, 89, 90, 91, 92, 93, 94, 97, 99, 100, 102, 103, 104, 106, 107, 109, 110, 113, 114, 115, 116, 118, 119, 120, 123, 125, 127, 128, 129, 130, 131, 132, 133, 134, 138, 139, 141, 142, 144, 145, 146, 147, 148, 149, 151, 153, 154, 155, 156, 157, 158, 160, 161, 163]

0:列表两数差小于阈值
解决
total testcase num : 577
correct testcase num : 292
correct percent 0.5060658578856152

1:括号匹配
未解决
total testcase num : 425
correct testcase num : 24
correct percent 0.05647058823529412

2:取小数部分
解决
total testcase num : 1100
correct testcase num : 53
correct percent 0.04818181818181818
生成正确的test过于简单,复杂的test不对

3:找到和小于0的点
pt,ct未解决,tT已经解决
total testcase num : 687
correct testcase num : 382
correct percent 0.5560407569141194
pt有歧义,生成的test需要选择得到没歧义的

4.计算平均绝对误差
解决
total testcase num : 633
correct testcase num : 26
correct percent 0.04107424960505529
生成的test有很多0,重复比较多

5.往列表中间隔插数
解决
total testcase num : 283
correct testcase num : 24
correct percent 0.08480565371024736
生成的test无问题

6.括号匹配并计算括号层数
未解决
total testcase num : 769
correct testcase num : 110
correct percent 0.14304291287386217
输入中用空格分开两个需要匹配的括号,生成的没有用上空格也就是匹配的括号组数量一直为1

10.在提供的字符串上生成回文字符串
total testcase num : 395
correct testcase num : 153
correct percent 0.38734177215189874
生成的程序输入都是回文串,没有体现函数功能,有歧义且没意义.

17. 字符串的映射。
total testcase num : 564
correct testcase num : 72
correct percent 0.1276595744680851
生成的例子重复度太高.

19. 数字排序但需要先把英文单词转化成罗马数字
total testcase num : 1090
correct testcase num : 345
correct percent 0.3165137614678899

32.找多项式的零点.
total testcase num : 631
correct testcase num : 10
correct percent 0.01584786053882726
生成的例子过于简单,但pt已经满足条件

36.找出小于n且被11和13整除的数中数字7出现的次数
total testcase num : 1154
correct testcase num : 87
correct percent 0.07538994800693241
生成的测试用例没问题

38.实现一个循环解码器
total testcase num : 689
correct testcase num : 162
correct percent 0.2351233671988389
生成的测试用例没什么问题,就是体现不出输入输出的变化,实现函数完全依赖于文档描述

64.找出字符串中的元音
total testcase num : 1875
correct testcase num : 567
correct percent 0.3024
生成的重复太多

70.列表按给定要求排序
total testcase num : 268
correct testcase num : 8
correct percent 0.029850746268656716
生成的测试用例太简单,没什么实际意义

71.给3条边,计算三角形面积
total testcase num : 643
correct testcase num : 206
correct percent 0.3203732503888025
生成的都是极端例子,没什么意义且覆盖面不广

74.给定两个字符串构成的列表,返回总字符数最小的列表
total testcase num : 574
correct testcase num : 232
correct percent 0.40418118466898956
重复多,去重后的用例没什么问题