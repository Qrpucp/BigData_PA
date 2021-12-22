# 大数据计算基础大作业

- 题目：选修41题

> 41.【 X=1.0】 时间序列是一种重要的结构化数据形式，其处理对经济、金融数据尤为重要。
> 时间序列异常检测在工业界是非常常见的任务，模型常常要求对所判断出的异常给出合理的
> 解释，从而帮助人们做出相应的动作。近年来，可解释的时序建模多着眼于离散时序，在时
> 间轴上将时序分段，然后从分段中抓出可以用于判断异常的表示，常见的方法包括：
> （ 1） 字典方法：寻找时序分段的特征值；
> （ 2） 形状方法：寻找时序分段的特殊波形；
> （ 3） 聚类方法：寻找时序分段的分类特征。
> 实验内容：
> （ 1） 基于以上背景，请查阅相关资料，了解关于离散时序异常检测的相关方法；
> （ 2） 选择合适的工业相关时序数据集，并基于此数据集实现至少两种异常检测方法；
> （ 3） 对使用的多种策略的准确性进行比较和分析，考察不同类型数据集对不同方法的
> 影响（即某种策略是否只在某种特定数据集上表现良好）。

- raw_dataset.csv：原始数据集，[来源](https://github.com/SilenceSengoku/IsolationFroest2)

- anomaly_detection.py：孤立森林及字典法异常检测的实现
- processed_data.csv：处理后生成的数据

# 参考资料

[SilenceSengoku/IsolationFroest2](https://github.com/SilenceSengoku/IsolationFroest2)

[[机器学习]实战异常检测算法-Isolation Forest](https://zhuanlan.zhihu.com/p/93281351)

[时间序列丨基础概念理论 & 异常检测算法 & 相关学习资源 & 公开数据集](https://dreamhomes.top/posts/202106291700/)

[异常检测算法 -- 孤立森林（Isolation Forest）剖析](https://zhuanlan.zhihu.com/p/74508141)

[【异常检测】孤立森林（Isolation Forest）算法简介](https://www.cnblogs.com/guoyaohua/p/Isolation_Forest.html)

[时间序列异常检测（一）—— 算法综述](https://zhuanlan.zhihu.com/p/142320349)



