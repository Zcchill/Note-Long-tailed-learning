# Long tail learning via logit adjustment （ICLR 2021）



## Introduction & Related work

​		这篇paper前面主要把长尾相关的方法分为三类，分别是改变模型输入(re-sampling 和 transfering 这些）、改变模型输出、改变模型中的损失函数。文中提到一般处理长尾问题时，改变输出或损失函数时都会搭配使用一下改变模型输入。这篇文章主要follow关注的是后面这两类方法：



- 改变模型输出（也就是改变最后的分类器）：

  - 对训练好的模型最后分类器做权重标准化（Post-hoc weight normalisation）
- 改变损失函数：
  - 偏re-balancing角度出发改变
  - 从调整margin出发改变


​		

​		按我的理解，这两类方法所涉及的大部分methods都是要将训练样本的类别频率作为先验分布的

