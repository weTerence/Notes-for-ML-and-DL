# Preview

markdown使用：<https://zhuanlan.zhihu.com/p/56943330>

# 深度学习八股文

<https://blog.csdn.net/weixin_42693876/article/details/120345924>
CV八股文：<https://blog.csdn.net/weixin_42693876/article/details/120345924>

## 生成模型VQ-VAE

全名：vector quantized variational autoencoder
参考链接：<https://zhuanlan.zhihu.com/p/463043201>

VQ-VAE相比VAE有两个重要的区别：

- 首先VQ-VAE采用离散隐变量，而不是像VAE那样采用连续的隐变量
- 然后VQ-VAE需要单独训练一个基于自回归的模型如PixelCNN来学习先验（prior），而不是像VAE那样采用一个固定的先验（标准正态分布）。

```python
print('A case for python codes')
```

## 生成模型PixelCNN

DeepMind于2016提出
参考链接：<https://zhuanlan.zhihu.com/p/461693342>

**生成模型**可以分成两种：**explicit density**和**implicit density**。

- explicit density是指的显式地定义概率密度来进行建模，
- 而implicit density则是通过其它方式间接地来估计概率密度，比如GAN通过对抗学习来间接学习到数据分布。

VAE和PixelCNN都属于对概率密度显式建模的生成模型，但是**VAE需要引入隐变量来估计概率密度**，而**PixelCNN是通过链式法则将概率密度转成一系列条件概率之积，然后通过网络来估计这些条件分布**。这里我们将介绍原始的PixelCNN以及后续改进版本GatedPixelCNN的原理以及具体的代码实现。
![generativeModels](2022-06-24-09-53-19.png)

**自回归模型简单介绍**
自回归模型（Autoregressive Model）是用自身做回归变量的过程，即利用**前期**若干时刻的随机变量的线性组合来描述**以后某时**刻随机变量的**线性回归模型**，它是**时间序列**中的一种常 见形式。

### PixelCNN

mask conv：堆积很多masked conv层之后，其感受野就可以扩展到左上全部像素范围
![](2022-06-24-11-48-53.png)
mask掉中心pixel的masked conv记为Mask A，即上图中的中心点取0，它用在网络的第一层。：**比如要预测green通道，此时输入应该只包括之前所有的pixels特征（下图中conext）以及当前pixel的red通道对应的特征**。
不mask掉中心pixel，这种方式称为Mask B，即上图中的中心点取1，它用在网络第一层之后的所有层：**比如要预测green通道，此时输入应该只包括之前所有的pixels特征（下图中conext）以及当前pixel的==red通道和green通道==对应的特征**。
![](2022-06-24-11-39-44.png)
预测顺序为逐像素点排列，并且每个像素点的三个通道按照RGB的顺序预测，例如：$Rx_1,Gx_1,Bx_1;Rx_2,Gx_2,Bx_2,...$

对于RGB三个通道，我们可以将每层的特征在channel维度分成三个部分，每个部分对应其中一个通道，然后可以通过对卷积核的输入通道做mask处理来限制连接。

PixelCNN网络结构：
![](2022-06-24-14-31-54.png)

PixelCNN总结

- explicit density形式的生成模型，通过链式法则将概率密度转成一系列条件概率之积，然后通过网络来估计这些条件分布
- PixelCNN属于自回归模型（autoregressive models）
- 对于PixelCNN，**训练过程是并行的**，即可以通过一次前向处理得到所有pixel的条件分布，因为对于训练数据我们已知图像的所有像素值，但是在**生成过程（推理过程）是序列处理的（串行）**，此时需要逐个pixel进行预测，共需要$n^2$次前向预测。 ​

### PixelCNN++

论文链接：<https://arxiv.org/abs/1701.05517>

### Gated PixelCNN

论文链接：<https://arxiv.org/abs/1606.05328>

对PixelCNN进行改进，改进的策略主要是两个部分：

- 一是在卷积中引入LSTM的gate机制；
  - 将两个masked conv间的ReLU用gate activate单元来替换，这样就构成了GatedMaskedConv
- 二是将masked conv分解成两个conv来解决感受野的“Blind spot”。 ​
  - PixelCNN需要堆积多个卷积层来实现更大的感受野，理想情况下我们希望覆盖当前pixel的所有左上部分的pixels，但是实际上由于masked conv本身的缺陷，感受野会出现**“Blind spot”**。
  ![](2022-06-24-14-34-20.png)
  论文提出的解决方案是采用两个conv stacks：horizontal stack和vertical stack。其中horizontal stack只处理当前所在行左边的pixels，而vertical stack处理上面所有行的pixels，我们把两个conv stack结合在一起，就可以实现想要的感受野，并且可以避免盲区的出现。 ​
  ![](2022-06-24-14-37-40.png)

## Transformer

**1. Transformer**
一个纯基于注意力机制的编码器和解码器
表现比 RNN 架构好，在机器翻译任务
**2. BERT**
使用 一个 Transformer 编码器
拓展到更一般的 NLP 的任务
**使用了 完型填空 的自监督的训练机制**
不需要使用标号
去预测一个句子里面 不见 masked 的词
从而获取对文本特征抽取的能力
BERT 极大的扩展了 Transformer 的应用
在一个大规模的、没有标号的数据上 训练出非常好的模型出来
**3. ViT**
将 Transformer 用到 CV 上面
把整个图片分割成很多 16 * 16 的小方块
每一个方块 patch 做成一个词 token，然后放进 Transformer 进行训练
证明：训练数据足够大 （1,000万 或者 一个亿的训练样本）的时候，Transformer 的架构精度优于CNN 架构
**4. MAE**
BERT 的一个 CV 的版本
基于 ViT ，BERT化
把整个训练 拓展到没有标号的数据上面
**通过完型填空来获取图片的一个理解**
不是第一个将 BERT 拓展到 CV 上
MAE 很有可能 未来影响最大
BERT 加速了 Transformer 架构 在 NLP 的应用
MAE 加速 Transformer 在 CV 上的应用
【论文解读】MAE：Masked Autoencoders Are Scalable Vision Learners

**5. SimMIM**
MAE论文中只尝试了使用原版ViT架构作为编码器，而表现更好的分层设计结构（以==Swin Transformer==为代表），并不能直接用上MAE方法。

于是，一场整合的范式就此在研究团队中上演。

代表工作之一是来自清华、微软亚研院以及西安交大提出SimMIM，它探索了Swin Transformer在MIM中的应用。

SimMIM和MAE有很多相似的设计和结论，而且效果也比较接近
不过相比MAE，SimMIM更加简单，而且也可以用来无监督训练金字塔结构的vision transformer模型如swin transformer等。
【论文解读】SimMIM：一种更简单的MIM方法_深兰深延AI的博客-CSDN博客

**6. GreenMIM**
与MAE相比，SimMIM它在可见和掩码图块均有操作，且计算量过大。有研究人员发现，即便是SimMIM的基本尺寸模型，也无法在一台配置8个32GB GPU的机器上完成训练。
基于这样的背景，东京大学&商汤&悉尼大学的研究员，提出了GreenMIM。
不光将Swin Transformer整合到了MAE框架上，既有与SimMIM相当的任务表现，还保证了计算效率和性能
将分层ViT的训练速度提高2.7倍，GPU内存使用量减少70%。
【论文解读】GreenMIM：将Swin与MAE结合，训练速度大大提升！

**5. SimMIM**
MAE论文中只尝试了使用原版ViT架构作为编码器，而表现更好的分层设计结构（以Swin Transformer为代表），并不能直接用上MAE方法。

于是，一场整合的范式就此在研究团队中上演。

代表工作之一是来自清华、微软亚研院以及西安交大提出SimMIM，它探索了Swin Transformer在MIM中的应用。

SimMIM和MAE有很多相似的设计和结论，而且效果也比较接近
不过相比MAE，SimMIM更加简单，而且也可以用来无监督训练金字塔结构的vision transformer模型如swin transformer等。
【论文解读】SimMIM：一种更简单的MIM方法_深兰深延AI的博客-CSDN博客

**6. GreenMIM**
与MAE相比，SimMIM它在可见和掩码图块均有操作，且计算量过大。有研究人员发现，即便是SimMIM的基本尺寸模型，也无法在一台配置8个32GB GPU的机器上完成训练。
基于这样的背景，东京大学&商汤&悉尼大学的研究员，提出了GreenMIM。
不光将Swin Transformer整合到了MAE框架上，既有与SimMIM相当的任务表现，还保证了计算效率和性能
将分层ViT的训练速度提高2.7倍，GPU内存使用量减少70%。
【论文解读】GreenMIM：将Swin与MAE结合，训练速度大大提升！

## transformer in NLP

self attention in NLP：<https://zhuanlan.zhihu.com/p/82312421>
英文版说的更清楚：<http://jalammar.github.io/illustrated-transformer/>

transformer如下：
![](2022-06-24-15-33-56.png)
更细致的transformer结构如下
![](![](2022-06-24-15-41-33.png).png)

encoder如下：
![](2022-06-24-15-34-10.png)

decoder如下：
![](2022-06-24-15-35-03.png)

deocder输出的self attention与encoder的输出做Attention吧......把encoder的输出当作Query做一个普通的Attention。

一个self-attention模块如下：（多头，8个attention模块）
![](2022-06-24-15-28-13.png)

### Vision Transformer(ViT)

Vision Transformer(ViT)将输入图片**拆分成16x16个patches**，每个patch做一次**线性变换降维同时嵌入位置信息**，然后**送入Transformer**，避免了像素级attention的运算。
ViT舍弃了CNN的归纳偏好问题，更加有利于在超大规模数据上学习知识，即大规模训练优归纳偏好，在众多图像分类任务上直逼SOTA。
![](2022-06-24-15-47-37.png)

#### 关于 transformer 的几个问题

1 transformer相对于CNN的优缺点：

- 优点：
  - Transformer关注全局信息，能建模更加长距离的依赖关系，而CNN关注局部信息，全局信息的捕捉能力弱。
  - Transformer避免了CNN中存在的**归纳偏好问题**（局部性假设）。

- 缺点：
  - Transformer复杂度比CNN高，但是ViT和Deformable DETR给出了一些解决方法来降低Transformer的复杂度。

2 归纳偏好：（即某种基于现实归纳出来的某种“假设”或“先验”）

在机器学习和深度学习中，很多学习算法经常会基于现实观察到的现象和规则对学习的问题做一些假设，这些假设就称为归纳偏好(Inductive Bias)。可以把归纳偏好理解为贝叶斯学习中的“先验”。

例如，在CNN中，假设特征具有局部性(Locality)的特点，即把相邻的一些特征融合到一起，会更容易得到“解”；在RNN中，假设每一时刻的计算依赖于历史计算结果；还有attention机制，也是从人的直觉、生活经验归纳得到的规则。

3 Q,K,V如何理解? 为什么不使用相同的Q和V？

- (1) 从点乘的物理意义上讲，两个向量的点乘表示两个向量的相似度。

- (2) 的物理意义是一样的，都表示同一个句子中不同token组成的矩阵。矩阵中的每一行，是表示一个token的word embedding向量。假设一个句子“Hello, how are you?”长度是6，embedding维度是300，那么  都是(6，300)的矩阵。

所以  和  的点乘可以理解为计算一个句子中每个token相对于句子中其他token的相似度，这个相似度可以理解为attetnion score，关注度得分。虽然有了attention score矩阵，但是这个矩阵是经过各种计算后得到的，已经很难表示原来的句子了，而  还代表着原来的句子，所以可以将attention score矩阵与  相乘，得到的是一个加权后的结果。

经过上面的解释，我们知道  和  的点乘是为了得到一个attention score 矩阵，用来对  进行提炼。  和  使用不同的  ,   来计算，可以理解为是在不同空间上的投影。正因为有了这种不同空间的投影，增加了表达能力，这样计算得到的attention score矩阵的泛化能力更高。这里解释下我理解的泛化能力，因为  和  使用了不同的  ,   来计算，得到的也是两个完全不同的矩阵，所以表达能力更强。但是如果不用  ，直接拿  和  点乘的话，attention score 矩阵是一个对称矩阵，所以泛化能力很差，这个矩阵对  进行提炼，效果会变差。