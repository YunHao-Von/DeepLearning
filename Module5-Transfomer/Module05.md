# Module05 Transfomer  
## 1.简介  
seq2seq模型，指输入和输出都为sequence vector的模型，一般来说，输出的长度还需要模型自行决定。  
+ 大部分常见的seq2seq模型的架构中包含了一个encoder-decoder(编码器-解码器)的架构，这两个架构都可以用RNN或者transfomer来实现，主要是用来解决输入和输出的长度不相同的情况。  
+ Encoder是将一连串的输入，如文字、图像和声音讯号等，编码为单个向量，这个向量可以认为是整体输入的抽象表示，因为它包含了整体输入的信息。  
+ Decoder是将Encoder输出的向量逐步解码，一次输出一个结果，直到最后的目标输出被产生出来为止，每次输出都会影响下一次的输出，一般会在开头加上"BOS"表示开始解码，在结尾加上"EOS"表示输出结束。  

### 作业介绍:  
+ 英文翻译中文  
    + 输入:一句英文
    + 输出:中文翻译  
+ Todo  
    + 训练一个RNN架构的seq2seq来进行翻译  
    + 训练一个Transfomer大幅度提升效能 
    + 制作Back-Translation大幅度提升效能  

### Fairseq  
Fairseq 是一个用 PyTorch 编写的序列建模工具包，它可以为翻译、摘要、语言建模和其他文本生成任务训练自定义模型。  
安装过程:  
```
git clone https://github.com/pytorch/fairseq.git
cd fairseq
pip install -r requirements.txt
python setup.py build
python setup.py develop
```  
注意，在windows系统下如果需要安装此包，需要安装C++ 14.0来满足依赖(因为Python3是使用C++14.0)来进行安装的。  
如果上述命令无法运行，不妨直接使用pip指定源来安装:  
```
pip install fairseq==0.10.0 -i http://pypi.mirrors.ustc.edu.cn/simple --trusted-host pypi.mirrors.ustc.edu.cn
```  

### 数据集介绍:  
选用Ted2020数据集。  
