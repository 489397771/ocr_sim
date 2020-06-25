### ctpn:
1. 执行setup.sh,如果失败，请切到具体语句逐条手动执行  

2. 训练和验证代码在ocr_smi/ctpn/ctpn下的train_net.py 和 predict.py  

3. 训练数据之前需要对数据及进行转换，具体转换方法因数据集而异，这里以VOCdevkit2007数据集为例，执行ocr_smi/ctpn/training_data下的ToVoc.sh转换生成对应的label文件。

### crnn：
1. 训练和验证代码在ocr_smi/crnn/train下的train_demo.py 和 predict.py  
2. 训练了两个model，网络框架不同所以对应的model不同，具体对应关系如下：
 `weights-addconvmax-19-1.04：denesentt.py`
 
    `weights-densenet4761k-09-0.19：denesent2.py`  

### ocr:
ocr识别主要调用ocr_smi目录下的ocr.py,会输出识别的内容，方便后台调用，但是模型以及正则匹配还没有很完善，需要继续优化