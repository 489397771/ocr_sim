#!/usr/bin/env bash
mkdir -p ../data/pretrain;
python split_label.py; # 将其他图片格式转换成jpg格式，并生成对应的label文件
rm -rf mlt;
python ToVoc.py; # 将数据转换成Voc数量集
mv ./TEXTVOC ../data/VOCdevkit2007;
#rm -rf TEXTVOC