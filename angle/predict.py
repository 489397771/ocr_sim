import os
import cv2 as cv
import numpy as np
from PIL import Image

import tensorflow.compat.v1 as tf
from keras.applications.vgg16 import preprocess_input, VGG16

tf.disable_v2_behavior()


def load_model():
    """加载模型"""
    model = VGG16(weights=None, classes=4)
    model.load_weights(os.path.join(os.getcwd(), 'angle/modelAngle.h5'), by_name=True)
    return model


model = load_model()
graph = tf.get_default_graph()


def predict(img=None):
    """
    图片文字方向预测
    """
    with graph.as_default():
        ROTATE = [0, 270, 180, 90]
        im = Image.fromarray(img).convert('RGB')
        w, h = im.size
        thesh = 0.05
        xmin, ymin, xmax, ymax = int(thesh * w), int(thesh * h), w - int(thesh * w), h - int(thesh * h)
        im = im.crop((xmin, ymin, xmax, ymax))  # 剪切图片边缘，清楚边缘噪声
        im = im.resize((224, 224))
        img = np.array(im)
        img = preprocess_input(img.astype(np.float32))
        pred = model.predict(np.array([img]))
        index = np.argmax(pred, axis=1)[0]
    return ROTATE[index]


def main(path):
    img = cv.imread(path)
    angle = predict(img=img)
    print(angle)


if __name__ == '__main__':
    img_path = r'/Users/cipher/Documents/work/ocr_sim/ctpn/data/idcard/2020-06-24 214059(47).jpg'
    main(img_path)
