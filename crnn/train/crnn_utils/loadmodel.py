import os
import numpy as np
from importlib import reload
from PIL import Image

from keras.models import Model
from keras.layers import Input
# import keys
# import densenet
from crnn_utils import densenet, keys

reload(densenet)

characters = keys.alphabetChinese[1:]
# 卍
characters = characters[:]
nclass = len(characters)

input = Input(shape=(32, None, 1), name='the_input')
y_pred = densenet.dense_cnn(input, nclass)
basemodel = Model(inputs=input, outputs=y_pred)

modelPath = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'h5_model_adam', 'test01/weights-01-183.95.h5')

if os.path.exists(modelPath):
    basemodel.load_weights(modelPath)
else:
    print('目录不存在：{}'.format(modelPath))


def decode(pred):
    char_list = []
    pred_text = pred.argmax(axis=2)[0]
    print(pred_text)
    for i in range(len(pred_text)):
        if pred_text[i] != nclass and (not (i > 0 and pred_text[i - 1])) or (
                i > 1 and pred_text[i] == pred_text[i - 2]):
            char_list.append(characters[pred_text[i]])
    return u''.join(char_list)


def predict(img):
    width, height = img.size[0], img.size[1]
    scale = height * 1.0 / 32
    wight = int(width / scale)

    img = img.resize([width, 32], Image.ANTIALIAS)

    img = np.array(img).astype(np.float32) / 255.0 - 0.5

    X = img.reshape(1, 32, width, 1)
    y_pred = basemodel.predict(X)
    y_pred = y_pred[:, :, :]
    out = decode(y_pred)

    return out
