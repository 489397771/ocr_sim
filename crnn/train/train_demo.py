import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import cv2
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from PIL import Image
# from keras import backend as K
import tensorflow.compat.v1.keras.backend as K
from keras.layers import Input
from keras.layers.core import Lambda
from keras.models import Model
from keras.callbacks import ModelCheckpoint, TensorBoard

from importlib import reload
from crnn_utils import densenet2, keys


def get_session(gpu_fraction=1.0):
    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction, allow_growth=True)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


def readfile(filename):
    res = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for i in lines:
            res.append(i.strip())
    dic = {}
    for i in res:
        p = i.split(' ')
        dic[p[0]] = p[1]
    return dic


class random_uniform_num():
    """
    均匀随机，确保每轮每个只出现一次
    """
    def __init__(self, total):
        self.total = total
        self.range = [i for i in range(total)]
        np.random.shuffle(self.range)
        self.index = 0

    def get(self, batchsize):
        r_n = []
        if (self.index + batchsize) > self.total:
            r_n_1 = self.range[self.index: self.total]
            np.random.shuffle(self.range)
            self.index = (self.index + batchsize) - self.total
            r_n_2 = self.range[0: self.index]
            r_n.extend(r_n_1)
            r_n.extend(r_n_2)
        else:
            r_n = self.range[self.index: self.index + batchsize]
            self.index = self.index + batchsize
        return r_n


def gen(data_file, batchsize=128, maxlabellength=10, imagesize=(32, 200)):
    characters = keys.alphabet[1:]
    characters = characters[:] + u'卍'
    image_label = readfile(data_file)
    # print(image_label)
    _imagefile = [i for i, j in image_label.items()]
    x = np.zeros((batchsize, imagesize[0], imagesize[1], 1), dtype=np.float)
    labels = np.ones([batchsize, maxlabellength]) * 10000
    input_length = np.zeros([batchsize, 1])
    label_length = np.zeros([batchsize, 1])

    r_n = random_uniform_num(len(_imagefile))
    _imagefile = np.array(_imagefile)
    while 1:
        shufimagefile = _imagefile[r_n.get(batchsize)]
        for i, j in enumerate(shufimagefile):
            # np_img = cv2.imread(j, 0)
            np_img = Image.open(j).convert('L')
            if np_img is None:
                print('此地址错误：{}'.format(j))
                continue
            else:
                # np_img1 = color_normalization(np_img, dst_min_gray=0, dst_max_gray=255)
                np_img1 = np.expand_dims(np_img, axis=-1)
                img1 = Image.fromarray(cv2.cvtColor(np_img1, cv2.COLOR_GRAY2RGB)).convert('L')
                if img1.size != imagesize:
                    img1 = img1.resize((imagesize[1], imagesize[0]), Image.BILINEAR)
                img = np.array(img1, 'f') / 255.0 - 0.5

                x[i] = np.expand_dims(img, axis=2)
                # print('img shape:', img.shape)
                str = image_label[j]
                label_length[i] = len(str)
                if len(str) <= 0:
                    print('len < 0', j)
                if len(str) > maxlabellength:
                    str = str[: maxlabellength-1]
                input_length[i] = imagesize[1] // 8
                symbol_keys = {
                    '？': '?',
                    '！': '!',
                    '：': ':'
                }
                labels[i, :len(str)] = [int(characters.index(char)) if char not in symbol_keys.keys()
                                        else int(characters.index(symbol_keys[char])) for char in str]
        inputs = {
            'the_input': x,
            'the_labels': labels,
            'input_length': input_length,
            'label_length': label_length,
        }

        outputs = {'ctc': np.zeros(batchsize)}
        yield (inputs, outputs)


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args

    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def get_model(img_h, nclass):
    input = Input(shape=(img_h, None, 1), name='the_input')
    y_pred = densenet2.dense_cnn(input, nclass)

    basemodel = Model(inputs=input, outputs=y_pred)
    basemodel.summary()
    # 控制训练最后几层，
    for layer in basemodel.layers[:-5]:
        layer.trainable = False

    labels = Input(name='the_labels', shape=[None], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])
    model = Model(inputs=[input, labels, input_length, label_length], outputs=[loss_out])
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam', metrics=['accuracy'])
    # model.summary()
    return basemodel, model


if __name__ == '__main__':
    task_name = 'test02'
    train_label_path = os.path.join(os.path.split(os.path.abspath(__file__))[0], 'train_val', 'test01_train.txt')
    val_label_path = os.path.join(os.path.split(os.path.abspath(__file__))[0], 'train_val', 'test01_val.txt')

    train_img_num = [index + 1 for index, _ in enumerate(open(train_label_path, 'r'))][-1]
    val_img_num = [index + 1 for index, _ in enumerate(open(val_label_path, 'r'))][-1]

    img_h = 32
    img_w = 280
    batch_size = 128
    maxlabellength = 10

    nclass = len(keys.alphabet)
    print('-'*30)
    print('nclass=', nclass)

    K.set_session(get_session())
    reload(densenet2)
    basemodel, model = get_model(img_h, nclass)

    # modelPath = 'load model'
    modelPath = os.path.join(os.path.dirname(__file__), 'h5_model_adam', 'weight', 'weights-densenet4761k-09-0.19.h5')
    if os.path.exists(modelPath):
        print('Loading model weight...')
        model.load_weights(modelPath)
        print('load success: {}'.format(modelPath))
    else:
        pass

    train_loader = gen(train_label_path, batchsize=batch_size, maxlabellength=maxlabellength,
                       imagesize=(img_h, img_w))
    test_loader = gen(val_label_path, batchsize=batch_size, maxlabellength=maxlabellength,
                      imagesize=(img_h, img_w))

    print(os.path.join(os.path.dirname(__file__), 'h5_model_adam', task_name))
    checkpoint = ModelCheckpoint(filepath=os.path.join(os.path.dirname(__file__), 'h5_model_adam', task_name,
                                 'weights-{epoch:02d}-{val_loss:.2f}.h5'), monitor='val_loss',
                                 save_best_only=False)

    tensorboard = TensorBoard(log_dir=os.path.join(os.path.dirname(__file__), 'h5_model_adam', task_name, 'logs'))
    print('start training')
    model.fit_generator(
        train_loader,
        steps_per_epoch=train_img_num // batch_size,
        epochs=30,
        initial_epoch=0,
        max_queue_size=batch_size,
        validation_data=test_loader,
        validation_steps=val_img_num // batch_size,
        callbacks=[checkpoint, tensorboard]
    )
