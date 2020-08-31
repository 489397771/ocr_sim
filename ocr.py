import os
from glob import glob

import numpy as np
import cv2 as cv
import tensorflow.compat.v1 as tf

from PIL import Image
from math import degrees, atan2, fabs, cos, sin, radians

from keras.layers import Input
from keras.models import Model

from angle.predict import predict as angle_detect

from crnn.train.crnn_utils import keys
from crnn.train.crnn_utils.densenet import dense_cnn

from ctpn.ctpn.cfg import Config
from ctpn.ctpn.other import resize_im, draw_boxes
from ctpn.ctpn.detectors import TextDetector as Detector
from ctpn.lib.networks.factory import get_network
from ctpn.lib.fast_rcnn.config import cfg
from ctpn.lib.fast_rcnn.test import test_ctpn

from idcard import IdCard
from lawyerLicense import LQC

tf.disable_v2_behavior()


class EndToEndPredict(object):
    def __init__(self):
        self.__imgH = 32
        self.charcaters = keys.alphabet[1:]
        self.charcaters = self.charcaters[:] + u'卍'
        self.nclass = len(self.charcaters)

    def load(self):
        self.__graph = tf.Graph()
        self.__ctpngraph = tf.Graph()
        self.__session = tf.Session(graph=self.__graph)

        with self.__session.as_default(), self.__session.graph.as_default():
            input = Input(shape=(self.__imgH, None, 1), name='the_input')
            y_pred = dense_cnn(input, self.nclass)
            basemodel = Model(inputs=input, outputs=y_pred)
            crnn_model_path = os.path.join(os.getcwd(), 'crnn', 'train', 'h5_model_adam', 'weight')
            # weights-addconvmax-19-1.04 denesentt.py
            # weights-densenet4761k-09-0.19 denesent2
            basemodel.load_weights(os.path.join(crnn_model_path, 'weights-addconvmax-19-1.04.h5'))
            self.__rmodel = basemodel

        config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=config, graph=self.__ctpngraph)
        with self.__ctpngraph.as_default(), self.sess.as_default():
            cfg.TEST.HAS_RPN = True  # use RPN for proposals

            # load network
            self.net = get_network('VGGnet_test')
            # load model
            self.saver = tf.train.Saver()
            ctpn_model_path = os.path.join(os.getcwd(), 'ctpn', 'models')
            ckpt = tf.train.get_checkpoint_state(ctpn_model_path)
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)

            self.textdetector = Detector()
            self.scale, self.max_scale = Config.SCALE, Config.MAX_SCALE

    def get_answer(self, image):
        origin = (image.shape[1], image.shape[0])
        img, f = resize_im(image, scale=self.scale, max_scale=self.max_scale)
        with self.__session.as_default(), self.sess.as_default():
            # detector
            scores, boxes = test_ctpn(self.sess, self.net, img)
            boxes = self.textdetector.detect(boxes, scores[:, np.newaxis], img.shape[:2])
            text_recs, tmp = draw_boxes(img, boxes, caption='img_name', wait=True, is_display=False)
            dst = tmp.shape[1], tmp.shape[0]
            text_recs = self._to_size(dst, origin, text_recs)

        text_recs = self.sort_box(text_recs)

        with self.__session.as_default(), self.__session.graph.as_default():
            # recognition
            results = []
            xdim, ydim = image.shape[1], image.shape[0]

            for index, rec in enumerate(text_recs):

                pt1 = (max(1, rec[0]), max(1, rec[1]))
                pt2 = (rec[2], rec[3])
                pt3 = (min(rec[6], xdim - 2), min(rec[7], ydim - 2))
                pt4 = (rec[4], rec[5])

                degree = degrees(atan2(pt2[1] - pt1[1], pt2[0] - pt1[0]))
                partImg = self.dumpRotateImage(image, degree, pt1, pt2, pt3, pt4)

                # 过滤异常图片
                if partImg.shape[0] < 1 or partImg.shape[1] < 1 or partImg.shape[0] > partImg.shape[1]:
                    # print('less 0')
                    continue

                partImg = Image.fromarray(partImg).convert('L')
                width, height = partImg.size[0], partImg.size[1]
                scale = height * 1.0 / self.__imgH
                width = int(width / scale)
                partImg = partImg.resize([width, self.__imgH], Image.ANTIALIAS)
                partImg = np.array(partImg).astype(np.float32) / 255.0 - 0.5
                X = partImg.reshape([1, self.__imgH, width, 1])
                y_pred = self.__rmodel.predict(X)
                y_pred = y_pred[:, :, :]
                # print(max(y_pred[:, :]))
                out = self._decode(y_pred)

                if len(out) > 0:
                    results.append({
                        # 'position': {
                        #     'point1': pt1,
                        #     'point2': pt2,
                        #     'point3': pt3,
                        #     'point4': pt4,
                        # },
                        'text': out,
                    })
            return results

    def _decode(self, pred):
        char_list = []
        pred_text = pred.argmax(axis=2)[0]
        for i in range(len(pred_text)):
            if pred_text[i] != self.nclass - 1 and (
                    (not (i > 0 and pred_text[i] == pred_text[i - 1])) or (i > 1 and pred_text[i] == pred_text[i - 2])):
                char_list.append(self.charcaters[pred_text[i]])
        return u''.join(char_list)

    def _to_size(self, origin, dst, bbox):
        for index, res in enumerate(bbox):
            for i in range(len(res)):
                bbox[index, i] = int(bbox[index, i] * dst[i % 2] / float(origin[i % 2]))
        return bbox

    def sort_box(self, box):
        """
        对box进行排序
        """
        box = sorted(box, key=lambda x: sum([x[1], x[3], x[5], x[7]]))
        return box

    def dumpRotateImage(self, img, degree, pt1, pt2, pt3, pt4):
        height, width = img.shape[:2]
        heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
        widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))
        matRotation = cv.getRotationMatrix2D((width // 2, height // 2), degree, 1)
        matRotation[0, 2] += (widthNew - width) // 2
        matRotation[1, 2] += (heightNew - height) // 2

        imgRotation = cv.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
        pt1 = list(pt1)
        pt3 = list(pt3)

        [[pt1[0]], [pt1[1]]] = np.dot(matRotation, np.array([[pt1[0]], [pt1[1]], [1]]))
        [[pt3[0]], [pt3[1]]] = np.dot(matRotation, np.array([[pt3[0]], [pt3[1]], [1]]))
        ydim, xdim = imgRotation.shape[:2]
        imgOut = imgRotation[max(1, int(pt1[1])): min(ydim - 1, int(pt3[1])),
                 max(1, int(pt1[0])): min(xdim - 1, int(pt3[0]))]

        return imgOut

    def _unload(self):
        self.__session.close()
        self.sess.close()
        del self.__rmodel, self.__session, self.sess


def ocr_result(img_path):

    img = cv.imread(img_path)

    angle = angle_detect(img=img)
    im = Image.fromarray(img)
    if angle == 90:
        im = im.transpose(Image.ROTATE_90)
    elif angle == 180:
        im = im.transpose(Image.ROTATE_180)
    elif angle == 270:
        im = im.transpose(Image.ROTATE_270)
    img = np.array(im)

    print('predict start')

    model = EndToEndPredict()
    model.load()
    results = model.get_answer(img)
    # print(results)
    idcard = IdCard(results)
    # print(idcard.res)
    if len(idcard.res.keys()) < 3:
        return {'error': '上传图片错误或者无法识别，请重新上传或手动填写'}

    return idcard.res


def ocr_bar_license(img_path):

    img = cv.imread(img_path)

    angle = angle_detect(img=img)
    im = Image.fromarray(img)
    if angle == 90:
        im = im.transpose(Image.ROTATE_90)
    elif angle == 180:
        im = im.transpose(Image.ROTATE_180)
    elif angle == 270:
        im = im.transpose(Image.ROTATE_270)
    img = np.array(im)

    print('predict start')

    model = EndToEndPredict()
    model.load()
    results = model.get_answer(img)
    # print(results)
    # print('--' * 100)
    lqc = LQC(results)
    if len(lqc.res.keys()) < 3:
        return {'error': '上传图片错误或者无法识别，请重新上传或手动填写'}

    return lqc.res


def main():
    img_path = r'/Users/cipher/Documents/work/ocr_sim/ctpn/data/idcard/*.jpg'
    with open('ocr_result.txt', 'a', encoding='utf-8') as ocr_r:

        for imgPath in glob(img_path):
            name = imgPath.split('2020-')[1]
            result = ocr_result(imgPath)
            ocr_r.write('name:{} \n result:{}\n'.format(name, result))
            print('{} ocr success'.format(name))


if __name__ == '__main__':
    # main()
    # img_path = r'/Users/cipher/Documents/work/ocr_sim/ctpn/data/demo/5.jpeg'
    img_path = r'/Users/cipher/Documents/work/ocr_sim/ctpn/data/1-1/21.jpg'

    print(ocr_bar_license(img_path))
