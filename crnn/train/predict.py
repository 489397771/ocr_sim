import os
import numpy as np
from PIL import Image
from glob import glob
import cv2
from crnn_utils import keys
from crnn_utils import loadmodel as model
# from crnn_utils.color_normalization import color_normalization


val_path = '/Users/cipher/Documents/work/ocr/text_renderer-master/output/train/*.jpg'
# /Users/cipher/Documents/work/ocr/text_renderer-master/output/val/00000255.jpg
image_files = glob(val_path)
label_all = keys.alphabetChinese[1:]
label_all = label_all[:]


def get_label(filename):
    res = []
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i in lines:
            res.append(i.strip())
    dic = {}
    for i in res:
        p = i.split(' ', 1)
        # print(p)
        dic[p[0]] = str(p[1])
    return dic


if __name__ == '__main__':
    num = 0
    task_name = 'test01'
    result_dir = './test_result'
    accuracy_path = os.path.join(result_dir, task_name)
    if not os.path.exists(accuracy_path):
        os.makedirs(accuracy_path)

    val_label_path = os.path.join(os.path.split(os.path.abspath(__file__))[0], 'train_val', '{}_val.txt'.format(task_name))
    image_label = get_label(val_label_path)
    # print(image_label)
    all_num = len(image_label.keys())
    print('make val label finish')
    suc = 0
    with open(os.path.join(accuracy_path, 'accuracy.txt'), 'w', encoding='utf-8')as f:
        f.write('image路径：{}\n'.format(os.path.split(val_path)[0]))
        print('val predict running...')
        for image_file in image_files:
            num += 1
            np_img = cv2.imread(image_file)
            # np_img1 = color_normalization(np_img, dst_min_gray=0, dst_max_gray=255)
            np_img1 = np.expand_dims(np_img, axis=-1)
            image = Image.fromarray(cv2.cvtColor(np_img1, cv2.COLOR_GRAY2RGB)).convert('L')

            out = model.predict(image)
            symbol_keys = {
                '？': '?',
                '！': '!',
                '：': ':'
            }
            out = ''.join([char if char not in symbol_keys.keys() else symbol_keys[char] for char in out])
            # f.write('{0:15}label: {1:50}\n'.format(os.path.split(image_file)[1],
            #                                        image_label[os.path.split(image_file)[1]]))
            # f.write('{0:13}predict: {1}'.format('', out))
            print(out)
        #     if out == image_label[os.path.split(image_file)[1]]:
        #         suc += 1
        #     else:
        #         f.write('↑'*20 + 'error' + '↑'*20)
        #     if num % 2000 == 1999:
        #         print('accuracy:{:.4f}'.format(suc / all_num))
        # f.write('accuracy:{:.4f}'.format(suc / all_num))
