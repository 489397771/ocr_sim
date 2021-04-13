import logging
import re
from load_model import EndToEndPredict
from angle.predict import angle_image

model = EndToEndPredict()
model.load()


class LQC(object):
    """
    身份证结构化识别
    """

    def __init__(self, result):
        self.result = result
        self.N = len(self.result)
        self.res = {}
        self.institution()
        self.license_classes()
        self.licenseNo()
        self.certificateNo()
        self.name()
        self.sex()
        self.birthNo()
        self.issuedday()

    def institution(self):
        """
        执业机构
        """
        institution = {}
        for i in range(self.N):
            txt = self.result[i]['text'].replace(' ', '')
            res = re.findall("([\u4e00-\u9fa5]+)[律师事务所]{3}", txt)

            if len(res) > 0:
                institution['执业机构'] = res[0].replace('执业机构', '').split('律')[0] + '律师事务所'
                self.res.update(institution)
                break

    def license_classes(self):
        """
        执业证类别
        """
        classes = {}
        for i in range(self.N):
            txt = self.result[i]['text'].replace(' ', '')
            res = re.findall("[执业证类别]{4,5}.*", txt)
            if len(res) > 0:
                if "职律师" in res[0]:
                    classes["执业证类别"] = re.findall("[执业证类别]{4,5}([\u4e00-\u9fa5]+)", res[0])[0]
                elif "职律师" in self.result[i-1]['text'].replace(' ', ''):
                    classes["执业证类别"] = self.result[i-1]['text'].replace(' ', '')
                elif "职律师" in self.result[i+1]['text'].replace(' ', ''):
                    classes["执业证类别"] = self.result[i+1]['text'].replace(' ', '')
                else:
                    continue
                self.res.update(classes)
                break

    def licenseNo(self):
        """
        执业证号
        """
        No = {}
        for i in range(self.N):
            txt = self.result[i]['text'].replace(' ', '').replace(' ', '').replace('i', '1').replace('s', '3').replace('O', '0').replace('o', '0')
            res = re.findall(r'[执业证号]{3}([0-9]{17})', txt)
            if len(res) > 0:
                No['执业证号'] = res[0]
                self.res.update(No)
                break
            res = re.findall(r'[执业证号]{3}', txt)
            if len(res) > 0 and len(self.result[i-1]['text'].replace(' ', '')) == 17 and \
                    self.result[i-1]['text'].replace(' ', '').replace('i', '1').replace('s', '3').replace('O', '0').isdigit():
                No['执业证号'] = self.result[i-1]['text'].replace(' ', '').replace('i', '1').replace('s', '3').replace('O', '0').replace('o', '0')
                self.res.update(No)
                break

    def certificateNo(self):
        """
        法律职业资格或律师资格证号
        """
        No = {}
        for i in range(self.N):
            txt = self.result[i]['text'].replace(' ', '').replace('i', '1').replace('s', '3').replace('O', '0').replace('o', '0').replace('h', 'A')
            res = re.findall("[职业资格]{0,4}[\u4e00-\u9fa5]+([ABC][0-9a-zA-Z]{14})", txt)
            res += re.findall("[资格证号]{0,4}[\u4e00-\u9fa5]+([ABC][0-9a-zA-Z]{14})", txt)
            res += re.findall("[法律职业资格]{3,6}[\u4e00-\u9fa5]+([ABC]{0,1}[0-9a-zA-Z]{14})", txt)
            res += re.findall("[法律职业资格]{3,6}(（[\u4e00-\u9fa5]）[\u4e00-\u9fa5]+[0-9]+)", txt)
            res += re.findall("[或律师资格证号]{3,6}([0-9]+.*)", txt)

            if len(res) > 0:
                No['法律职业资格或律师资格证号'] = ''.join(res)
                self.res.update(No)
                break

    def name(self):
        """
        持证人
        """
        name = {}

        for i in range(self.N):
            txt = self.result[i]['text'].replace(' ', '')
            txt = txt.replace(' ', '')
            res = re.findall("持证人.*", txt)
            if len(res) > 0:
                for j in range(i-2, i+2):
                    if j == i:
                        if len(res[0]) > 3:
                            s = re.findall("持证人([\u4e00-\u9fa5]+)", res[0])[0]
                            if len(s) < 2 or '法律' in s or '证号' in s or '资格' in s or not re.match('[\u4e00-\u9fa5]+', s):
                                continue
                            else:
                                name['持证人'] = s
                                self.res.update(name)
                                break
                        else:
                            continue
                    s = self.result[j]['text'].replace(' ', '')
                    if len(s) < 2 or '法律' in s or '证号' in s or '资格' in s or not re.match('[\u4e00-\u9fa5]+', s):
                        continue
                    else:
                        s = self.result[j]['text'].replace(' ', '').replace('考', '0')
                        name['持证人'] = s.split('0')[0]
                        self.res.update(name)
                        break

    def sex(self):
        """
        性别
        """
        sex = {}
        for i in range(self.N):
            txt = self.result[i]['text'].replace(' ', '')
            txt = txt.replace(' ', '')
            if txt == '男':
                sex["性别"] = '男'
                self.res.update(sex)
                break
            elif txt == '女':
                sex["性别"] = '女'
                self.res.update(sex)
                break

    def birthNo(self):
        """
        身份证号码
        """
        No = {}
        for i in range(self.N):
            txt = self.result[i]['text'].replace(' ', '').replace('s', '3').replace('O', '0').replace('i', '1').\
                replace('o', '0').replace('l', '1').replace('I', '1').replace('!', '1')
            # txt = txt.replace(' ', '')
            # 身份证号码
            res = re.findall(r'号\d*[X|x]', txt)
            # res += re.findall('号码\d*', txt)
            res += re.findall(r'\d{18}', txt)
            res += re.findall(r'\d{17}[X|x]', txt)
            # res += re.findall(r'.*s\d*[X|x]', txt)
            # res += re.findall(r'.*s\d.*', txt)
            # res += re.findall(r'.*O\d.*', txt)
            # res += re.findall(r'.*i\d.*', txt)
            # res += re.findall(r'.*o\d.*', txt)
            if len(res) > 0:
                if len(res[0]) >= 18:
                    No['身份证号'] = res[0].replace('号', '')
                    self.res.update(No)
                    break

    def issuedday(self):
        """
        发证日期
        """
        day = {}
        for i in range(self.N):
            txt = self.result[i]['text'].replace(' ', '')
            txt = txt.replace(' ', '')
            # 出生年月
            # res = re.findall('出生\d*年\d*月\d*', txt)
            res = re.findall(r'\d*年\d*月\d*', txt)

            if len(res) > 0:
                day['发证日期'] = res[0].replace('年', '-').replace('月', '-').replace('日', '')
                self.res.update(day)
                break


def ocr_bar_license(img_path):

    img = angle_image(img_path)

    logging.info('bar license predict start')
    model = EndToEndPredict()
    model.load()
    results = model.get_answer(img)
    lqc = LQC(results)
    if len(lqc.res.keys()) < 3:
        return {'error': '上传图片错误或者无法识别，请重新上传或手动填写'}

    return lqc.res


if __name__ == "__main__":
    img_path = r'/Users/cipher/Documents/work/ocr_sim/ctpn/data/1-1/21.jpg'
    print(ocr_bar_license(img_path))
