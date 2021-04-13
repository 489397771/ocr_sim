import logging
import re

from load_model import EndToEndPredict
from angle.predict import angle_image

model = EndToEndPredict()
model.load()


class IdCard(object):
    """
    身份证结构化识别
    """
    def __init__(self, result):
        self.result = result
        self.N = len(self.result)
        self.res = {}
        self.full_name()
        self.sex()
        self.nation()
        self.birthday()
        self.address()
        self.birthNo()

    def _re_sub(self, txt):
        txt = re.sub(u"([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a])", "", txt['text'])
        txt = txt.replace(' ', '')
        return txt

    def full_name(self):
        """
        身份证姓名
        """
        name = {}
        for i in range(self.N):
            txt = self._re_sub(self.result[i])
            # 民族汉
            res = re.findall(".*姓.*名[\u4e00-\u9fa5]+", txt)

            if len(res) > 0:
                name["姓名"] = res[0].replace('姓名', '').replace('名', '')
                self.res.update(name)
                break

    def nation(self):
        """
        民族汉
        """
        nation = {}
        for i in range(self.N):
            txt = self._re_sub(self.result[i])
            # 民族汉
            res = re.findall(".*民.*族[\u4e00-\u9fa5]+|^汉$", txt)

            if len(res) > 0:
                nation["民族"] = res[0].split('族')[-1]
                self.res.update(nation)
                break

    def sex(self):
        """
        性别女
        """
        sex = {}
        for i in range(self.N):
            txt = self._re_sub(self.result[i])
            res = re.findall('.*性.*别[\u4e00-\u9fa5]+|^男$|^女$', txt)
            res += re.findall(r'.*性[\u4e00-\u9fa5]+|^男$|^女$', txt)
            res += re.findall(r'.*别[\u4e00-\u9fa5]+|^男$|^女$', txt)

            if len(res) > 0:
                txt = res[0].split('别')[-1]
                if '男' in txt:
                    sex["性别"] = '男'
                    self.res.update(sex)
                    break
                elif '女' in txt:
                    sex["性别"] = '女'
                    self.res.update(sex)
                    break

    def birthday(self):
        """
        出生年月
        """
        birth = {}
        for i in range(self.N):
            txt = self._re_sub(self.result[i])
            # 出生年月
            # res = re.findall('出生\d*年\d*月\d*', txt)
            res = re.findall(r'\d*年\d*月\d*', txt)

            if len(res) > 0:
                birth['出生年月'] = res[0].replace('出生', '').replace('年', '-').replace('月', '-').replace('日', '')
                self.res.update(birth)
                break

    def birthNo(self):
        """
        身份证号码
        """
        No = {}
        for i in range(self.N):
            txt = self._re_sub(self.result[i])
            # 身份证号码
            res = re.findall(r'号码\d*[X|x]', txt)
            # res += re.findall('号码\d*', txt)
            res += re.findall(r'\d{16,18}', txt)
            res += re.findall(r'.*s\d*[X|x]', txt)
            res += re.findall(r'.*s\d.*', txt)
            res += re.findall(r'.*O\d.*', txt)
            res += re.findall(r'.*i\d.*', txt)
            if len(res) > 0:
                if len(res[0]) >= 18:
                    No['身份证号码'] = res[0].replace('公民身份号码', '').replace('号码', '').replace('s', '3').replace('O', '0').replace('i', '1')
                    self.res.update(No)
                    break

    def address(self):
        """
        身份证地址
        ##此处地址匹配还需完善
        """
        add = {}
        addString = []
        for i in range(self.N):
            txt = self._re_sub(self.result[i])

            # 身份证地址
            if '住址' in txt or '省' in txt or '市' in txt or '县' in txt or '街' in txt or '村' in txt \
                    or "镇" in txt or "区" in txt or "城" in txt or '楼' in txt or '路' in txt or '大院' in txt:
                txt = re.sub('住.', '', txt)
                addString.append(txt.replace('住址', ''))

        if len(addString) > 0:
            add['身份证地址'] = ''.join(addString).replace('住', '').replace('址', '')
            self.res.update(add)


def ocr_id_card(img_path):
    img = angle_image(img_path)

    logging.info('id_card predict start')
    results = model.get_answer(img)
    print(results)
    idcard = IdCard(results)
    if len(idcard.res.keys()) < 3:
        return {'error': '上传图片错误或者无法识别，请重新上传或手动填写'}

    return idcard.res


if __name__ == "__main__":
    p = r'/Users/cipher/Documents/work/ocr_sim/ctpn/data/idcard/2020-06-24 214059(40).jpg'
    print(ocr_id_card(p))
