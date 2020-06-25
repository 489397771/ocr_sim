import re


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

    def full_name(self):
        """
        身份证姓名
        """
        name = {}
        # for i in range(self.N):
        txt = self.result[0]['text'].replace(' ', '').replace('，', '').replace('。', '')
        txt = txt.replace(' ', '')
        # 匹配身份证姓名
        res = re.findall("[\u4e00-\u9fa5]{1,4}", txt)
        if len(res) > 0:
            name['姓名'] = res[0].replace('姓名', '').replace('名', '')
            self.res.update(name)
            # break

    def nation(self):
        """
        民族汉
        """
        nation = {}
        for i in range(self.N):
            txt = self.result[i]['text'].replace(' ', '').replace('，', '').replace('。', '')
            txt = txt.replace(' ', '')
            # 民族汉
            res = re.findall(".*民.*族[\u4e00-\u9fa5]+", txt)
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
            txt = self.result[i]['text'].replace(' ', '').replace('，', '').replace('。', '')
            txt = txt.replace(' ', '')
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
            txt = self.result[i]['text'].replace(' ', '').replace('，', '').replace('。', '')
            txt = txt.replace(' ', '')
            # 出生年月
            # res = re.findall('出生\d*年\d*月\d*', txt)
            res = re.findall('\d*年\d*月\d*', txt)

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
            txt = self.result[i]['text'].replace(' ', '').replace('，', '').replace('。', '')
            txt = txt.replace(' ', '')
            # 身份证号码
            res = re.findall('号码\d*[X|x]', txt)
            # res += re.findall('号码\d*', txt)
            res += re.findall('\d{16,18}', txt)
            res += re.findall('.*s\d*[X|x]', txt)
            res += re.findall('.*s\d.*', txt)
            res += re.findall('.*O\d.*', txt)

            if len(res) > 0:
                No['身份证号码'] = res[0].replace('公民身份号码', '').replace('号码', '').replace('s', '3').replace('O', '0')
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
            txt = self.result[i]['text'].replace(' ', '').replace('，', '').replace('。', '')
            txt = txt.replace(' ', '')

            # 身份证地址
            if '住址' in txt or '省' in txt or '市' in txt or '县' in txt or '街' in txt or '村' in txt \
                    or "镇" in txt or "区" in txt or "城" in txt or '楼' in txt or '路'in txt or '大院' in txt:
                addString.append(txt.replace('住址', ''))

        if len(addString) > 0:
            add['身份证地址'] = ''.join(addString)
            self.res.update(add)
