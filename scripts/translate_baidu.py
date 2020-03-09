# -*- coding: utf-8 -*- 
# @Time : 2019/8/8 11:58 
# @Author : Allen 
# @Site :百度翻译
import json
import time
import urllib.request
import urllib.parse
import hashlib
from config import baidu_key, baidu_passwd


class TranslatorBaidu(object):
    def __init__(self):
        self.url = "http://api.fanyi.baidu.com/api/trans/vip/translate"

    def get_md5(self, s):
        m = hashlib.md5(s.encode())
        return m.hexdigest()

    def request_translate(self, _from, to, text):
        prams = {
            "q": text,
            "from": _from,
            "to": to,
            "appid": baidu_key,
            "salt": "1",
            "sign": self.get_md5(baidu_key + text + "1" + baidu_passwd),
        }
        prams = urllib.parse.urlencode(prams)
        prams = prams.encode('utf-8')
        response = urllib.request.urlopen(self.url, data=prams, timeout=10)
        time.sleep(1)
        if response.status == 200:
            response = response.read().decode("utf-8")
            resp = json.loads(response)
            if "trans_result" not in resp.keys():
                return resp
            else:
                return resp['trans_result'][0]['dst']
        else:
            return "请求接口错误！"

    def translate_back(self, _from, to, text):
        zh_to_other = self.request_translate(_from=_from, to=to, text=text)
        other_to_zh = self.request_translate(_from=to, to=_from, text=zh_to_other)
        return other_to_zh


if __name__ == '__main__':
    baidu = TranslatorBaidu()
    text = baidu.request_translate(_from="en", to="zh", text="snow")
    print(text)