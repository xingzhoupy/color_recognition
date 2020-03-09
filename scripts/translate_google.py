# -*- coding: UTF-8 -*-
# !/usr/bin/python
# @Time     :2019/3/21 14:30
# @author   :Mo
# @function :回译调用谷歌翻译，模拟google token访问

import logging as logger
import urllib.parse as parse

import execjs
import requests


class TranslateGoogle:
    def __init__(self):
        self.ctx = execjs.compile("""
        function TL(a) {
        var k = "";
        var b = 406644;
        var b1 = 3293161072;
        var jd = ".";
        var $b = "+-a^+6";
        var Zb = "+-3^+b+-f";
        for (var e = [], f = 0, g = 0; g < a.length; g++) {
            var m = a.charCodeAt(g);
            128 > m ? e[f++] = m : (2048 > m ? e[f++] = m >> 6 | 192 : (55296 == (m & 64512) && g + 1 < a.length && 56320 == (a.charCodeAt(g + 1) & 64512) ? (m = 65536 + ((m & 1023) << 10) + (a.charCodeAt(++g) & 1023),
            e[f++] = m >> 18 | 240,
            e[f++] = m >> 12 & 63 | 128) : e[f++] = m >> 12 | 224,
            e[f++] = m >> 6 & 63 | 128),
            e[f++] = m & 63 | 128)
        }
        a = b;
        for (f = 0; f < e.length; f++) a += e[f],
        a = RL(a, $b);
        a = RL(a, Zb);
        a ^= b1 || 0;
        0 > a && (a = (a & 2147483647) + 2147483648);
        a %= 1E6;
        return a.toString() + jd + (a ^ b)
    };
    function RL(a, b) {
        var t = "a";
        var Yb = "+";
        for (var c = 0; c < b.length - 2; c += 3) {
            var d = b.charAt(c + 2),
            d = d >= t ? d.charCodeAt(0) - 87 : Number(d),
            d = b.charAt(c + 1) == Yb ? a >>> d: a << d;
            a = b.charAt(c) == Yb ? a + d & 4294967295 : a ^ d
        }
        return a
    }
    """)

    def get_google_token(self, text):
        """
           获取谷歌访问token
        :param text: str, input sentence
        :return: 
        """
        return self.ctx.call("TL", text)

    def open_url(self, url):
        """
          新增header，并request访问
        :param url: str, url地址
        :return: str, 目标url地址返回
        """
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) '
                          'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36'}
        req = requests.get(url=url, headers=headers)
        return req

    def any_to_any_translate(self, content, from_='zh-CN', to_='en'):
        """
           自定义选择
        :param content: str, 4891个字， 用户输入
        :param from_: str, original language
        :param to_:   str, target language
        :return: str, result of translate
        """
        max_len = self.max_length(content)
        if max_len:
            content = content[0:max_len]
        tk = self.get_google_token(content)
        content = parse.quote(content)
        url = "http://translate.google.cn/translate_a/single?client=t&sl={0}&tl={1}" \
              "&hl=zh-CN&dt=at&dt=bd&dt=ex&dt=ld&dt=md&dt=qca&dt=rw&dt=rm&dt=ss&dt=t&" \
              "ie=UTF-8&oe=UTF-8&source=btn&ssel=3&tsel=3&kc=0&tk={2}&q={3}".format(from_, to_, tk, content)
        result = self.open_url(url)
        result_json = result.json()
        res = self.translate_result(result_json)
        return res

    def any_to_any_translate_back(self, content, _from='zh-CN', to='en'):
        """
          中英，英中回译
        :param content:str, 4891个字， 用户输入
        :param from_: str, original language
        :param to_:   str, target language
        :return: str, result of translate
        """
        translate_content = self.any_to_any_translate(content, from_=_from, to_=to)
        result = self.any_to_any_translate(translate_content, from_=to, to_=_from)
        return result

    def max_length(self, content):
        """
          超过最大长度就不翻译
        :param content: str, need translate
        :return:
        """
        if len(content) > 4891:
            logger.info("翻译文本超过限制！")
            return 4891
        else:
            return None

    def translate_result(self, result):
        """
          删去无关词
        :param result: str
        :return: str
        """
        result_last = ''
        for res in result[0]:
            if res[0]:
                result_last += res[0]
        return result_last


if __name__ == '__main__':
    go = TranslateGoogle()
    sentence = "hello ,I like playing basketball and watching movies"
    text_translate = go.any_to_any_translate(sentence,from_='en', to_='zh-CN')
    print(text_translate)
    # for language_short_google_one in language_short_google:
    #     text_translate = go.any_to_any_translate_back(sentence, from_='zh', to_=language_short_google_one)
    #     print(text_translate)
