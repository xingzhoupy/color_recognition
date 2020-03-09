#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time  : 2020/3/7 14:56
# @Author: zhouxing
# PRODUCT_NAME: PyCharm 
# 抽取颜色
from bs4 import BeautifulSoup
from scripts.translate_google import TranslateGoogle
from scripts.translate_baidu import TranslatorBaidu
from config import read_file, save_excel
import time
from tqdm import tqdm


def color():
    # go = TranslateGoogle()
    baidu = TranslatorBaidu()
    colors_dict = {
        "颜色名称": [],
        "中文名称": [],
        "颜色RGB": [],
        "颜色16进制": []
    }

    def catch_color(body, color_dict):
        trs = body.find_all("tr")
        for tr in tqdm(trs):
            tds = tr.find_all("td")
            color_name = tds[1].text
            color_rgb = tds[2].text
            color_num = tds[3].text
            text_translate = baidu.request_translate(_from="en", to="zh", text=color_name)
            color_dict["颜色名称"].append(color_name)
            color_dict["中文名称"].append(text_translate)
            color_dict["颜色RGB"].append(color_rgb)
            color_dict["颜色16进制"].append(color_num)
        return color_dict

    str_html = read_file(r"C:\Users\Xiaoi\Desktop\demo.html")
    html_bs = BeautifulSoup(str_html, 'html.parser')
    bodys = html_bs.find_all("tbody")
    colors_dict = catch_color(bodys[2], colors_dict)
    colors_dict = catch_color(bodys[3], colors_dict)
    save_excel(model=colors_dict, path=r"C:\Users\Xiaoi\Desktop\color.xlsx")


if __name__ == '__main__':
    color()
