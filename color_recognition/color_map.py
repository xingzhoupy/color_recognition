#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time  : 2020/3/9 15:52
# @Author: zhouxing
# PRODUCT_NAME: PyCharm

def color_map_color(num_to_id_color_name_dict, result):
    """
    将预测的颜色对应色板
    :param num_to_id_color_name_dict:预测颜色id：色板ID_颜色中文名称
    :param result:预测结果
    :return:结果
    """
    print(result)
    colors = result["color"]
    result = {
        "color_info": [],
        "type":result["type"],
        "code":1
    }

    # 排序
    colors = sorted(colors.items(), key=lambda x: x[1], reverse=True)
    theme_id, theme_name = num_to_id_color_name_dict[colors[0][0].split("_")[-1]].split("_")
    result["color_id"] = theme_id
    result["color"] = theme_name

    for color in colors:
        map_name,map_id = color[0].split("_")
        result["color_info"].append({
            "map_id":map_id,
            "map_name":map_name,
            "map_score":color[1]
        })
    return result
