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
    # print(result)
    colors = result["color"]
    result = {
        "color_info": [],
        "code": 1,
        "color_original": result["color"]
    }
    # 映射合并,将预测的颜色映射为excel颜色，并且将同样颜色的比例相加
    map_color = {}
    for color, score in colors.items():
        map_name, map_id = color.split("_")
        num_name = num_to_id_color_name_dict[map_id]
        if num_name not in map_color.keys():
            map_color[num_name] = score
        else:
            map_color[num_name] += score
    # 排序
    colors = sorted(map_color.items(), key=lambda x: x[1], reverse=True)
    theme_id, theme_name = colors[0][0].split("_")
    result["color_id"] = theme_id
    result["color"] = theme_name
    # 清洗掉 占比小于10%的颜色
    for id_name, score in colors:
        if score > 0.1:
            map_id, map_name = id_name.split("_")
            result["color_info"].append({
                "map_id": map_id,
                "map_name": map_name,
                "map_score": score
            })
    # 计算类型，如果颜色大于1小于4是拼接色，>4 则是花色
    if len(result["color_info"]) > 1 and len(result["color_info"]) <= 4:
        result["type"] = "拼接色"
    elif len(result["color_info"]) == 1:
        result["type"] = "纯色"
    else:
        result["type"] = "花色"
    return result
