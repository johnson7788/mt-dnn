#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2020/06/28 4:56 下午
# @File  : api.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  : 还没有经过模型预测，是一个假装的api
######################################################
# 改造成一个flask api，
# 包括预测接口api
# /api/predict
######################################################
import json
import os
import re
import jieba


import logging.config

logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': True,
})

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -  %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger("Main")

from flask import Flask, request, jsonify, abort
app = Flask(__name__)

def pinpai_fake_predict(data):
    """
    假装预测结果
    :param data:[[text, keywords_text]]
    :type data:
    :return: 嵌套列表 预测的返回的结果，keyword，对应的标签，一个概率值，位置信息
                    one_result = [keyword, label, '0.5', start, end]
    :rtype:
    """
    hard_pinpai_text = "谷雨，兰，后，吕，CL，Three，追风，滋源，凡士林，霸王，UH，有情，马丁，春夏，Haa，福来，大宝，歌剧魅影，蒂花之秀，井田，天使之眼，伊索，春雨，维多利亚的秘密，火烈鸟，青蛙王子，万花镜，自然乐园，赫拉，李医生，摩洛哥油，934，三生花，好孩子，海瑟薇，Goat，塞巴斯汀，Ren，神秘博士，C咖，依云，森田，安娜苏，凌博士，德妃，儒意，溯华，红之，混合故事"
    hard_pinpai = hard_pinpai_text.split("，")
    for one in data:
        result = []
        text, keywords_text = one
        keywords = keywords_text.split(',')
        # 假设所有困难词用jieba分词后，没有匹配上，那么就返回预测不是品牌，否则预测为品牌
        # 不在困难词里面的都假设为是品牌
        text_split = jieba.lcut(text)
        for keyword in keywords:
            find_res = re.finditer(re.escape(keyword), text)
            if keyword in hard_pinpai:
                if keyword in text_split:
                    label = "品牌"
                else:
                    label = "不是品牌"
            else:
                # 不在困难词里面，直接判断为品牌
                label = "品牌"
            if find_res:
                for r in find_res:
                    start, end = r.span()
                    # 预测的返回的结果，keyword，对应的标签，一个概率值，位置信息
                    one_result = [keyword, label, '0.5', start, end]
                    result.append(one_result)
    # 过滤掉位置被包含的那些单词，如果一个单词的位置被另一个完全包含，说明是一个子词，子词的父词必须只能有一个有实体标志
    # 其实位置相交，也是有问题的，这里暂时过滤相交的了
    will_pop = []
    sorted_res_by_start = sorted(result, key=lambda x: x[3])
    for idx in range(1, len(result)):
        end = sorted_res_by_start[idx][-1]
        start = sorted_res_by_start[idx][-2]
        last_idx = 1
        last_start = sorted_res_by_start[idx-last_idx][-2]
        last_end = sorted_res_by_start[idx-last_idx][-1]
        last_word = sorted_res_by_start[idx-last_idx][0]
        # 如果发现上一个词已经是需要弹出的了，并且上一个词的位置还在，那继续寻找上上个词
        while last_word in will_pop and idx-last_idx >=0:
            last_idx += 1
            last_end = sorted_res_by_start[idx - last_idx][-1]
            last_word = sorted_res_by_start[idx - last_idx][0]
        word = sorted_res_by_start[idx][0]
        if end <= last_end:
            # 说明这个单词在上一个单词的内部，那么就不需要了，可以弹出去了
            if last_word != word:
                will_pop.append(word)
                print(f"检测到单词:{word}在单词:{last_word}当中，已去除:{word}")
        elif start == last_start and last_end < end:
            if last_word != word:
                will_pop.append(last_word)
                print(f"检测到单词:{last_word}在单词:{word}当中，已去除:{last_word}")
    final_result = [i for i in sorted_res_by_start if i[0] not in will_pop]
    # 嵌套一层结果，因为这里假设是单条数据的原因
    results = [final_result]
    return results


@app.route("/api/label_studio_pinpai_predict", methods=['POST'])
def pinpainer_predict():
    """
    用于label studio的品牌的预测, aspects词是可能是多个，是用逗号隔开
    Args:
        test_data: 需要预测的数据，是一个文字列表, [(content,aspects),...,]
        如果传过来的数据没有索引，那么需要自己去查找索引 [(content,aspects),...,]
    Returns: 返回格式是[one_result,one_result2,one_result3]
     嵌套列表 预测的返回的结果，keyword，对应的标签，一个概率值，位置信息
                    one_result = [keyword, label, '0.5', start, end]
    """
    jsonres = request.get_json()
    test_data = jsonres.get('data', None)
    results = pinpai_fake_predict(test_data)
    logger.info(f"收到的数据是:{test_data}")
    logger.info(f"预测的结果是:{results}")
    return jsonify(results)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3336, debug=False, threaded=False)