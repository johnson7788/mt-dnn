#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2021/6/30 2:00 下午
# @File  : api_test.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  :

import requests
import json

def dopredict_absa(test_data, host="127.0.0.1:5010"):
    """
    预测结果
    :param test_data:
    :return:
    """
    url = f"http://{host}/api/absa_predict"
    data = {'data': test_data}
    headers = {'content-type': 'application/json'}
    r = requests.post(url, headers=headers, data=json.dumps(data),  timeout=360)
    return r.json()
def dopredict_dem8(test_data, host="127.0.0.1:5010"):
    """
    预测结果
    :param test_data:
    :return:
    """
    url = f"http://{host}/api/dem8_predict"
    data = {'data': test_data}
    headers = {'content-type': 'application/json'}
    r = requests.post(url, headers=headers, data=json.dumps(data),  timeout=360)
    return r.json()


if __name__ == '__main__':
    # host = "127.0.0.1:5018"
    host = "192.168.50.189:5018"
    test_data = [('持妆不能输雅诗兰黛上妆即定妆雅诗兰黛DW粉底是我的心头好持妆遮瑕磨皮粉底液测评', '遮瑕'), ('活动有赠品比较划算，之前买过快用完了，一支可以分两次使用，早上抗氧化必备VC', '抗氧化'), ('海洋冰泉水润清透是MG面膜深受顾客喜爱的经典款面膜之一，已经使用了两年多了。该产品外包装精致、里面的面膜质感很好，与面部的贴合度、大小符合度都不错，使面膜的精华液能很好的均匀的敷于脸部各个部位。适用于各种肌肤，补水效果好，用后皮肤水润、光滑，以后还会回购的。', '水润'), ('海洋冰泉水润清透是MG面膜深受顾客喜爱的经典款面膜之一，已经使用了两年多了。该产品外包装精致、里面的面膜质感很好，与面部的贴合度、大小符合度都不错，使面膜的精华液能很好的均匀的敷于脸部各个部位。适用于各种肌肤，补水效果好，用后皮肤水润、光滑，以后还会回购的。', '质感'), ('海洋冰泉水润清透是MG面膜深受顾客喜爱的经典款面膜之一，已经使用了两年多了。该产品外包装精致、里面的面膜质感很好，与面部的贴合度、大小符合度都不错，使面膜的精华液能很好的均匀的敷于脸部各个部位。适用于各种肌肤，补水效果好，用后皮肤水润、光滑，以后还会回购的。', '补水')]
    # dopredict_absa(host=host,test_data="absa")
    dopredict_dem8(host=host,test_data="dem8")