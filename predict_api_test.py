#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2021/6/30 2:00 下午
# @File  : api_test.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  :

import requests
import json

def dopredict_absa(test_data, host="127.0.0.1:3326"):
    """
    预测结果
    :param test_data:
    :return:
    """
    url = f"http://{host}/api/absa_predict"
    data = {'data': test_data}
    headers = {'content-type': 'application/json'}
    r = requests.post(url, headers=headers, data=json.dumps(data),  timeout=360)
    print(r.json())
    return r.json()
def dopredict_absa_fullscore(test_data, host="127.0.0.1:3326"):
    """
    预测结果
    :param test_data:
    :return:
    """
    url = f"http://{host}/api/absa_predict_fullscore"
    data = {'data': test_data}
    headers = {'content-type': 'application/json'}
    r = requests.post(url, headers=headers, data=json.dumps(data),  timeout=360)
    print(r.json())
    return r.json()
def dopredict_dem8(test_data, host="127.0.0.1:3326"):
    """
    预测结果
    :param test_data: [('持妆不能输雅诗兰黛上妆即定妆雅诗兰黛DW粉底是我的心头好持妆遮瑕磨皮粉底液测评', '遮瑕', '成分'),...]
    :return:
    """
    url = f"http://{host}/api/dem8_predict"
    data = {'data': test_data}
    headers = {'content-type': 'application/json'}
    r = requests.post(url, headers=headers, data=json.dumps(data),  timeout=360)
    print(r.json())
    return r.json()
def dopredict_purchase(test_data, host="127.0.0.1:3326"):
    """
    预测结果
    :param test_data: [('持妆不能输雅诗兰黛上妆即定妆雅诗兰黛DW粉底是我的心头好持妆遮瑕磨皮粉底液测评', '遮瑕', '成分'),...]
    :return:
    """
    url = f"http://{host}/api/purchase_predict"
    data = {'data': test_data}
    headers = {'content-type': 'application/json'}
    r = requests.post(url, headers=headers, data=json.dumps(data),  timeout=360)
    print(r.json())
    return r.json()
def dem8(test_data, host="127.0.0.1:3326"):
    """
    预测结果, 多个aspect关键字的情况
    :param test_data: [('持妆不能输雅诗兰黛上妆即定妆雅诗兰黛DW粉底是我的心头好持妆遮瑕磨皮粉底液测评', ['遮瑕','粉底'], '成分'),...]
    :return:
    """
    url = f"http://{host}/api/dem8"
    data = {'data': test_data}
    headers = {'content-type': 'application/json'}
    r = requests.post(url, headers=headers, data=json.dumps(data),  timeout=360)
    print(r.json())
    return r.json()
def dopredict_absa_dem8(test_data, host="127.0.0.1:3326"):
    """
    预测属性之后预测情感
    :param test_data: [('持妆不能输雅诗兰黛上妆即定妆雅诗兰黛DW粉底是我的心头好持妆遮瑕磨皮粉底液测评', '遮瑕', '成分'),...]
    :return:
    """
    url = f"http://{host}/api/absa_dem8_predict"
    data = {'data': test_data}
    headers = {'content-type': 'application/json'}
    r = requests.post(url, headers=headers, data=json.dumps(data),  timeout=360)
    print(r.json())
    return r.json()
if __name__ == '__main__':
    # host = "127.0.0.1:3326"
    # host = "192.168.50.139:3326"
    host = "192.168.50.189:3326"
    absa_data = [('这个遮瑕效果很差，很不好用', '遮瑕'), ('抗氧化效果一般', '抗氧化'), ('海洋冰泉水润清透是MG面膜深受顾客喜爱的经典款面膜之一，已经使用了两年多了。该产品外包装精致、里面的面膜质感很好，与面部的贴合度、大小符合度都不错，使面膜的精华液能很好的均匀的敷于脸部各个部位。适用于各种肌肤，补水效果好，用后皮肤水润、光滑，以后还会回购的。', '水润'), ('海洋冰泉水润清透是MG面膜深受顾客喜爱的经典款面膜之一，已经使用了两年多了。该产品外包装精致、里面的面膜质感很好，与面部的贴合度、大小符合度都不错，使面膜的精华液能很好的均匀的敷于脸部各个部位。适用于各种肌肤，补水效果好，用后皮肤水润、光滑，以后还会回购的。', '质感'), ('海洋冰泉水润清透是MG面膜深受顾客喜爱的经典款面膜之一，已经使用了两年多了。该产品外包装精致、里面的面膜质感很好，与面部的贴合度、大小符合度都不错，使面膜的精华液能很好的均匀的敷于脸部各个部位。适用于各种肌肤，补水效果好，用后皮肤水润、光滑，以后还会回购的。', '补水')]
    dem8_data = [('持妆不能输雅诗兰黛上妆即定妆雅诗兰黛DW粉底是我的心头好持妆遮瑕磨皮粉底液测评', '遮瑕', '成分'), ('活动有赠品比较划算，之前买过快用完了，一支可以分两次使用，早上抗氧化必备VC', '抗氧化','成分'), ('海洋冰泉水润清透是MG面膜深受顾客喜爱的经典款面膜之一，已经使用了两年多了。该产品外包装精致、里面的面膜质感很好，与面部的贴合度、大小符合度都不错，使面膜的精华液能很好的均匀的敷于脸部各个部位。适用于各种肌肤，补水效果好，用后皮肤水润、光滑，以后还会回购的。', '水润', '功效'), ('海洋冰泉水润清透是MG面膜深受顾客喜爱的经典款面膜之一，已经使用了两年多了。该产品外包装精致、里面的面膜质感很好，与面部的贴合度、大小符合度都不错，使面膜的精华液能很好的均匀的敷于脸部各个部位。适用于各种肌肤，补水效果好，用后皮肤水润、光滑，以后还会回购的。', '质感','功效'), ('海洋冰泉水润清透是MG面膜深受顾客喜爱的经典款面膜之一，已经使用了两年多了。该产品外包装精致、里面的面膜质感很好，与面部的贴合度、大小符合度都不错，使面膜的精华液能很好的均匀的敷于脸部各个部位。适用于各种肌肤，补水效果好，用后皮肤水润、光滑，以后还会回购的。', '补水','功效')]
    purchase_data = [['飘了大概两周 终于到了\n希望我的头发别再掉了\n但是我又纠结要不要用，\n看到各种各样的评价，还说有疯狂的脱发期，那我要不要用？\n担心越用越脱啊。再加上我头发甚至连头皮都是干巴巴的类型。\n#REGENERATE \n#Grow Gorgeous Grow Gorgeous强效防脱增发精华头发增长生发液密发增发英国进口 \n#Dr.Hauschka 德国世家 \n#Alpecin 咖啡因C1 洗发水 \n#Alpecin 咖啡因防脱免洗发根滋养液 ', '买了alpecin洗发水增发液纠结要不要用？', 'grow gorgeous强效防脱增发精华']]
    # dem8_dd = [('持妆不能输雅诗兰黛上妆即定妆雅诗兰黛DW粉底是我的心头好持妆遮瑕磨皮粉底液测评', ['遮瑕','粉底'], '成分'), ('活动有赠品比较划算，之前买过快用完了，一支可以分两次使用，早上抗氧化必备VC', ['抗氧化'],'成分'), ('海洋冰泉水润清透是MG面膜深受顾客喜爱的经典款面膜之一，已经使用了两年多了。该产品外包装精致、里面的面膜质感很好，与面部的贴合度、大小符合度都不错，使面膜的精华液能很好的均匀的敷于脸部各个部位。适用于各种肌肤，补水效果好，用后皮肤水润、光滑，以后还会回购的。',['水润','补水'], '功效'), ('海洋冰泉水润清透是MG面膜深受顾客喜爱的经典款面膜之一，已经使用了两年多了。该产品外包装精致、里面的面膜质感很好，与面部的贴合度、大小符合度都不错，使面膜的精华液能很好的均匀的敷于脸部各个部位。适用于各种肌肤，补水效果好，用后皮肤水润、光滑，以后还会回购的。', ['质感'],'功效'), ('海洋冰泉水润清透是MG面膜深受顾客喜爱的经典款面膜之一，已经使用了两年多了。该产品外包装精致、里面的面膜质感很好，与面部的贴合度、大小符合度都不错，使面膜的精华液能很好的均匀的敷于脸部各个部位。适用于各种肌肤，补水效果好，用后皮肤水润、光滑，以后还会回购的。', ['补水'],'功效')]
    # dopredict_absa(host=host,test_data=absa_data)
    # dopredict_absa_fullscore(host=host,test_data=absa_data)
    # dopredict_dem8(host=host,test_data=dem8_data)
    # dopredict_purchase(host=host,test_data=purchase_data)
    # dem8(host=host,test_data=dem8_dd)
    # 句子情感
    # sentence_data = ['持妆不能输雅诗兰黛上妆即定妆雅诗兰黛DW粉底是我的心头好持妆遮瑕磨皮粉底液测评', '活动有赠品比较划算，之前买过快用完了，一支可以分两次使用，早上抗氧化必备VC']
    # dopredict_absa(host=host, test_data=sentence_data)
    dopredict_absa_dem8(test_data=dem8_data,host=host)
