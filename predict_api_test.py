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
def dopredict_absa_sentence(test_data, host="127.0.0.1:3326"):
    """
    预测属性之后预测情感
    :param test_data: [('持妆不能输雅诗兰黛上妆即定妆雅诗兰黛DW粉底是我的心头好持妆遮瑕磨皮粉底液测评', '遮瑕', '成分'),...]
    :return:
    """
    url = f"http://{host}/api/absa_predict_sentence"
    data = {'data': test_data}
    headers = {'content-type': 'application/json'}
    r = requests.post(url, headers=headers, data=json.dumps(data),  timeout=360)
    print(r.json())
    return r.json()
def dopredict_brand(test_data, host="127.0.0.1:3326"):
    """
    品牌功效的判断
    :param test_data:
    :return:
    """
    url = f"http://{host}/api/brand_predict"
    data = {'data': test_data}
    headers = {'content-type': 'application/json'}
    r = requests.post(url, headers=headers, data=json.dumps(data), timeout=360)
    print(r.json())
    return r.json()
if __name__ == '__main__':
    # host = "127.0.0.1:3326"
    # host = "192.168.50.139:3326"
    host = "192.168.50.189:3326"
    absa_data = [('这个遮瑕效果很差，很不好用', '遮瑕'), ('抗氧化效果一般', '抗氧化'), ('海洋冰泉水润清透是MG面膜深受顾客喜爱的经典款面膜之一，已经使用了两年多了。该产品外包装精致、里面的面膜质感很好，与面部的贴合度、大小符合度都不错，使面膜的精华液能很好的均匀的敷于脸部各个部位。适用于各种肌肤，补水效果好，用后皮肤水润、光滑，以后还会回购的。', '水润'), ('海洋冰泉水润清透是MG面膜深受顾客喜爱的经典款面膜之一，已经使用了两年多了。该产品外包装精致、里面的面膜质感很好，与面部的贴合度、大小符合度都不错，使面膜的精华液能很好的均匀的敷于脸部各个部位。适用于各种肌肤，补水效果好，用后皮肤水润、光滑，以后还会回购的。', '质感'), ('海洋冰泉水润清透是MG面膜深受顾客喜爱的经典款面膜之一，已经使用了两年多了。该产品外包装精致、里面的面膜质感很好，与面部的贴合度、大小符合度都不错，使面膜的精华液能很好的均匀的敷于脸部各个部位。适用于各种肌肤，补水效果好，用后皮肤水润、光滑，以后还会回购的。', '补水')]
    dem8_data = [('持妆不能输雅诗兰黛上妆即定妆雅诗兰黛DW粉底是我的心头好持妆遮瑕磨皮粉底液测评', '遮瑕', '成分'), ('活动有赠品比较划算，之前买过快用完了，一支可以分两次使用，早上抗氧化必备VC', '抗氧化','成分'), ('海洋冰泉水润清透是MG面膜深受顾客喜爱的经典款面膜之一，已经使用了两年多了。该产品外包装精致、里面的面膜质感很好，与面部的贴合度、大小符合度都不错，使面膜的精华液能很好的均匀的敷于脸部各个部位。适用于各种肌肤，补水效果好，用后皮肤水润、光滑，以后还会回购的。', '水润', '功效'), ('海洋冰泉水润清透是MG面膜深受顾客喜爱的经典款面膜之一，已经使用了两年多了。该产品外包装精致、里面的面膜质感很好，与面部的贴合度、大小符合度都不错，使面膜的精华液能很好的均匀的敷于脸部各个部位。适用于各种肌肤，补水效果好，用后皮肤水润、光滑，以后还会回购的。', '质感','功效'), ('海洋冰泉水润清透是MG面膜深受顾客喜爱的经典款面膜之一，已经使用了两年多了。该产品外包装精致、里面的面膜质感很好，与面部的贴合度、大小符合度都不错，使面膜的精华液能很好的均匀的敷于脸部各个部位。适用于各种肌肤，补水效果好，用后皮肤水润、光滑，以后还会回购的。', '补水','功效')]
    purchase_data = [['飘了大概两周 终于到了\n希望我的头发别再掉了\n但是我又纠结要不要用，\n看到各种各样的评价，还说有疯狂的脱发期，那我要不要用？\n担心越用越脱啊。再加上我头发甚至连头皮都是干巴巴的类型。\n#REGENERATE \n#Grow Gorgeous Grow Gorgeous强效防脱增发精华头发增长生发液密发增发英国进口 \n#Dr.Hauschka 德国世家 \n#Alpecin 咖啡因C1 洗发水 \n#Alpecin 咖啡因防脱免洗发根滋养液 ', '买了alpecin洗发水增发液纠结要不要用？', 'grow gorgeous强效防脱增发精华']]
    brand_data = [
{"text": " malin goetz清洁面膜。净化清洁 补水。温和不刺激 敏感肌都可用。柏瑞特dr.brandt清洁面膜。深层清洁 抗氧化 排浊 紧致皮肤。伊菲丹超级面膜。急救修护 补水 紧致抗老 提亮肤色。菲洛嘉十全大补面膜。补水保湿 细腻毛孔 提亮肤色。法尔曼幸福面膜。补水 修护 抗老 唤肤。奥伦纳素冰白面膜。深层补水 细腻毛孔 提亮肤色 舒缓修复。@美妆薯  @美妆情报局", "brand": {"name": "菲洛嘉十全大补面膜", "pos": [98, 107]}, "attribute": {"name": "提亮", "pos": [162, 164]}},
{"text": "修丽可有一说一真的好用啊  买爆r  买爆r  买爆r 。修丽可植萃亮妍精华露。色修又称色修精华，植物精萃，舒缓亮妍，白话一点就是，修护红敏，红血丝 祛痘印，肤色不匀称等。。修丽可cf精华。抗氧化防止皱纹生成，保护肌肤免受空气污染，抗衰指数5 同时有效的美白，淡化黑色素。。高端医美修丽可紫米精华。一瓶就含有10 玻色因，硬核抗老，饱满丰盈，抗衰紧致真的很心动，成分很安全敏感肌也可放心用哦！。修丽可b5保湿精华。兼具保湿和修护的两大功效，在给予水份锁住水份的同时，又能修护平日因刺激带来皮肤屏障损伤，很适合干敏肌的宝宝们 。修丽可发光瓶。3 传明酸   1 曲酸   5 烟酰胺   5 磺酸  去黄提亮 淡斑淡痘印 搭配同系列色修精华  高效淡化痘印的同时美白肌肤有效改善顽固黄褐斑。", "brand": {"name": "修丽可cf", "pos": [87, 92]}, "attribute": {"name": "抗衰", "pos": [116, 118]}}
]
    # dem8_dd = [('持妆不能输雅诗兰黛上妆即定妆雅诗兰黛DW粉底是我的心头好持妆遮瑕磨皮粉底液测评', ['遮瑕','粉底'], '成分'), ('活动有赠品比较划算，之前买过快用完了，一支可以分两次使用，早上抗氧化必备VC', ['抗氧化'],'成分'), ('海洋冰泉水润清透是MG面膜深受顾客喜爱的经典款面膜之一，已经使用了两年多了。该产品外包装精致、里面的面膜质感很好，与面部的贴合度、大小符合度都不错，使面膜的精华液能很好的均匀的敷于脸部各个部位。适用于各种肌肤，补水效果好，用后皮肤水润、光滑，以后还会回购的。',['水润','补水'], '功效'), ('海洋冰泉水润清透是MG面膜深受顾客喜爱的经典款面膜之一，已经使用了两年多了。该产品外包装精致、里面的面膜质感很好，与面部的贴合度、大小符合度都不错，使面膜的精华液能很好的均匀的敷于脸部各个部位。适用于各种肌肤，补水效果好，用后皮肤水润、光滑，以后还会回购的。', ['质感'],'功效'), ('海洋冰泉水润清透是MG面膜深受顾客喜爱的经典款面膜之一，已经使用了两年多了。该产品外包装精致、里面的面膜质感很好，与面部的贴合度、大小符合度都不错，使面膜的精华液能很好的均匀的敷于脸部各个部位。适用于各种肌肤，补水效果好，用后皮肤水润、光滑，以后还会回购的。', ['补水'],'功效')]
    # dopredict_absa(host=host,test_data=absa_data)
    # dopredict_absa_fullscore(host=host,test_data=absa_data)
    # dopredict_dem8(host=host,test_data=dem8_data)
    # dopredict_purchase(host=host,test_data=purchase_data)
    # dem8(host=host,test_data=dem8_dd)
    # 句子情感
    # sentence_data = ['持妆不能输雅诗兰黛上妆即定妆雅诗兰黛DW粉底是我的心头好持妆遮瑕磨皮粉底液测评', '活动有赠品比较划算，之前买过快用完了，一支可以分两次使用，早上抗氧化必备VC']
    # dopredict_absa_sentence(host=host, test_data=sentence_data)
    # dopredict_absa_dem8(test_data=dem8_data,host=host)
    # dopredict_brand(test_data=brand_data,host=host)
