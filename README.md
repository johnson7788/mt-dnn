# 8个维度的判断的模型接口
## 接口地址
/api/dem8

## 接收的参数的格式
type：类型是8种类型中的一种
[content:str, aspect_keywords:list, type:str, ...]


## type的类型包括
'effect','component','fragrance','pack','skin','promotion','service','price'

## 返回的结果格式
[[keyword1_info:dict, keyword2_info:dict],...]
keyword_info: {'keyword': keyword:str, 'labels': labels:list, 'locations': location:list, 'type': type:str}
location返回的是这个keyword的start和end的位置

## POST请求
输入示例:
```angular2html
def dem8(test_data, host="127.0.0.1:5010"):
    """
    预测结果
    :param test_data:
    :return:
    """
    url = f"http://{host}/api/dem8"
    data = {'data': test_data}
    headers = {'content-type': 'application/json'}
    r = requests.post(url, headers=headers, data=json.dumps(data),  timeout=360)
    print(r.json())
    return r.json()
dem8_dd = [('持妆不能输雅诗兰黛上妆即定妆雅诗兰黛DW粉底是我的心头好持妆遮瑕磨皮粉底液测评', ['遮瑕','粉底'], '成分'), ('活动有赠品比较划算，之前买过快用完了，一支可以分两次使用，早上抗氧化必备VC', ['抗氧化'],'成分'), ('海洋冰泉水润清透是MG面膜深受顾客喜爱的经典款面膜之一，已经使用了两年多了。该产品外包装精致、里面的面膜质感很好，与面部的贴合度、大小符合度都不错，使面膜的精华液能很好的均匀的敷于脸部各个部位。适用于各种肌肤，补水效果好，用后皮肤水润、光滑，以后还会回购的。',['水润','补水'], '功效'), ('海洋冰泉水润清透是MG面膜深受顾客喜爱的经典款面膜之一，已经使用了两年多了。该产品外包装精致、里面的面膜质感很好，与面部的贴合度、大小符合度都不错，使面膜的精华液能很好的均匀的敷于脸部各个部位。适用于各种肌肤，补水效果好，用后皮肤水润、光滑，以后还会回购的。', ['质感'],'功效'), ('海洋冰泉水润清透是MG面膜深受顾客喜爱的经典款面膜之一，已经使用了两年多了。该产品外包装精致、里面的面膜质感很好，与面部的贴合度、大小符合度都不错，使面膜的精华液能很好的均匀的敷于脸部各个部位。适用于各种肌肤，补水效果好，用后皮肤水润、光滑，以后还会回购的。', ['补水'],'功效')]
dem8(host=host,test_data=dem8_dd)
```
输出结果:
```angular2html
[[{'keyword': '遮瑕', 'labels': ['否'], 'locations': [[30, 32]], 'type': '成分'}, {'keyword': '粉底', 'labels': ['否', '否'], 'locations': [[20, 22], [34, 36]], 'type': '成分'}], [{'keyword': '抗氧化', 'labels': ['是'], 'locations': [[31, 34]], 'type': '成分'}], [{'keyword': '水润', 'labels': ['否', '是'], 'locations': [[4, 6], [115, 117]], 'type': '功效'}, {'keyword': '补水', 'labels': ['是'], 'locations': [[105, 107]], 'type': '功效'}], [{'keyword': '质感', 'labels': ['是'], 'locations': [[52, 54]], 'type': '功效'}], [{'keyword': '补水', 'labels': ['是'], 'locations': [[105, 107]], 'type': '功效'}]]
```