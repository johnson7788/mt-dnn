#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2021/6/30 2:00 下午
# @File  : api_test.py
# @Author: johnson
# @Contact : github: johnson7788
# @Desc  :

import requests
import json

def dopredict_papertext(test_data, host="127.0.0.1:3326"):
    """
    预测结果
    :param test_data:
    :return:
    """
    url = f"http://{host}/api/papertext"
    data = {'data': test_data}
    headers = {'content-type': 'application/json'}
    r = requests.post(url, headers=headers, data=json.dumps(data),  timeout=360)
    print(r.json())
    return r.json()


if __name__ == '__main__':
    host = "127.0.0.1:3326"
    # host = "192.168.50.139:3326"
    # host = "192.168.50.189:3326"
    papertext_data = [('Alexander M. Rush, Sumit Chopra, and Jason Weston. 2015. A neural attention model for abstractive sen- tence summarization. In Proceedings of the 2015 Conference on Empirical Methods in Natural Lan- guage Processing, pages 379–389, Lisbon, Portugal.','lines num:5,paragraph width:606.4753727682621,paragraph height:149.4655144971432,X0:200.05510049120073,X1:806.5304732594628,Y0:1851.4437028590435,Y1:2000.9092173561867,page width:1654,page height:2339')]
    dopredict_papertext(host=host,test_data=papertext_data)