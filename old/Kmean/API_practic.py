#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 15:57:00 2018

@author: suliang

应用编号：11786399 
应用名称：Kmean
AK：dsGvfI9jU0VGLO8fke6IWbg2VntgVo4U
SK：XQGZhFegSO40bgnuKtpNQbo4tP1vomsU
应用类别：服务端

案例：如何通过百度地图API获得地址经纬度并绘制热力图
1. 申请应用获得AK
2. 通过AK从百度获得地址的经纬度
3. 通过百度示例获得html的源码：http://lbsyun.baidu.com/jsdemo.htm#c1_15

"""

# 根据地址通过API转化为经纬度

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def loadDataset(filename):
    with open(filename, 'r') as f: #打开txt
        lines = f.readlines()    # 用f.readlines()读入的是一个list，而用f.read()读入的是一个str.
        lines = lines[1:-1]  # 去除第一行描述
    data = []
    for i,line in enumerate(lines):
        data.append(line.split())   # list的操作跟array不一样，特殊点：赋值要用append, 
                                        # 行切片简单：lst[2]就是第2行
    city = []
    price = []                      # 列切片要用for循环：
    for i in range(len(data)):
        city.append(data[i][0])
        price.append(float(data[i][1]))
        
    return data



def getPointLngLat(addressName):
    from urllib import parse
    import urllib.request
    import hashlib
    import json

    #myAk = "cCVPT8gkYK7yNXHWGdSoP7rHCGEeT1pe"
    #mySk = "X6sZGKepd7Y5NWm01Ew9CFLmuO5hx9qL"
    
    myAk = 'dsGvfI9jU0VGLO8fke6IWbg2VntgVo4U'
    mySk = 'XQGZhFegSO40bgnuKtpNQbo4tP1vomsU'
    
    queryStr = "/geocoder/v2/?address=%s&output=json&ak=%s" % (addressName,myAk)

    encodedStr = parse.quote(queryStr, safe="/:=&?#+!$,;'@()*[]")
    # 生成基础地址
    rawStr = encodedStr + mySk
    # 对基础地址进行sn编码
    sn = hashlib.md5(parse.quote_plus(rawStr).encode("utf8")).hexdigest()
    # 生成最终访问地址
    eventualUrl = parse.quote("http://api.map.baidu.com" + queryStr + "&sn="+sn, safe="/:=&?#+!$,;'@()*[]")

    req = urllib.request.urlopen(eventualUrl)
    res = req.read().decode("utf-8")
    tmp = json.loads(res)
    # 获得json数据中的经纬度值（json数据加载后为字典形式）
    location = tmp['result']['location'] #经纬度坐标

    lat = str(location['lat'])#纬度坐标 
    lng = str(location['lng'])#经度坐标 
    
    return lng, lat  # 精度,维度


def writeToJson(latArray, lngArray, countArray): # 基于百度热力图API, 生成热力图
    file = open(r'pointsJson.json','w') #建立json数据文件
    for i in range(len(latArray)):
        lat = latArray[i]
        lng = lngArray[i]
        c = countArray[i]
            
        str_temp = '{"lat":' + str(lat) + ',"lng":' + str(lng) + ',"count":' + str(c) +'},'
        file.write(str_temp) #写入文档
    file.close() #保存


# ------------------------------------------------------------------
if __name__ == '__main__':
    
    test_id = 1
    
    if test_id == 0:  # 测试单点
        addressName = "北京"
        lng, lat = getPointLngLat(addressName)
        print(addressName + '的经度是：{}, 维度是：{}'.format(lng,lat))
        
    elif test_id == 1:  # 测试文件
        filename = 'place.txt'
        data = loadDataset(filename)
        lng = []
        lat = []
        count = []
        for i in range(len(data)):  # 批量获得经纬度
            addr = data[i][0]
            c_ = data[i][1]
            lng_,lat_ = getPointLngLat(addr)
            lng.append(lng_)
            lat.append(lat_)
            count.append(c_)
        writeToJson(lat, lng, count)
        
    else:
        print('Wrong test_id!')
        
        
        
        
