# -*- coding: utf-8 -*-
#将爬取的信息写入csv文件中
import csv
import re
from bs4 import BeautifulSoup
import requests
score_file="./data.csv"
head = ['房源编号', '所在城市', '所在区县', '所在街道或地区', '小区名称', '面积', '租赁方式', '房间朝向', '月租', '计费方式', '室', '厅',
        '卫', '入住', '租期', '看房', '所在楼层', '总楼层', '电梯', '车位', '用水', '用电', '燃气', '采暖']    # 写入文件的标题行
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:78.0) Gecko/20100101 Firefox/78.0'}
with open(score_file, 'w', newline='') as file_write:
    filewriter = csv.writer(file_write)#创建csv写入对象
    filewriter.writerow(head)#写入第一行
    for page in range(50):#以页数为迭代单元
        url = 'https://xa.zu.ke.com/zufang/pg' + str(page+1) + '/#contentList'
        response = requests.get(url=url, headers=headers)
        page_text = response.text
        soup = BeautifulSoup(page_text, 'html.parser')#创建beautifulsoup对象
        div_list = soup.find_all(class_='content__list--item')#class为关键字，find_all方法需要用到class_
        codes = [] #编号
        areas = [] #地区
        for div in div_list:
            code = re.search(r'data-house_code="(.*?)" ', str(div)).group()[17:-2]#利用正则匹配
            codes.append(code)
        c_list = soup.find_all(class_='content__list--item--des')
        for c in c_list:
            a_list = c.find_all('a')#这里的a为html代码名
            area = []
            for i in range(len(a_list)):
                a_text = a_list[i].text
                area.append(a_text)
            areas.append(area)
        for i in range(len(codes)):
            info = [] # extend方法，写入迭代对象
            info.extend([codes[i], '西安'] + areas[i])
            #这边按顺序将信息写入列表中
            url = 'https://xa.zu.ke.com/zufang/' + codes[i] + '.html'
            response = requests.get(url=url, headers=headers)
            page_text = response.text
            soup = BeautifulSoup(page_text, 'html.parser')
            #这里有几条数据出现了问题，我用continue的方式越过，目前还没有特别处理某条数据
            try:
                ul_text = soup.find('ul', class_='content__aside__list').text
                div_text = soup.find('div', class_='content__aside--title').text
            except:
                continue
            #group（）在正则表达式中用于获取分段截获的字符串
            #这里的5:-1的字段可以保证获取全部的信息，刚开始写的5:6后来发现，部分信息会略掉。
            Area= re.search(r' (.*?)㎡', ul_text).group()[1:]    #面积
            lease = re.search(r'租赁方式：(.*?)\n', ul_text).group()[5:-1]    #租赁方式
            home_aspect = re.search(r'朝向楼层：(.*?) ', ul_text).group()[5:-1]    # 朝向
            price = re.search(r'([0-9]*?)元/月', div_text).group()    # 月租
            try:
                chargeing = re.search(r'\((.*?)\)', div_text).group()[1:-1]    # 计费
            except AttributeError:#异常处理
                chargeing = 'None'
            rooms = re.search(r'([0-9*?])室', ul_text).group()    # 几室
            halls = re.search(r'([0-9*?])厅', ul_text).group()    # 几厅
            toilets = re.search(r'([0-9*?])卫', ul_text).group()    # 几卫
            #同样适用extend方法
            info.extend([Area, lease, home_aspect, price, chargeing, rooms, halls, toilets])
            div = soup.find('div', class_='content__article__info')
            ul_list = div.find_all('ul')
            ul_text = ''
            for ul in ul_list:
                ul_text += ul.text
            #这里ul_text获取所有数据，后面通过正则匹配将数据分组，之后写入csv文件中
            check_in = re.search(r'入住：(.*?)\n', ul_text).group()[3:-1]    # 入住
            tenancy = re.search(r'租期：(.*?)\n', ul_text).group()[3:-1]    # 租期
            see = re.search(r'看房：(.*?)\n', ul_text).group()[3:-1]    # 看房
            floor = re.search(r'楼层：(.*?)/', ul_text).group()[3:-1]    # 所在楼层
            all_floors = re.search(r'/(.*?)\n', ul_text).group()[1:-1]    # 总楼层
            elevator = re.search(r'电梯：(.*?)\n', ul_text).group()[3:-1]    # 电梯
            stall = re.search(r'车位：(.*?)\n', ul_text).group()[3:-1]    # 车位
            water = re.search(r'用水：(.*?)\n', ul_text).group()[3:-1]    # 用水
            electricity = re.search(r'用电：(.*?)\n', ul_text).group()[3:-1]    # 用电
            gas = re.search(r'燃气：(.*?)\n', ul_text).group()[3:-1]    # 燃气
            heating = re.search(r'采暖：(.*?)\n', ul_text).group()[3:-1]    # 采暖
            info.extend([check_in, tenancy, see, floor,all_floors, elevator, stall, water, electricity, gas, heating])
            print('编号为',info[0], '写入成功')
            filewriter.writerow(info)
