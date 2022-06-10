"""
这个文件负责模仿前端对后端数据的传输
"""

import aiohttp
from aiohttp import web
import asyncio
import requests
from base64 import b64encode

async def check_alldata(session, url):
    """

    :param session:
    :param url:
    :return:
    """
    headers = {'content-type': 'application/json'}
    params = {'username': 'test_user_1', 'password': '123456', 'name': '你是谁'}
    async with session.get(url, json=params, headers=headers) as response:
        r = await response.json(content_type=None)
    return r


async def check_register(session, url):
    """
    测试注册方法，模仿前端输入数据，并提交
    :param session:
    :param url:
    :return:
    """
    # 传输json格式的数据
    headers = {'content-type': 'application/json'}
    params = {'username': 'test_user_1', 'password': '123456', 'name': '胡'}
    async with session.get(url, json=params, headers=headers) as response:
        r = await response.json(content_type=None)
    return r


async def check_login(session, url):
    """

    :param session:
    :param url:
    :return:
    """
    headers = {'content-type': 'application/json'}
    params = {'username': 'test_user_1', 'password': '123456', 'name': '你是谁'}
    async with session.get(url, json=params, headers=headers) as response:
        r = await response.json(content_type=None)
    return r


async def check_del(session, url):
    """

    :param session:
    :param url:
    :return:
    """
    headers = {'content-type': 'application/json'}
    params = {'id': 2, 'username': 'test_user_1', 'password': '123456', 'name': '你是谁'}
    async with session.get(url, json=params, headers=headers) as response:
        r = await response.json(content_type=None)
    return r


async def post_json(session, url):
    # 传输json格式的数据
    headers = {'content-type': 'application/json'}
    params = {'name': '关', 'age': '33'}
    async with session.post(url, json=params, headers=headers) as response:
        r = await response.json(content_type=None)
    return r

async def post_img(session, url, file_path=None):
    """

    :param session:
    :param url:
    :return:
    """

    # with open(file_path,'rb') as f:
    #     # trans to bytes
    #     image_byte=b64encode(f.read())
    # image_str = image_byte.decode('ascii')
    # async with session.post(url, json={'img': image_str}) as response:
    #     r = await response.json(content_type=None)
    text = requests.post(url, files={'file': open(file_path, 'rb')})
    return text.json()



async def main():
    async with aiohttp.ClientSession() as session:
        # response_json = await check_register(session, 'http://127.0.0.1:8080/system/register')
        # print("通过get_json返回的数据:", response_json['data'])
        response_json = await check_login(session, 'http://127.0.0.1:8080/login')
        print("通过get_json返回的数据:", response_json['data'])
        # response_json = await check_del(session, 'http://127.0.0.1:8080/system/delete')
        # print("通过get_json返回的数据:", response_json['data'])
        # response_json = await check_del(session, 'http://127.0.0.1:8080/show_all')
        # print("通过get_json返回的数据:", response_json['data'])
        # response_json = await post_img(session, 'http://127.0.0.1:8080/system/pre_img', file_path='../test_tra_1.png')
        # print("通过post_json返回的数据:", response_json)

        await session.close()


class cli:
    def __init__(self):
        self.num = 10

if __name__ == '__main__':
    # loop = asyncio.get_event_loop()
    # loop.run_until_complete(main())
    pass