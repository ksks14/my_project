#!/usr/bin/env python3
# -*- coding: utf-8 -*-
' a test aiohttp server '

__author__ = 'TianJiang Gui'

import asyncio
from aiohttp import web


async def index(request):
    await asyncio.sleep(0.5)
    return web.Response(body=b'<h1>Index</h1>', content_type='text/html')


async def get_json(request):
    data = await request.json()
    name = data["name"]
    print('请求get的信息data为: %s' % str(data))
    try:
        return web.json_response({'code': 0, 'data': name})
    except Exception as e:
        return web.json_response({'msg': e.value}, status=500)


async def post_json(request):
    # put或者post方式参数获取
    data = await request.json()
    name = data["name"]
    print('请求post的信息data为: %s' % str(data))
    try:
        return web.json_response({'code': 0, 'data': name})
    except Exception as e:
        return web.json_response({'msg': e.value}, status=500)


if __name__ == '__main__':
    app = web.Application()
    app.add_routes([web.get('/', index),
                    web.get('/getjson', get_json),
                    web.post('/postjson', post_json)])

    web.run_app(app, host='127.0.0.1', port=8100)