from aiohttp import web, ClientSession
from db import user, login_check, register_in, delete_data
from aiohttp_jinja2 import template
import asyncio

from utils.utils import predict

@template('index.html')
async def index(request):
    """

    :param request:
    :return:
    """
    async with request.app['db'].acquire() as conn:
        cursor = await conn.execute(user.select())
        records = await cursor.fetchall()
    return {'users': [dict(p) for p in records]}


async def show_all(request):
    """

    :param request:
    :return:
    """
    res = []
    async with request.app['db'].acquire() as conn:
        async for row in conn.execute(user.select()):
           res.append([i for i in row.values()])
    return web.json_response(data={'code': 0, 'data': res})


async def login(request):
    """

    :param request:
    :return:
    """
    # get the sql engine
    async with request.app['db'].acquire() as conn:
        # get the data
        data = await request.json()
        res = await login_check(conn=conn, data=data)
        if res:
            # return web.json_response({'code': 0, 'data': [i for i in res.values()]})
            # return web.json_response({'code': 0, 'data': 'success'})
            return web.json_response({'code': 0, 'data': 'success'})
        else:
            return web.json_response({'code': 0, 'data': 'please input correct username and password!'})
    return web.json_response({'code': 0})


async def register(request):
    """
    use the get
    :param request:
    :return:
    """
    # get the data
    async with request.app['db'].acquire() as conn:
        try:
            data = await request.json()
        except Exception as e:
            # error
            raise web.HTTPException(text=str(e))
        try:
            await register_in(conn, data)
            return web.json_response({'code': 0, 'data': '注册成功'})
        except Exception as e:
            # error
            raise web.HTTPNotFound(text=str(e))



async def delete_user(request):
    """

    :param request:
    :return:
    """
    async with request.app['db'].acquire() as conn:
        try:
            data = await request.json()
        except Exception as e:
            raise web.HTTPException(text=str(e))
        try:
            await delete_data(conn=conn, data=data)
            return web.json_response(data={'code': 0, 'data': '删除成功'})
        except Exception as e:
            return web.json_response(data={'code': 1, 'data': str(e)})


async def pre_img(request):
    """
    get the img from the client
    :param request:
    :return:
    """
    # print('in the server')
    form = await request.post()
    # load the image data
    filed = form['file']
    file_byte = filed.file.read()
    executor = request.app['executor']
    loop = asyncio.get_event_loop()
    r = loop.run_in_executor
    res = await r(executor, predict, file_byte)
    return web.json_response(data={'code': 0, 'data': 'test'})





