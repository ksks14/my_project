from aiohttp import web
from system.routes import setup_routes
from asyncio import SelectorEventLoop
from utils.model import init_models


async def init_app():
    """
    init the app
    :return: async object
    """
    # create app
    app = web.Application()
    # setup_routes
    setup_routes(app=app)
    # init model
    await init_models(app=app)
    return app


def get_app():
    """

    :return: app
    """
    loop = SelectorEventLoop()
    app = loop.run_until_complete(init_app(), )
    # web.run_app(app=app, loop=loop, host='127.0.0.1')
    # put on the internet
    web.run_app(app=app, loop=loop, host='127.0.0.1')


