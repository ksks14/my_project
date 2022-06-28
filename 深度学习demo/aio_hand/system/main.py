from aiohttp import web
from system.routes import setup_routes
from asyncio import SelectorEventLoop
from system.models_set import init_models

async def init_app():
    """

    :return:
    """
    # create app
    app = web.Application()
    # setup_routes
    setup_routes(app=app)
    # init models
    await init_models(app=app)
    return app


def run_app(host='127.0.0.1', port='8806'):
    """

    :param host:
    :param port:
    :return:
    """
    # create loop
    loop = SelectorEventLoop()
    app = loop.run_until_complete(init_app(), )
    web.run_app(app=app, loop=loop, host=host, port=int(port))

if __name__ == '__main__':
    run_app()
