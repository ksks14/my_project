from aiohttp import web
import aiohttp_cors
from routes import setup_routes
from settings import config, BASE_DIR
from db import pg_context
from asyncio import SelectorEventLoop
from utils.model import init_models

import aiohttp_jinja2
import jinja2


async def init_app():
    """
    init the app
    :return:
    """
    # create a app
    app = web.Application()
    # solve the cors
    # cors = aiohttp_cors.setup(app, defaults={
    #     "*": aiohttp_cors.ResourceOptions(
    #         allow_credentials=True,
    #         expose_headers="*",
    #         allow_headers="*",
    #     )
    # })

    # setup the route
    setup_routes(app=app)
    app['conf'] = config
    aiohttp_jinja2.setup(app, loader=jinja2.FileSystemLoader(str(BASE_DIR / 'system' / 'templates')))
    # link the database
    app.cleanup_ctx.append(pg_context)
    # init the model, make the executor to use the model
    # executor = await init_models(app=app)
    await init_models(app=app)
    return app



async def get_app():
    """
    return the app
    :return:
    """
    app = await init_app()
    return app






if __name__ == '__main__':
    # create loop
    loop = SelectorEventLoop()
    # get the app and add it to the loop
    app = loop.run_until_complete(init_app(), )
    # start the app service
    web.run_app(app=app, loop=loop, host='127.0.0.1')
