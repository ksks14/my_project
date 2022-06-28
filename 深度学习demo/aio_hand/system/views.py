from aiohttp import web
from asyncio import get_event_loop
from system.models_set import detect

async def predict(request):
    """

    :param request:
    :return:
    """
    # get the file
    form = await request.post()
    # load the image data
    filed = form['file']
    file_byte = filed.file.read()
    executor = request.app['executor']
    loop = get_event_loop()
    r = loop.run_in_executor
    # get the res
    res = await r(executor, detect, file_byte)

    return web.Response(body=res, headers={'allow_headers': '*'})