from aiohttp import web
from asyncio import get_event_loop, gather, shield
from concurrent.futures import ProcessPoolExecutor
from utils.utils import clean_model_with_signal, load_model_with_signal



async def init_models(app: web.Application, ) -> ProcessPoolExecutor:
    """

    :param app:
    :return:
    """
    conf = app['conf']['workers']
    # get the model path
    model_path = conf['model_path']
    # get the num_workers
    max_workers = conf['max_workers']
    # set the max executors num
    executor = ProcessPoolExecutor(max_workers=max_workers)
    # create the loop
    loop = get_event_loop()
    #
    run = loop.run_in_executor
    # set the executor in runing
    fs = [run(executor, load_model_with_signal, model_path) for i in range(max_workers)]
    await gather(*fs)

    async def close_executor(app: web.Application, ) -> None:
        # executor
        # set the cleaning model in executor
        fs = [run(executor, clean_model_with_signal) for i in range(max_workers)]
        # protect the fs from being cancelled
        await shield(gather(*fs))
        # until all events done
        executor.shutdown(wait=True)

    app.on_cleanup.append(close_executor)
    app['executor'] = executor
    return executor

