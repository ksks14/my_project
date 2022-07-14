import aiohttp
import requests
import asyncio


async def post_img(session, url, file_path=None):
    """

    :param session:
    :param url:
    :param file_path:
    :return:
    """
    text = requests.post(url, files={'file': open(file_path, 'rb')})
    return text.json()


async def main():
    """

    :return:
    """
    file_path = '../data/img/test_tra_1.png'
    async with aiohttp.ClientSession() as session:
        response_json = await post_img(session, 'http://175.178.65.191:8083/predict', file_path=file_path)
        print('res: ', response_json)

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())