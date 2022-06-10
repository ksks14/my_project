import re
import os
import requests
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path


def event(ts_url_id, ts_url, ts_len):
    """

    :param ts_url_id:
    :param ts_url:
    :param ts_len:
    :return:
    """

    response = requests.get(url=ts_url, headers=headers, timeout=(5, 30))
    if response.status_code == 200:
        ts_content = response.content
    else:
        print(ts_url + '访问出错')
        raise ValueError('访问出错')
    ts_file_name = 'ts/{}.ts'.format(ts_url_id)
    with open(ts_file_name, 'wb') as fp:
        fp.write(ts_content)  # 将ts数据写入文件
        print('{0}/{1} 下载完成'.format(ts_url_id, ts_len))


def download_ts_files(ts_url_list=None):
    """

    :param ts_url_list:
    :return:
    """
    print('start downloading············')
    if not os.path.exists('./ts'):
        os.mkdir('./ts')
    # 异步进行，保证效率，调用线程池
    executor = ThreadPoolExecutor(max_workers=5)
    print('创建线程池完成')
    ts_len = len(ts_url_list)
    for ts_url_id in range(ts_len):
        print('{0}/{1} 开始下载'.format(ts_url_id, ts_len))
        executor.submit(event, ts_url_id, ts_url_list[ts_url_id], ts_len)
    executor.shutdown(wait=True)

    # add to mp4
    if not os.path.exists('./mp4'):
        os.mkdir('./mp4')
    mp4_file_path = './mp4/demo.mp4'
    for ts_url_id in range(ts_len):
        ts_file_name = Path('ts/{}.ts'.format(ts_url_id))

        if ts_file_name.exists():
            with open(ts_file_name, 'rb') as fp:
                ts_content = fp.read()  # 读取ts数据
            with open(mp4_file_path, 'ab') as fp:
                print('writing ' + str(ts_file_name) + '······')
                fp.write(ts_content)  # 将ts数据追加写入文件


# set the url
m3u8_url = 'https://v3.cdtlas.com/20220517/mEv29ASn/1100kb/hls/index.m3u8'
# set header
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:97.0) Gecko/20100101 Firefox/97.0',
}

# get m3u8
m3u8_file = requests.get(url=m3u8_url, headers=headers).text

# get ts_files
ts_url_list = re.findall(',\n(.*?)\n#', m3u8_file)

# download ts_files,
download_ts_files(ts_url_list)
