from time import sleep
from concurrent.futures import ThreadPoolExecutor


def print_th(index_1, index_2):
    print(index_2)
    sleep(2)

if __name__ == '__main__':
    executor = ThreadPoolExecutor(max_workers=10)
    for i in range(30):
        executor.submit(print_th, i, i+1)