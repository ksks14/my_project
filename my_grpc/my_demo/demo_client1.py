import grpc

import logging

import demo1_pb2
import demo1_pb2_grpc


def run():
    """

    """
    with grpc.insecure_channel('localhost:8081') as channel:
        stub = demo1_pb2_grpc.demo1_serviceStub(channel=channel)
        reponse = stub.hello(demo1_pb2.request(data=10))
        print('client: %s', reponse.res)


if __name__ == '__main__':
    run()
