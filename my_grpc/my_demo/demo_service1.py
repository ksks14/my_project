from concurrent import futures

import os
import grpc
import demo1_pb2
import demo1_pb2_grpc


if os.environ.get('https_proxy'):
    del os.environ['https_proxy']
if os.environ.get('http_proxy'):
    del os.environ['http_proxy']

class Demo1(demo1_pb2_grpc.demo1_serviceServicer):
    # 继承服务

    def hello(self, request, context):
        return demo1_pb2.response(res=request.data)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    demo1_pb2_grpc.add_demo1_serviceServicer_to_server(Demo1(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    serve()