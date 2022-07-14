# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import demo1_pb2 as demo1__pb2


class demo1_serviceStub(object):
    """定义服务
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.hello = channel.unary_unary(
                '/demo1_service/hello',
                request_serializer=demo1__pb2.request.SerializeToString,
                response_deserializer=demo1__pb2.response.FromString,
                )


class demo1_serviceServicer(object):
    """定义服务
    """

    def hello(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_demo1_serviceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'hello': grpc.unary_unary_rpc_method_handler(
                    servicer.hello,
                    request_deserializer=demo1__pb2.request.FromString,
                    response_serializer=demo1__pb2.response.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'demo1_service', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class demo1_service(object):
    """定义服务
    """

    @staticmethod
    def hello(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/demo1_service/hello',
            demo1__pb2.request.SerializeToString,
            demo1__pb2.response.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
