import logging
import time
from concurrent import futures

import grpc

from rtp_llm.config.engine_config import EngineConfig
from rtp_llm.config.log_config import setup_logging
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    CacheStatusPB,
    CacheVersionPB,
    MMPreprocessConfigPB,
    MultimodalInputsPB,
    MultimodalOutputPB,
    MultimodalOutputsPB,
    StatusVersionPB,
    WorkerStatusPB,
)
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc import (
    MultimodalRpcServiceServicer,
    add_MultimodalRpcServiceServicer_to_server,
)
from rtp_llm.distribute.distributed_server import DistributedServer, get_world_info
from rtp_llm.distribute.worker_info import g_worker_info
from rtp_llm.model_factory import ModelFactory
from rtp_llm.multimodal.mm_process_engine import MMEmbeddingRes, MMProcessEngine
from rtp_llm.server.server_args.server_args import setup_args
from rtp_llm.utils.base_model_datatypes import MMPreprocessConfig, MultimodalInput
from rtp_llm.utils.grpc_util import trans_from_tensor, trans_tensor


def trans_output(res: MMEmbeddingRes):
    t0 = time.perf_counter()
    output_pb = MultimodalOutputsPB()
    contain_pos = (res.position_ids is not None) and (len(res.position_ids) > 0)
    contain_deepstack = (res.deepstack_embeds is not None) and (
        len(res.deepstack_embeds) > 0
    )
    for i in range(len(res.embeddings)):
        logging.info(f"embedding[{i}].shape = {res.embeddings[i].shape}")
        if contain_deepstack:
            logging.info(
                f"deepstack_embeds[{i}.shape = {res.deepstack_embeds[i].shape}]"
            )
        output = MultimodalOutputPB(
            multimodal_embedding=trans_from_tensor(res.embeddings[i]),
            multimodal_pos_id=(
                trans_from_tensor(res.position_ids[i]) if contain_pos else None
            ),
            multimodal_deepstack_embeds=(
                trans_from_tensor(res.deepstack_embeds[i])
                if contain_deepstack
                else None
            ),
        )
        output_pb.multimodal_outputs.append(output)
    t1 = time.perf_counter()
    # 计算序列化后的大小
    # serialized_size = output_pb.ByteSize()
    serialized_size = 0
    logging.info(
        f"[VIT_GRPC_TIMING] trans_output: elapsed: {(t1-t0)*1e6:.2f}us, "
        f"output_count: {len(res.embeddings)}, serialized_size: {serialized_size} bytes ({serialized_size/1024/1024:.2f} MB)"
    )
    return output_pb


class MultimodalRpcServer(MultimodalRpcServiceServicer):
    def __init__(self, mm_process_engine: MMProcessEngine):
        self.engine = mm_process_engine

    def RemoteMultimodalEmbedding(self, multimodal_inputs: MultimodalInputsPB, context):
        logging.info(f"[VIT_GRPC_TIMING] RemoteMultimodalEmbedding: start www")
        t0 = time.perf_counter()
        # 计算输入大小
        input_size = multimodal_inputs.ByteSize()
        t_embedding_start = time.perf_counter()
        res: MMEmbeddingRes = self.engine.mm_embedding_rpc(multimodal_inputs)
        t_embedding_end = time.perf_counter()
        t_trans_start = time.perf_counter()
        output_pb = trans_output(res)
        t_trans_end = time.perf_counter()
        t_final = time.perf_counter()
        # 计算输出大小
        # output_size = output_pb.ByteSize()
        output_size = 0
        logging.info(
            f"[VIT_GRPC_TIMING] RemoteMultimodalEmbedding: total: {(t_final-t0)*1e6:.2f}us, "
            f"embedding: {(t_embedding_end-t_embedding_start)*1e6:.2f}us, "
            f"trans_output: {(t_trans_end-t_trans_start)*1e6:.2f}us, "
            f"input_size: {input_size} bytes ({input_size/1024/1024:.2f} MB), "
            f"output_size: {output_size} bytes ({output_size/1024/1024:.2f} MB)"
        )
        return output_pb

    def GetWorkerStatus(self, request: StatusVersionPB, context):
        worker_status = WorkerStatusPB()
        worker_status.role = "VIT"
        worker_status.status_version = 1
        worker_status.alive = True
        return worker_status

    def GetCacheStatus(self, request: CacheVersionPB, context):
        return CacheStatusPB()

    def stop(self):
        self.engine.stop()


def create_rpc_server():
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=200),
        options=[
            ("grpc.max_send_message_length", 1024 * 1024 * 1024),
            ("grpc.max_receive_message_length", 1024 * 1024 * 1024),
            ("grpc.max_concurrent_streams", -1),
            ("grpc.http2.min_ping_interval_without_data_ms", 1000),
            ("grpc.http2.max_ping_strikes", 1000),
        ],
    )
    return server
