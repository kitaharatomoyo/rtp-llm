#pragma once

#include <functional>
#include <algorithm>
#include <string>
#include <vector>
#include <torch/python.h>
#include "absl/status/statusor.h"
#include "rtp_llm/cpp/multimodal_processor/MultimodalTypes.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/cpp/utils/StatusUtil.h"
#include "rtp_llm/cpp/pybind/PyUtils.h"
#include "rtp_llm/cpp/model_rpc/RPCPool.h"
#include "rtp_llm/cpp/multimodal_processor/MultimodalProcessor.h"
#include "rtp_llm/cpp/model_rpc/QueryConverter.h"
#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "aios/autil/autil/TimeUtility.h"

namespace py = pybind11;

namespace rtp_llm {

class RemoteMultimodalProcessor: public MultimodalProcessor {
public:
    RemoteMultimodalProcessor(const MMModelConfig& mm_model_config, int64_t max_seq_len):
        MultimodalProcessor(py::none(), mm_model_config, max_seq_len) {}

private:
    MultimodalRpcPool pool_;
    std::string       vit_cluster_name_;

    ErrorResult<MultimodalOutput> MultimodalEmbedding(const std::vector<rtp_llm::MultimodalInput> mm_inputs,
                                                      std::string                                 ip_port = "") {
        if (ip_port == "") {
            return ErrorInfo(ErrorCode::MM_NOT_SUPPORTED_ERROR, "ip:port is empty in remote multimodal processing");
        }
        RTP_LLM_LOG_INFO("[LLM_GRPC_TIMING] RemoteMultimodalEmbedding start vvv");
        int64_t t0                = autil::TimeUtility::currentTimeInMicroSeconds();
        auto    connection_status = pool_.getConnection(ip_port);
        if (!connection_status.ok()) {
            return ErrorInfo(ErrorCode::MM_EMPTY_ENGINE_ERROR, connection_status.status().ToString());
        }
        auto& connection = connection_status.value();

        int64_t            t_trans_input_start = autil::TimeUtility::currentTimeInMicroSeconds();
        MultimodalInputsPB input_pb            = QueryConverter::transMMInputsPB(mm_inputs);
        int64_t            t_trans_input_end   = autil::TimeUtility::currentTimeInMicroSeconds();
        int64_t            input_size          = input_pb.ByteSize();

        auto                stub = connection.stub;
        MultimodalOutputsPB output_pb;
        grpc::ClientContext context;
        int64_t             t_grpc_start = autil::TimeUtility::currentTimeInMicroSeconds();
        auto                status       = stub->RemoteMultimodalEmbedding(&context, input_pb, &output_pb);
        int64_t             t_grpc_end   = autil::TimeUtility::currentTimeInMicroSeconds();
        if (!status.ok()) {
            return ErrorInfo(ErrorCode::MM_PROCESS_ERROR, status.error_message());
        }
        // int64_t output_size = output_pb.ByteSize();
        int64_t output_size = 0;

        int64_t t_trans_output_start = autil::TimeUtility::currentTimeInMicroSeconds();
        auto    result               = QueryConverter::transMMOutput(&output_pb);
        int64_t t_trans_output_end   = autil::TimeUtility::currentTimeInMicroSeconds();
        int64_t t_final              = autil::TimeUtility::currentTimeInMicroSeconds();

        RTP_LLM_LOG_INFO("[LLM_GRPC_TIMING] RemoteMultimodalEmbedding: total: %ld us, "
                         "trans_input: %ld us, grpc_call: %ld us, trans_output: %ld us, "
                         "input_size: %ld bytes (%.2f MB), output_size: %ld bytes (%.2f MB)",
                         t_final - t0,
                         t_trans_input_end - t_trans_input_start,
                         t_grpc_end - t_grpc_start,
                         t_trans_output_end - t_trans_output_start,
                         input_size,
                         input_size / 1024.0 / 1024.0,
                         output_size,
                         output_size / 1024.0 / 1024.0);
        return result;
    }
};

}  // namespace rtp_llm
