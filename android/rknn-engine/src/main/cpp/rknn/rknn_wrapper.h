#ifndef RKNN_ENGINE_RKNN_WRAPPER_H
#define RKNN_ENGINE_RKNN_WRAPPER_H

#include <opencv2/core.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rknn_api.h"

namespace rknn {

struct RknnTensorInfo {
    uint32_t index = 0;
    std::string name;
    std::vector<uint32_t> shape;
    rknn_tensor_format format = RKNN_TENSOR_UNDEFINED;
    rknn_tensor_type type = RKNN_TENSOR_FLOAT32;
    rknn_tensor_qnt_type quantization_type = RKNN_TENSOR_QNT_NONE;
    int32_t zero_point = 0;
    float scale = 0.0f;
};

struct RknnTensorView {
    const RknnTensorInfo* info = nullptr;
    const float* values = nullptr;
    size_t element_count = 0;
};

struct RknnFrameInfo {
    int source_width = 0;
    int source_height = 0;
    int input_width = 0;
    int input_height = 0;
    float resize_scale = 1.0f;
    int padding_left = 0;
    int padding_top = 0;
};

struct RknnDetection {
    int class_id = 0;
    std::string class_name;
    float confidence = 0.0f;
    float left = 0.0f;
    float top = 0.0f;
    float right = 0.0f;
    float bottom = 0.0f;
};

struct RknnDetectionConfig {
    float confidence_threshold = 0.25f;
    float nms_threshold = 0.45f;
    int max_detections = 300;
    int reg_max = 16;
    int num_classes = 0;
    bool include_raw_outputs = false;
    std::vector<int> strides{8, 16, 32};
    std::vector<std::string> class_names;
};

struct RknnInferenceResult {
    std::string model_name;
    RknnFrameInfo frame;
    std::vector<RknnTensorView> output_views;
    std::vector<RknnDetection> detections;
    bool include_raw_outputs = true;
    double preprocess_millis = 0.0;
    double inference_millis = 0.0;
    double postprocess_millis = 0.0;
    double total_millis = 0.0;
};

class RknnPreprocessor {
public:
    virtual ~RknnPreprocessor() = default;

    virtual int preprocess(
            const cv::Mat& source,
            int input_width,
            int input_height,
            cv::Mat& output,
            RknnFrameInfo& frame_info) = 0;
};

class RknnLetterboxPreprocessor final : public RknnPreprocessor {
public:
    int preprocess(
            const cv::Mat& source,
            int input_width,
            int input_height,
            cv::Mat& output,
            RknnFrameInfo& frame_info) override;
};

class RknnOutputProcessor {
public:
    virtual ~RknnOutputProcessor() = default;

    virtual int process(
            const RknnFrameInfo& frame_info,
            const std::vector<RknnTensorView>& outputs,
            RknnInferenceResult& result) = 0;
};

class RknnPassThroughOutputProcessor final : public RknnOutputProcessor {
public:
    int process(
            const RknnFrameInfo& frame_info,
            const std::vector<RknnTensorView>& outputs,
            RknnInferenceResult& result) override;
};

class RknnWrapper {
public:
    RknnWrapper();
    ~RknnWrapper();

    RknnWrapper(const RknnWrapper&) = delete;
    RknnWrapper& operator=(const RknnWrapper&) = delete;

    int load(const char* model_path, const std::string& model_name);
    int load(const void* model_data, size_t model_size, const std::string& model_name);
    int load(
            const void* model_data,
            size_t model_size,
            const std::string& model_name,
            const RknnDetectionConfig& detection_config);

    void set_preprocessor(std::shared_ptr<RknnPreprocessor> preprocessor);
    void set_output_processor(std::shared_ptr<RknnOutputProcessor> processor);

    int infer(const cv::Mat& rgb, RknnInferenceResult& result);

    bool is_loaded() const;
    int input_width() const;
    int input_height() const;
    std::string model_info() const;
    void release();

private:
    rknn_context context_ = 0;
    std::vector<char> model_data_;
    std::string model_name_;
    std::string api_version_;
    std::string driver_version_;

    rknn_tensor_attr input_attr_{};
    std::vector<rknn_tensor_attr> output_attrs_;
    std::vector<RknnTensorInfo> output_infos_;
    std::vector<std::vector<float>> output_buffers_;
    cv::Mat input_buffer_;
    bool include_raw_outputs_ = true;

    std::shared_ptr<RknnPreprocessor> preprocessor_;
    std::shared_ptr<RknnOutputProcessor> output_processor_;

    int loadInternal(
            const void* model_data,
            size_t model_size,
            const std::string& model_name,
            const RknnDetectionConfig* detection_config);
};

}

#endif
