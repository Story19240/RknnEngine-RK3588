#include "rknn_wrapper.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <sstream>
#include <utility>

#include "rknn_yolov8_output_processor.h"

namespace rknn {
namespace {

double elapsed_millis(
        const std::chrono::steady_clock::time_point& start,
        const std::chrono::steady_clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

size_t tensor_element_count(const rknn_tensor_attr& attr) {
    if (attr.n_elems > 0) {
        return static_cast<size_t>(attr.n_elems);
    }

    size_t count = 1;
    const size_t dimensions = std::min<size_t>(attr.n_dims, RKNN_MAX_DIMS);
    for (size_t index = 0; index < dimensions; ++index) {
        count *= static_cast<size_t>(attr.dims[index]);
    }
    return count;
}

RknnTensorInfo tensor_info(const rknn_tensor_attr& attr) {
    RknnTensorInfo info;
    info.index = attr.index;
    info.name = attr.name;
    info.format = attr.fmt;
    info.type = attr.type;
    info.quantization_type = attr.qnt_type;
    info.zero_point = attr.zp;
    info.scale = attr.scale;

    const size_t dimensions = std::min<size_t>(attr.n_dims, RKNN_MAX_DIMS);
    info.shape.reserve(dimensions);
    for (size_t index = 0; index < dimensions; ++index) {
        info.shape.push_back(attr.dims[index]);
    }
    return info;
}

bool infer_input_dimensions(
        const rknn_tensor_attr& attr,
        int& width,
        int& height,
        uint32_t& channels) {
    if (attr.n_dims < 4) {
        return false;
    }

    if (attr.fmt == RKNN_TENSOR_NCHW) {
        channels = attr.dims[1];
        height = static_cast<int>(attr.dims[2]);
        width = static_cast<int>(attr.dims[3]);
        return width > 0 && height > 0 && channels > 0;
    }

    if (attr.fmt == RKNN_TENSOR_NHWC) {
        height = static_cast<int>(attr.dims[1]);
        width = static_cast<int>(attr.dims[2]);
        channels = attr.dims[3];
        return width > 0 && height > 0 && channels > 0;
    }

    if (attr.dims[1] == 3) {
        channels = attr.dims[1];
        height = static_cast<int>(attr.dims[2]);
        width = static_cast<int>(attr.dims[3]);
        return width > 0 && height > 0;
    }

    return false;
}

}

int RknnLetterboxPreprocessor::preprocess(
        const cv::Mat& source,
        int input_width,
        int input_height,
        cv::Mat& output,
        RknnFrameInfo& frame_info) {
    if (source.empty() || source.type() != CV_8UC3 || input_width <= 0 || input_height <= 0) {
        return -1;
    }

    const float scale = std::min(
            static_cast<float>(input_width) / static_cast<float>(source.cols),
            static_cast<float>(input_height) / static_cast<float>(source.rows));
    int resized_width = std::max(1, static_cast<int>(std::lround(source.cols * scale)));
    int resized_height = std::max(1, static_cast<int>(std::lround(source.rows * scale)));
    resized_width = std::min(resized_width, input_width);
    resized_height = std::min(resized_height, input_height);

    const int padding_left = (input_width - resized_width) / 2;
    const int padding_top = (input_height - resized_height) / 2;

    output.setTo(cv::Scalar(114, 114, 114));
    cv::Mat target = output(cv::Rect(padding_left, padding_top, resized_width, resized_height));
    cv::resize(source, target, cv::Size(resized_width, resized_height), 0.0, 0.0, cv::INTER_LINEAR);

    frame_info.source_width = source.cols;
    frame_info.source_height = source.rows;
    frame_info.input_width = input_width;
    frame_info.input_height = input_height;
    frame_info.resize_scale = scale;
    frame_info.padding_left = padding_left;
    frame_info.padding_top = padding_top;
    return 0;
}

int RknnPassThroughOutputProcessor::process(
        const RknnFrameInfo&,
        const std::vector<RknnTensorView>&,
        RknnInferenceResult&) {
    return 0;
}

RknnWrapper::RknnWrapper()
    : preprocessor_(std::make_shared<RknnLetterboxPreprocessor>()),
      output_processor_(std::make_shared<RknnPassThroughOutputProcessor>()) {
    std::memset(&input_attr_, 0, sizeof(input_attr_));
}

RknnWrapper::~RknnWrapper() {
    release();
}

int RknnWrapper::load(const char* model_path, const std::string& model_name) {
    if (!model_path) {
        return -1;
    }

    std::ifstream stream(model_path, std::ios::binary);
    if (!stream.is_open()) {
        return -1;
    }

    std::vector<char> data(
            (std::istreambuf_iterator<char>(stream)),
            std::istreambuf_iterator<char>());
    if (data.empty()) {
        return -1;
    }
    return load(data.data(), data.size(), model_name);
}

int RknnWrapper::load(
        const void* model_data,
        size_t model_size,
        const std::string& model_name) {
    return loadInternal(model_data, model_size, model_name, nullptr);
}

int RknnWrapper::load(
        const void* model_data,
        size_t model_size,
        const std::string& model_name,
        const RknnDetectionConfig& detection_config) {
    return loadInternal(model_data, model_size, model_name, &detection_config);
}

int RknnWrapper::loadInternal(
        const void* model_data,
        size_t model_size,
        const std::string& model_name,
        const RknnDetectionConfig* detection_config) {
    release();
    if (!model_data || model_size == 0) {
        return -1;
    }

    model_data_.assign(
            static_cast<const char*>(model_data),
            static_cast<const char*>(model_data) + model_size);

    int return_code = rknn_init(
            &context_,
            model_data_.data(),
            static_cast<uint32_t>(model_data_.size()),
            0,
            nullptr);
    if (return_code < 0) {
        release();
        return return_code;
    }

    rknn_input_output_num tensor_count{};
    return_code = rknn_query(
            context_,
            RKNN_QUERY_IN_OUT_NUM,
            &tensor_count,
            sizeof(tensor_count));
    if (return_code < 0 || tensor_count.n_input != 1 || tensor_count.n_output == 0) {
        release();
        return return_code < 0 ? return_code : -2;
    }

    input_attr_.index = 0;
    return_code = rknn_query(
            context_,
            RKNN_QUERY_INPUT_ATTR,
            &input_attr_,
            sizeof(input_attr_));
    if (return_code < 0) {
        release();
        return return_code;
    }

    int input_width = 0;
    int input_height = 0;
    uint32_t input_channels = 0;
    if (!infer_input_dimensions(
                input_attr_,
                input_width,
                input_height,
                input_channels) ||
            input_channels != 3) {
        release();
        return -3;
    }

    output_attrs_.resize(tensor_count.n_output);
    output_infos_.reserve(tensor_count.n_output);
    output_buffers_.reserve(tensor_count.n_output);
    for (uint32_t index = 0; index < tensor_count.n_output; ++index) {
        rknn_tensor_attr& attr = output_attrs_[index];
        std::memset(&attr, 0, sizeof(attr));
        attr.index = index;
        return_code = rknn_query(
                context_,
                RKNN_QUERY_OUTPUT_ATTR,
                &attr,
                sizeof(attr));
        if (return_code < 0) {
            release();
            return return_code;
        }

        const size_t element_count = tensor_element_count(attr);
        if (element_count == 0 ||
            element_count > std::numeric_limits<uint32_t>::max() / sizeof(float)) {
            release();
            return -4;
        }
        output_infos_.push_back(tensor_info(attr));
        output_buffers_.emplace_back(element_count, 0.0f);
    }

    rknn_sdk_version sdk_version{};
    if (rknn_query(
                context_,
                RKNN_QUERY_SDK_VERSION,
                &sdk_version,
                sizeof(sdk_version)) == 0) {
        api_version_ = sdk_version.api_version;
        driver_version_ = sdk_version.drv_version;
    }

    input_buffer_ = cv::Mat(
            input_height,
            input_width,
            CV_8UC3,
            cv::Scalar(114, 114, 114));
    model_name_ = model_name;
    if (detection_config) {
        output_processor_ = std::make_shared<RknnYolov8OutputProcessor>(*detection_config);
        include_raw_outputs_ = detection_config->include_raw_outputs;
    } else {
        output_processor_ = std::make_shared<RknnPassThroughOutputProcessor>();
        include_raw_outputs_ = true;
    }
    return 0;
}

void RknnWrapper::set_preprocessor(std::shared_ptr<RknnPreprocessor> preprocessor) {
    preprocessor_ = preprocessor
            ? std::move(preprocessor)
            : std::make_shared<RknnLetterboxPreprocessor>();
}

void RknnWrapper::set_output_processor(std::shared_ptr<RknnOutputProcessor> processor) {
    output_processor_ = processor
            ? std::move(processor)
            : std::make_shared<RknnPassThroughOutputProcessor>();
}

int RknnWrapper::infer(const cv::Mat& rgb, RknnInferenceResult& result) {
    result = RknnInferenceResult{};
    if (!is_loaded() || rgb.empty() || rgb.type() != CV_8UC3) {
        return -1;
    }

    const auto total_start = std::chrono::steady_clock::now();
    result.model_name = model_name_;
    result.frame.source_width = rgb.cols;
    result.frame.source_height = rgb.rows;
    result.frame.input_width = input_buffer_.cols;
    result.frame.input_height = input_buffer_.rows;
    result.include_raw_outputs = include_raw_outputs_;

    const int preprocess_return = preprocessor_->preprocess(
            rgb,
            input_buffer_.cols,
            input_buffer_.rows,
            input_buffer_,
            result.frame);
    const auto preprocess_end = std::chrono::steady_clock::now();
    result.preprocess_millis = elapsed_millis(total_start, preprocess_end);
    if (preprocess_return < 0 ||
        input_buffer_.empty() ||
        input_buffer_.type() != CV_8UC3 ||
        !input_buffer_.isContinuous()) {
        return preprocess_return < 0 ? preprocess_return : -2;
    }

    rknn_input input{};
    input.index = 0;
    input.buf = input_buffer_.data;
    input.size = static_cast<uint32_t>(input_buffer_.total() * input_buffer_.elemSize());
    input.pass_through = 0;
    input.type = RKNN_TENSOR_UINT8;
    input.fmt = RKNN_TENSOR_NHWC;

    int return_code = rknn_inputs_set(context_, 1, &input);
    if (return_code < 0) {
        return return_code;
    }

    const auto inference_start = std::chrono::steady_clock::now();
    return_code = rknn_run(context_, nullptr);
    const auto inference_end = std::chrono::steady_clock::now();
    result.inference_millis = elapsed_millis(inference_start, inference_end);
    if (return_code < 0) {
        return return_code;
    }

    std::vector<rknn_output> outputs(output_attrs_.size());
    for (size_t index = 0; index < output_attrs_.size(); ++index) {
        rknn_output& output = outputs[index];
        std::memset(&output, 0, sizeof(output));
        output.want_float = 1;
        output.is_prealloc = 1;
        output.index = output_attrs_[index].index;
        output.buf = output_buffers_[index].data();
        output.size = static_cast<uint32_t>(
                output_buffers_[index].size() * sizeof(float));
    }

    const auto postprocess_start = std::chrono::steady_clock::now();
    return_code = rknn_outputs_get(
            context_,
            static_cast<uint32_t>(outputs.size()),
            outputs.data(),
            nullptr);
    if (return_code < 0) {
        return return_code;
    }

    result.output_views.reserve(outputs.size());
    for (size_t index = 0; index < outputs.size(); ++index) {
        result.output_views.push_back(RknnTensorView{
                &output_infos_[index],
                output_buffers_[index].data(),
                output_buffers_[index].size()});
    }

    return_code = output_processor_->process(
            result.frame,
            result.output_views,
            result);
    rknn_outputs_release(
            context_,
            static_cast<uint32_t>(outputs.size()),
            outputs.data());
    const auto postprocess_end = std::chrono::steady_clock::now();
    result.postprocess_millis = elapsed_millis(postprocess_start, postprocess_end);
    result.total_millis = elapsed_millis(total_start, postprocess_end);
    return return_code;
}

bool RknnWrapper::is_loaded() const {
    return context_ != 0;
}

int RknnWrapper::input_width() const {
    return input_buffer_.cols;
}

int RknnWrapper::input_height() const {
    return input_buffer_.rows;
}

std::string RknnWrapper::model_info() const {
    std::ostringstream stream;
    stream << "model=" << model_name_
           << " input=" << input_buffer_.cols << "x" << input_buffer_.rows
           << " outputs=" << output_infos_.size()
           << " api=" << api_version_
           << " driver=" << driver_version_;
    return stream.str();
}

void RknnWrapper::release() {
    if (context_) {
        rknn_destroy(context_);
        context_ = 0;
    }
    model_data_.clear();
    model_name_.clear();
    api_version_.clear();
    driver_version_.clear();
    std::memset(&input_attr_, 0, sizeof(input_attr_));
    output_attrs_.clear();
    output_infos_.clear();
    output_buffers_.clear();
    input_buffer_.release();
}

}
