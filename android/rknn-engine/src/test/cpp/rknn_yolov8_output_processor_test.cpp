#include "rknn/rknn_yolov8_output_processor.h"

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

constexpr size_t kAnchors = 84;
constexpr size_t kClasses = 5;
constexpr size_t kRegChannels = 64;

void require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "[FAIL] " << message << "\n";
        std::exit(1);
    }
}

void require_close(
        float actual,
        float expected,
        float tolerance,
        const std::string& message) {
    if (std::fabs(actual - expected) > tolerance) {
        std::cerr << "[FAIL] " << message
                  << ": expected " << expected
                  << ", got " << actual << "\n";
        std::exit(1);
    }
}

struct TestCase {
    std::vector<float> reg;
    std::vector<float> cls;
};

struct OutputSet {
    std::shared_ptr<rknn::RknnTensorInfo> reg_info;
    std::shared_ptr<rknn::RknnTensorInfo> cls_info;
    std::shared_ptr<std::vector<float>> reg_values;
    std::shared_ptr<std::vector<float>> cls_values;
    std::vector<rknn::RknnTensorView> views;
};

TestCase make_test_case(bool same_class) {
    TestCase test_case;
    test_case.reg.assign(kAnchors * kRegChannels, -10.0f);
    test_case.cls.assign(kAnchors * kClasses, -10.0f);

    const std::vector<std::pair<size_t, float>> anchor_distances = {
            {0, 0.0f},
            {1, 0.0f},
    };
    for (const auto& anchor_distance : anchor_distances) {
        const size_t anchor = anchor_distance.first;
        test_case.reg[(0 * 16 + 0) * kAnchors + anchor] = 10.0f;
        test_case.reg[(1 * 16 + 0) * kAnchors + anchor] = 10.0f;
        test_case.reg[(2 * 16 + (anchor == 0 ? 3 : 2)) * kAnchors + anchor] = 10.0f;
        test_case.reg[(3 * 16 + 3) * kAnchors + anchor] = 10.0f;
    }

    test_case.cls[0 * kAnchors + 0] = 10.0f;
    test_case.cls[(same_class ? 0 : 1) * kAnchors + 1] = 9.0f;
    return test_case;
}

rknn::RknnDetectionConfig make_config() {
    rknn::RknnDetectionConfig config;
    config.confidence_threshold = 0.9f;
    config.nms_threshold = 0.45f;
    config.max_detections = 300;
    config.reg_max = 16;
    config.num_classes = static_cast<int>(kClasses);
    config.strides = {8, 16, 32};
    return config;
}

rknn::RknnFrameInfo make_frame_info() {
    rknn::RknnFrameInfo frame_info;
    frame_info.source_width = 64;
    frame_info.source_height = 64;
    frame_info.input_width = 64;
    frame_info.input_height = 64;
    frame_info.resize_scale = 1.0f;
    return frame_info;
}

rknn::RknnTensorInfo make_tensor_info(
        const std::string& name,
        rknn_tensor_format format,
        std::vector<uint32_t> shape) {
    rknn::RknnTensorInfo info;
    info.name = name;
    info.format = format;
    info.shape = std::move(shape);
    return info;
}

OutputSet make_nchw_outputs(
        const TestCase& test_case,
        bool swap_outputs) {
    OutputSet output_set;
    output_set.reg_info = std::make_shared<rknn::RknnTensorInfo>(make_tensor_info(
            "output_reg",
            RKNN_TENSOR_NCHW,
            {1, static_cast<uint32_t>(kRegChannels), static_cast<uint32_t>(kAnchors)}));
    output_set.cls_info = std::make_shared<rknn::RknnTensorInfo>(make_tensor_info(
            "output_cls",
            RKNN_TENSOR_NCHW,
            {1, static_cast<uint32_t>(kClasses), static_cast<uint32_t>(kAnchors)}));
    output_set.reg_values =
            std::make_shared<std::vector<float>>(test_case.reg);
    output_set.cls_values =
            std::make_shared<std::vector<float>>(test_case.cls);

    rknn::RknnTensorView reg{
            output_set.reg_info.get(),
            output_set.reg_values->data(),
            output_set.reg_values->size(),
    };
    rknn::RknnTensorView cls{
            output_set.cls_info.get(),
            output_set.cls_values->data(),
            output_set.cls_values->size(),
    };
    if (swap_outputs) {
        output_set.views = {cls, reg};
    } else {
        output_set.views = {reg, cls};
    }
    return output_set;
}

std::vector<float> transpose_to_nhwc(
        const std::vector<float>& values,
        size_t channels,
        size_t anchors) {
    std::vector<float> transposed(values.size());
    for (size_t anchor = 0; anchor < anchors; ++anchor) {
        for (size_t channel = 0; channel < channels; ++channel) {
            transposed[anchor * channels + channel] =
                    values[channel * anchors + anchor];
        }
    }
    return transposed;
}

OutputSet make_nhwc_outputs(
        const TestCase& test_case,
        bool swap_outputs) {
    OutputSet output_set;
    output_set.reg_info = std::make_shared<rknn::RknnTensorInfo>(make_tensor_info(
            "output_reg",
            RKNN_TENSOR_NHWC,
            {1, static_cast<uint32_t>(kAnchors), static_cast<uint32_t>(kRegChannels)}));
    output_set.cls_info = std::make_shared<rknn::RknnTensorInfo>(make_tensor_info(
            "output_cls",
            RKNN_TENSOR_NHWC,
            {1, static_cast<uint32_t>(kAnchors), static_cast<uint32_t>(kClasses)}));

    output_set.reg_values = std::make_shared<std::vector<float>>(
            transpose_to_nhwc(test_case.reg, kRegChannels, kAnchors));
    output_set.cls_values = std::make_shared<std::vector<float>>(
            transpose_to_nhwc(test_case.cls, kClasses, kAnchors));
    rknn::RknnTensorView reg_view{
            output_set.reg_info.get(),
            output_set.reg_values->data(),
            output_set.reg_values->size(),
    };
    rknn::RknnTensorView cls_view{
            output_set.cls_info.get(),
            output_set.cls_values->data(),
            output_set.cls_values->size(),
    };
    if (swap_outputs) {
        output_set.views = {cls_view, reg_view};
    } else {
        output_set.views = {reg_view, cls_view};
    }
    return output_set;
}

void verify_single_detection(
        const rknn::RknnInferenceResult& result,
        int expected_class_id) {
    require(result.detections.size() == 1, "expected one detection");
    const rknn::RknnDetection& detection = result.detections.front();
    require(
            detection.class_id == expected_class_id,
            "unexpected detection class");
    require_close(detection.confidence, 0.9999546f, 1e-6f, "confidence");
    require_close(detection.left, 4.0f, 1e-5f, "left");
    require_close(detection.top, 4.0f, 1e-5f, "top");
    require_close(detection.right, 28.0f, 1e-5f, "right");
    require_close(detection.bottom, 28.0f, 1e-5f, "bottom");
}

void test_nchw_and_output_order() {
    const TestCase test_case = make_test_case(true);
    rknn::RknnYolov8OutputProcessor processor(make_config());

    rknn::RknnInferenceResult result;
    const OutputSet outputs = make_nchw_outputs(test_case, false);
    int return_code = processor.process(
            make_frame_info(),
            outputs.views,
            result);
    require(return_code == 0, "NCHW decode failed");
    verify_single_detection(result, 0);

    const OutputSet swapped_outputs = make_nchw_outputs(test_case, true);
    return_code = processor.process(
            make_frame_info(),
            swapped_outputs.views,
            result);
    require(return_code == 0, "swapped NCHW outputs failed");
    verify_single_detection(result, 0);
}

void test_nhwc_matches_nchw() {
    const TestCase test_case = make_test_case(true);
    rknn::RknnYolov8OutputProcessor processor(make_config());

    rknn::RknnInferenceResult nchw_result;
    const OutputSet nchw_outputs = make_nchw_outputs(test_case, false);
    const int nchw_return_code = processor.process(
            make_frame_info(),
            nchw_outputs.views,
            nchw_result);
    require(nchw_return_code == 0, "NCHW reference decode failed");

    rknn::RknnInferenceResult nhwc_result;
    const OutputSet nhwc_outputs = make_nhwc_outputs(test_case, false);
    const int nhwc_return_code = processor.process(
            make_frame_info(),
            nhwc_outputs.views,
            nhwc_result);
    require(nhwc_return_code == 0, "NHWC decode failed");
    require(
            nchw_result.detections.size() == nhwc_result.detections.size(),
            "NCHW and NHWC detection counts differ");
    for (size_t index = 0; index < nchw_result.detections.size(); ++index) {
        const rknn::RknnDetection& expected = nchw_result.detections[index];
        const rknn::RknnDetection& actual = nhwc_result.detections[index];
        require(actual.class_id == expected.class_id, "layout class differs");
        require_close(actual.confidence, expected.confidence, 1e-6f, "layout score");
        require_close(actual.left, expected.left, 1e-5f, "layout left");
        require_close(actual.top, expected.top, 1e-5f, "layout top");
        require_close(actual.right, expected.right, 1e-5f, "layout right");
        require_close(actual.bottom, expected.bottom, 1e-5f, "layout bottom");
    }
}

void test_class_aware_nms() {
    const TestCase test_case = make_test_case(false);
    rknn::RknnYolov8OutputProcessor processor(make_config());

    rknn::RknnInferenceResult result;
    const OutputSet outputs = make_nchw_outputs(test_case, false);
    const int return_code = processor.process(
            make_frame_info(),
            outputs.views,
            result);
    require(return_code == 0, "class-aware NMS decode failed");
    require(result.detections.size() == 2, "different classes must both survive");
    require(result.detections[0].class_id == 0, "unexpected first class");
    require(result.detections[1].class_id == 1, "unexpected second class");
}

void test_missing_output() {
    const TestCase test_case = make_test_case(true);
    rknn::RknnYolov8OutputProcessor processor(make_config());
    rknn::RknnInferenceResult result;
    const OutputSet outputs = make_nchw_outputs(test_case, false);
    const int return_code = processor.process(
            make_frame_info(),
            {outputs.views.front()},
            result);
    require(return_code != 0, "missing output must fail");
}

}

int main() {
    test_nchw_and_output_order();
    test_nhwc_matches_nchw();
    test_class_aware_nms();
    test_missing_output();
    std::cout << "[OK] all YOLOv8 output processor tests passed\n";
    return 0;
}
