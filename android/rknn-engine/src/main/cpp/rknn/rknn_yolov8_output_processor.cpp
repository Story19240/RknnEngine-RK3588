#include "rknn_yolov8_output_processor.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace rknn {
namespace {

constexpr size_t kMaxCandidates = 30000;

struct TensorLayout {
    size_t channels = 0;
    size_t anchors = 0;
    bool channel_major = true;
};

struct Candidate {
    int class_id = 0;
    float confidence = 0.0f;
    float left = 0.0f;
    float top = 0.0f;
    float right = 0.0f;
    float bottom = 0.0f;
};

std::string lowercase(std::string value) {
    std::transform(
            value.begin(),
            value.end(),
            value.begin(),
            [](unsigned char character) { return static_cast<char>(std::tolower(character)); });
    return value;
}

bool output_name_matches(const std::string& actual, const std::string& expected) {
    const std::string actual_lower = lowercase(actual);
    const std::string expected_lower = lowercase(expected);
    return actual_lower == expected_lower ||
           actual_lower.find(expected_lower) != std::string::npos;
}

const RknnTensorView* find_named_output(
        const std::vector<RknnTensorView>& outputs,
        const std::string& expected_name) {
    for (const RknnTensorView& output : outputs) {
        if (output.info && output_name_matches(output.info->name, expected_name)) {
            return &output;
        }
    }
    return nullptr;
}

bool resolve_layout(
        const RknnTensorInfo& info,
        size_t expected_channels,
        size_t expected_anchors,
        TensorLayout& layout) {
    if (info.shape.size() != 3 || info.shape[0] != 1) {
        return false;
    }

    const size_t dimension_one = info.shape[1];
    const size_t dimension_two = info.shape[2];
    size_t channel_axis = 0;
    if (info.format == RKNN_TENSOR_NCHW) {
        if (dimension_one != expected_channels) {
            return false;
        }
        channel_axis = 1;
    } else if (info.format == RKNN_TENSOR_NHWC) {
        if (dimension_two != expected_channels) {
            return false;
        }
        channel_axis = 2;
    } else if (dimension_one == expected_channels) {
        channel_axis = 1;
    } else if (dimension_two == expected_channels) {
        channel_axis = 2;
    } else {
        return false;
    }

    layout.channels = expected_channels;
    layout.anchors = channel_axis == 1 ? dimension_two : dimension_one;
    if (expected_anchors != 0 && layout.anchors != expected_anchors) {
        return false;
    }

    layout.channel_major = channel_axis == 1;
    return true;
}

bool has_complete_tensor(
        const RknnTensorView& view,
        const TensorLayout& layout) {
    if (!view.values || layout.channels == 0 || layout.anchors == 0) {
        return false;
    }
    if (layout.channels >
        std::numeric_limits<size_t>::max() / layout.anchors) {
        return false;
    }
    return view.element_count >= layout.channels * layout.anchors;
}

float tensor_value(
        const RknnTensorView& view,
        const TensorLayout& layout,
        size_t channel,
        size_t anchor) {
    const size_t index = layout.channel_major
            ? channel * layout.anchors + anchor
            : anchor * layout.channels + channel;
    return view.values[index];
}

float sigmoid(float value) {
    if (value >= 0.0f) {
        const float exponent = std::exp(-value);
        return 1.0f / (1.0f + exponent);
    }
    const float exponent = std::exp(value);
    return exponent / (1.0f + exponent);
}

float decode_dfl_side(
        const RknnTensorView& reg,
        const TensorLayout& reg_layout,
        size_t anchor,
        size_t side,
        int reg_max) {
    const size_t channel_base = side * static_cast<size_t>(reg_max);
    float maximum = -std::numeric_limits<float>::infinity();
    for (int index = 0; index < reg_max; ++index) {
        maximum = std::max(
                maximum,
                tensor_value(reg, reg_layout, channel_base + static_cast<size_t>(index), anchor));
    }

    float sum = 0.0f;
    for (int index = 0; index < reg_max; ++index) {
        sum += std::exp(
                tensor_value(reg, reg_layout, channel_base + static_cast<size_t>(index), anchor) -
                maximum);
    }

    float expected = 0.0f;
    for (int index = 0; index < reg_max; ++index) {
        const float probability = std::exp(
                tensor_value(reg, reg_layout, channel_base + static_cast<size_t>(index), anchor) -
                maximum) / sum;
        expected += probability * static_cast<float>(index);
    }
    return expected;
}

struct GridPoint {
    float x = 0.0f;
    float y = 0.0f;
    float stride = 0.0f;
};

std::vector<GridPoint> make_grid(
        int input_width,
        int input_height,
        const std::vector<int>& strides) {
    std::vector<GridPoint> grid;
    for (int stride : strides) {
        if (stride <= 0 || input_width % stride != 0 || input_height % stride != 0) {
            return {};
        }
        const int width = input_width / stride;
        const int height = input_height / stride;
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                grid.push_back(GridPoint{
                        static_cast<float>(x),
                        static_cast<float>(y),
                        static_cast<float>(stride)});
            }
        }
    }
    return grid;
}

float intersection_over_union(const Candidate& first, const Candidate& second) {
    const float width = std::max(
            0.0f,
            std::min(first.right, second.right) - std::max(first.left, second.left));
    const float height = std::max(
            0.0f,
            std::min(first.bottom, second.bottom) - std::max(first.top, second.top));
    const float intersection = width * height;
    const float first_area =
            std::max(0.0f, first.right - first.left) *
            std::max(0.0f, first.bottom - first.top);
    const float second_area =
            std::max(0.0f, second.right - second.left) *
            std::max(0.0f, second.bottom - second.top);
    const float union_area = first_area + second_area - intersection;
    return union_area > 0.0f ? intersection / union_area : 0.0f;
}

std::vector<Candidate> class_aware_nms(
        std::vector<Candidate> candidates,
        float iou_threshold,
        int max_detections) {
    std::stable_sort(
            candidates.begin(),
            candidates.end(),
            [](const Candidate& first, const Candidate& second) {
                return first.confidence > second.confidence;
            });

    std::vector<Candidate> kept;
    kept.reserve(std::min(candidates.size(), static_cast<size_t>(max_detections)));
    for (const Candidate& candidate : candidates) {
        bool suppressed = false;
        for (const Candidate& selected : kept) {
            if (candidate.class_id == selected.class_id &&
                intersection_over_union(candidate, selected) > iou_threshold) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) {
            kept.push_back(candidate);
            if (static_cast<int>(kept.size()) >= max_detections) {
                break;
            }
        }
    }
    return kept;
}

}

RknnYolov8OutputProcessor::RknnYolov8OutputProcessor(RknnDetectionConfig config)
    : config_(std::move(config)) {
}

int RknnYolov8OutputProcessor::process(
        const RknnFrameInfo& frame_info,
        const std::vector<RknnTensorView>& outputs,
        RknnInferenceResult& result) {
    result.detections.clear();

    if (config_.reg_max <= 0 ||
        config_.max_detections <= 0 ||
        config_.num_classes < 0 ||
        config_.confidence_threshold < 0.0f ||
        config_.confidence_threshold > 1.0f ||
        config_.nms_threshold < 0.0f ||
        config_.nms_threshold > 1.0f ||
        config_.strides.empty()) {
        return -1;
    }

    const size_t reg_channels = static_cast<size_t>(config_.reg_max * 4);
    const RknnTensorView* reg_output = find_named_output(outputs, "output_reg");
    const RknnTensorView* cls_output = find_named_output(outputs, "output_cls");
    if (!reg_output || !cls_output || !reg_output->info || !cls_output->info) {
        return -1;
    }

    TensorLayout reg_layout;
    if (!resolve_layout(*reg_output->info, reg_channels, 0, reg_layout)) {
        return -2;
    }

    const size_t cls_channels = config_.num_classes > 0
            ? static_cast<size_t>(config_.num_classes)
            : 0;
    TensorLayout cls_layout;
    if (cls_channels > 0) {
        if (!resolve_layout(*cls_output->info, cls_channels, reg_layout.anchors, cls_layout)) {
            return -3;
        }
    } else {
        const std::vector<uint32_t>& shape = cls_output->info->shape;
        if (shape.size() != 3 ||
            (shape[1] != reg_layout.anchors && shape[2] != reg_layout.anchors)) {
            return -3;
        }
        cls_layout.anchors = reg_layout.anchors;
        cls_layout.channels = shape[1] == reg_layout.anchors ? shape[2] : shape[1];
        cls_layout.channel_major = shape[1] != reg_layout.anchors;
    }

    if (!has_complete_tensor(*reg_output, reg_layout) ||
        !has_complete_tensor(*cls_output, cls_layout)) {
        return -4;
    }

    const std::vector<GridPoint> grid = make_grid(
            frame_info.input_width,
            frame_info.input_height,
            config_.strides);
    if (grid.size() != reg_layout.anchors || cls_layout.anchors != reg_layout.anchors) {
        return -4;
    }

    std::vector<Candidate> candidates;
    candidates.reserve(std::min(reg_layout.anchors, kMaxCandidates));
    for (size_t anchor = 0; anchor < reg_layout.anchors; ++anchor) {
        int best_class = 0;
        float best_confidence = -1.0f;
        for (size_t class_id = 0; class_id < cls_layout.channels; ++class_id) {
            const float confidence = sigmoid(
                    tensor_value(*cls_output, cls_layout, class_id, anchor));
            if (confidence > best_confidence) {
                best_confidence = confidence;
                best_class = static_cast<int>(class_id);
            }
        }
        if (best_confidence <= config_.confidence_threshold) {
            continue;
        }
        if (candidates.size() >= kMaxCandidates) {
            break;
        }

        const float left_distance =
                decode_dfl_side(*reg_output, reg_layout, anchor, 0, config_.reg_max);
        const float top_distance =
                decode_dfl_side(*reg_output, reg_layout, anchor, 1, config_.reg_max);
        const float right_distance =
                decode_dfl_side(*reg_output, reg_layout, anchor, 2, config_.reg_max);
        const float bottom_distance =
                decode_dfl_side(*reg_output, reg_layout, anchor, 3, config_.reg_max);

        const GridPoint& point = grid[anchor];
        Candidate candidate;
        candidate.class_id = best_class;
        candidate.confidence = best_confidence;
        candidate.left = (point.x + 0.5f - left_distance) * point.stride;
        candidate.top = (point.y + 0.5f - top_distance) * point.stride;
        candidate.right = (point.x + 0.5f + right_distance) * point.stride;
        candidate.bottom = (point.y + 0.5f + bottom_distance) * point.stride;
        candidates.push_back(candidate);
    }

    const std::vector<Candidate> kept = class_aware_nms(
            std::move(candidates),
            config_.nms_threshold,
            config_.max_detections);
    result.detections.reserve(kept.size());
    for (const Candidate& candidate : kept) {
        RknnDetection detection;
        detection.class_id = candidate.class_id;
        if (candidate.class_id >= 0 &&
            candidate.class_id < static_cast<int>(config_.class_names.size())) {
            detection.class_name = config_.class_names[static_cast<size_t>(candidate.class_id)];
        } else {
            detection.class_name = std::to_string(candidate.class_id);
        }
        detection.confidence = candidate.confidence;
        detection.left = std::max(
                0.0f,
                std::min(
                        static_cast<float>(frame_info.source_width),
                        (candidate.left - static_cast<float>(frame_info.padding_left)) /
                                frame_info.resize_scale));
        detection.top = std::max(
                0.0f,
                std::min(
                        static_cast<float>(frame_info.source_height),
                        (candidate.top - static_cast<float>(frame_info.padding_top)) /
                                frame_info.resize_scale));
        detection.right = std::max(
                0.0f,
                std::min(
                        static_cast<float>(frame_info.source_width),
                        (candidate.right - static_cast<float>(frame_info.padding_left)) /
                                frame_info.resize_scale));
        detection.bottom = std::max(
                0.0f,
                std::min(
                        static_cast<float>(frame_info.source_height),
                        (candidate.bottom - static_cast<float>(frame_info.padding_top)) /
                                frame_info.resize_scale));
        result.detections.push_back(detection);
    }
    return 0;
}

}
