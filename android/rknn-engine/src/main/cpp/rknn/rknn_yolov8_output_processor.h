#ifndef RKNN_ENGINE_RKNN_YOLOV8_OUTPUT_PROCESSOR_H
#define RKNN_ENGINE_RKNN_YOLOV8_OUTPUT_PROCESSOR_H

#include "rknn_wrapper.h"

namespace rknn {

class RknnYolov8OutputProcessor final : public RknnOutputProcessor {
public:
    explicit RknnYolov8OutputProcessor(RknnDetectionConfig config);

    int process(
            const RknnFrameInfo& frame_info,
            const std::vector<RknnTensorView>& outputs,
            RknnInferenceResult& result) override;

private:
    RknnDetectionConfig config_;
};

}

#endif
