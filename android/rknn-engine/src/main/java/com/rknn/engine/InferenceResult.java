package com.rknn.engine;

public final class InferenceResult {
    private final String modelName;
    private final int frameWidth;
    private final int frameHeight;
    private final int inputWidth;
    private final int inputHeight;
    private final double preprocessMillis;
    private final double inferenceMillis;
    private final double postprocessMillis;
    private final double totalMillis;
    private final TensorOutput[] outputs;
    private final DetectionResult detectionResult;

    public InferenceResult(
            String modelName,
            int frameWidth,
            int frameHeight,
            int inputWidth,
            int inputHeight,
            double preprocessMillis,
            double inferenceMillis,
            double postprocessMillis,
            double totalMillis,
            TensorOutput[] outputs) {
        this(
                modelName,
                frameWidth,
                frameHeight,
                inputWidth,
                inputHeight,
                preprocessMillis,
                inferenceMillis,
                postprocessMillis,
                totalMillis,
                outputs,
                new DetectionResult(null));
    }

    public InferenceResult(
            String modelName,
            int frameWidth,
            int frameHeight,
            int inputWidth,
            int inputHeight,
            double preprocessMillis,
            double inferenceMillis,
            double postprocessMillis,
            double totalMillis,
            TensorOutput[] outputs,
            DetectionResult detectionResult) {
        this.modelName = modelName;
        this.frameWidth = frameWidth;
        this.frameHeight = frameHeight;
        this.inputWidth = inputWidth;
        this.inputHeight = inputHeight;
        this.preprocessMillis = preprocessMillis;
        this.inferenceMillis = inferenceMillis;
        this.postprocessMillis = postprocessMillis;
        this.totalMillis = totalMillis;
        this.outputs = outputs;
        this.detectionResult = detectionResult == null
                ? new DetectionResult(null)
                : detectionResult;
    }

    public String getModelName() {
        return modelName;
    }

    public int getFrameWidth() {
        return frameWidth;
    }

    public int getFrameHeight() {
        return frameHeight;
    }

    public int getInputWidth() {
        return inputWidth;
    }

    public int getInputHeight() {
        return inputHeight;
    }

    public double getPreprocessMillis() {
        return preprocessMillis;
    }

    public double getInferenceMillis() {
        return inferenceMillis;
    }

    public double getPostprocessMillis() {
        return postprocessMillis;
    }

    public double getTotalMillis() {
        return totalMillis;
    }

    public TensorOutput[] getOutputs() {
        return outputs;
    }

    public DetectionResult getDetectionResult() {
        return detectionResult;
    }
}
