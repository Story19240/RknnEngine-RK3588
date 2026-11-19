package com.rknn.engine;

public final class DetectionConfig {
    private final float confidenceThreshold;
    private final float nmsThreshold;
    private final int maxDetections;
    private final boolean includeRawOutputs;
    private final String[] classNames;

    public DetectionConfig(
            float confidenceThreshold,
            float nmsThreshold,
            int maxDetections,
            String[] classNames) {
        this(confidenceThreshold, nmsThreshold, maxDetections, false, classNames);
    }

    public DetectionConfig(
            float confidenceThreshold,
            float nmsThreshold,
            int maxDetections,
            boolean includeRawOutputs,
            String[] classNames) {
        if (confidenceThreshold < 0.0f || confidenceThreshold > 1.0f) {
            throw new IllegalArgumentException("confidenceThreshold must be in [0, 1]");
        }
        if (nmsThreshold < 0.0f || nmsThreshold > 1.0f) {
            throw new IllegalArgumentException("nmsThreshold must be in [0, 1]");
        }
        if (maxDetections <= 0) {
            throw new IllegalArgumentException("maxDetections must be positive");
        }

        this.confidenceThreshold = confidenceThreshold;
        this.nmsThreshold = nmsThreshold;
        this.maxDetections = maxDetections;
        this.includeRawOutputs = includeRawOutputs;
        this.classNames = classNames == null ? new String[0] : classNames.clone();
    }

    public static DetectionConfig defaults() {
        return new DetectionConfig(0.25f, 0.45f, 300, null);
    }

    public static DetectionConfig forDefectModel() {
        return new DetectionConfig(
                0.25f,
                0.45f,
                300,
                false,
                new String[] {"变形", "涂层丢失", "烧灼烧穿", "缺损", "裂纹"});
    }

    public float getConfidenceThreshold() {
        return confidenceThreshold;
    }

    public float getNmsThreshold() {
        return nmsThreshold;
    }

    public int getMaxDetections() {
        return maxDetections;
    }

    public boolean isIncludeRawOutputs() {
        return includeRawOutputs;
    }

    public String[] getClassNames() {
        return classNames.clone();
    }
}
