package com.rknn.engine;

public final class DetectionResult {
    private static final Detection[] EMPTY = new Detection[0];

    private final Detection[] detections;

    public DetectionResult(Detection[] detections) {
        this.detections = detections == null ? EMPTY : detections.clone();
    }

    public Detection[] getDetections() {
        return detections.clone();
    }

    public int size() {
        return detections.length;
    }

    public boolean isEmpty() {
        return detections.length == 0;
    }
}
