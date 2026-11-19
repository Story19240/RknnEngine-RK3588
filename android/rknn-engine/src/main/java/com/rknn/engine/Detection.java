package com.rknn.engine;

public final class Detection {
    private final int classId;
    private final String className;
    private final float confidence;
    private final float left;
    private final float top;
    private final float right;
    private final float bottom;

    public Detection(
            int classId,
            String className,
            float confidence,
            float left,
            float top,
            float right,
            float bottom) {
        this.classId = classId;
        this.className = className == null ? "" : className;
        this.confidence = confidence;
        this.left = left;
        this.top = top;
        this.right = right;
        this.bottom = bottom;
    }

    public int getClassId() {
        return classId;
    }

    public String getClassName() {
        return className;
    }

    public float getConfidence() {
        return confidence;
    }

    public float getLeft() {
        return left;
    }

    public float getTop() {
        return top;
    }

    public float getRight() {
        return right;
    }

    public float getBottom() {
        return bottom;
    }
}
