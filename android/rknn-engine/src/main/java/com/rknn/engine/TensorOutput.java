package com.rknn.engine;

public final class TensorOutput {
    private final String name;
    private final int[] shape;
    private final float[] values;
    private final String format;

    public TensorOutput(String name, int[] shape, float[] values) {
        this(name, shape, values, "UNKNOWN");
    }

    public TensorOutput(String name, int[] shape, float[] values, String format) {
        this.name = name;
        this.shape = shape;
        this.values = values;
        this.format = format == null ? "UNKNOWN" : format;
    }

    public String getName() {
        return name;
    }

    public int[] getShape() {
        return shape;
    }

    public float[] getValues() {
        return values;
    }

    public String getFormat() {
        return format;
    }
}
