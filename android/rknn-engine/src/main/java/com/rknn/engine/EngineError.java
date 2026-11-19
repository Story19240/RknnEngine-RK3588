package com.rknn.engine;

public final class EngineError {
    private final int code;
    private final String message;
    private final int frameWidth;
    private final int frameHeight;

    public EngineError(int code, String message, int frameWidth, int frameHeight) {
        this.code = code;
        this.message = message;
        this.frameWidth = frameWidth;
        this.frameHeight = frameHeight;
    }

    public int getCode() {
        return code;
    }

    public String getMessage() {
        return message;
    }

    public int getFrameWidth() {
        return frameWidth;
    }

    public int getFrameHeight() {
        return frameHeight;
    }
}
