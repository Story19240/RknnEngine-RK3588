package com.rknn.engine;

import android.content.res.AssetManager;
import android.graphics.Bitmap;

import java.io.Closeable;

public final class RknnEngine implements Closeable {
    public static final int ERROR_INVALID_ARGUMENT = -1;
    public static final int ERROR_ENGINE_UNAVAILABLE = -2;
    public static final int ERROR_INFERENCE_FAILED = -3;
    public static final int ERROR_CALLBACK_FAILED = -4;
    public static final int ERROR_MODEL_LOAD_FAILED = -5;

    private static final String NATIVE_LIBRARY = "rknnengine";

    static {
        System.loadLibrary(NATIVE_LIBRARY);
    }

    private long nativeHandle;
    private volatile EngineCallback resultCallback;

    public RknnEngine() {
        nativeHandle = nativeCreate();
        if (nativeHandle == 0L) {
            throw new IllegalStateException("Failed to create native RKNN engine");
        }
    }

    public synchronized boolean loadModel(AssetManager assetManager, String modelName) {
        return loadModel(assetManager, modelName, null);
    }

    public synchronized boolean loadModel(
            AssetManager assetManager,
            String modelName,
            DetectionConfig detectionConfig) {
        if (nativeHandle == 0L || assetManager == null || modelName == null || modelName.isEmpty()) {
            return false;
        }
        if (detectionConfig == null) {
            return nativeLoadModel(nativeHandle, assetManager, modelName) == 0;
        }
        return nativeLoadModelWithConfig(
                nativeHandle,
                assetManager,
                modelName,
                detectionConfig.getConfidenceThreshold(),
                detectionConfig.getNmsThreshold(),
                detectionConfig.getMaxDetections(),
                detectionConfig.isIncludeRawOutputs(),
                detectionConfig.getClassNames()) == 0;
    }

    public void setResultCallback(EngineCallback callback) {
        resultCallback = callback;
    }

    public synchronized boolean submitFrame(Bitmap frame) {
        if (nativeHandle == 0L || frame == null || frame.isRecycled()) {
            return false;
        }

        EngineCallback callback = resultCallback;
        if (callback == null) {
            return false;
        }

        Bitmap input = frame;
        boolean converted = false;
        if (frame.getConfig() != Bitmap.Config.ARGB_8888) {
            try {
                input = frame.copy(Bitmap.Config.ARGB_8888, false);
            } catch (OutOfMemoryError | IllegalArgumentException exception) {
                callback.onError(new EngineError(
                        ERROR_INVALID_ARGUMENT,
                        "Failed to convert frame to ARGB_8888",
                        frame.getWidth(),
                        frame.getHeight()));
                return false;
            }
            if (input == null) {
                callback.onError(new EngineError(
                        ERROR_INVALID_ARGUMENT,
                        "Failed to convert frame to ARGB_8888",
                        frame.getWidth(),
                        frame.getHeight()));
                return false;
            }
            converted = true;
        }

        try {
            return nativeProcessFrame(nativeHandle, input, callback) == 0;
        } finally {
            if (converted && input != null) {
                input.recycle();
            }
        }
    }

    @Override
    public synchronized void close() {
        if (nativeHandle != 0L) {
            nativeRelease(nativeHandle);
            nativeHandle = 0L;
        }
    }

    private static native long nativeCreate();

    private static native int nativeLoadModel(
            long handle,
            AssetManager assetManager,
            String modelName);

    private static native int nativeLoadModelWithConfig(
            long handle,
            AssetManager assetManager,
            String modelName,
            float confidenceThreshold,
            float nmsThreshold,
            int maxDetections,
            boolean includeRawOutputs,
            String[] classNames);

    private static native int nativeProcessFrame(
            long handle,
            Bitmap frame,
            EngineCallback callback);

    private static native void nativeRelease(long handle);
}
