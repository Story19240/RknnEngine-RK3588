package com.rknn.engine;

public interface EngineCallback {
    void onResult(InferenceResult result);

    void onError(EngineError error);
}
