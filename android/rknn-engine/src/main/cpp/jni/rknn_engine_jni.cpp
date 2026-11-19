#include <android/asset_manager_jni.h>
#include <android/bitmap.h>
#include <jni.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <cstdint>
#include <limits>
#include <mutex>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include "rknn/rknn_wrapper.h"

namespace {

constexpr int kErrorInvalidArgument = -1;
constexpr int kErrorEngineUnavailable = -2;
constexpr int kErrorInferenceFailed = -3;
constexpr int kErrorCallbackFailed = -4;
constexpr int kErrorModelUnavailable = -5;

struct RknnEngineHandle {
    std::mutex mutex;
    rknn::RknnWrapper wrapper;
};

bool clear_pending_exception(JNIEnv* env);

RknnEngineHandle* handle_from(jlong handle) {
    return reinterpret_cast<RknnEngineHandle*>(static_cast<intptr_t>(handle));
}

std::string string_from_java(JNIEnv* env, jstring value) {
    if (!value) {
        return {};
    }
    const char* characters = env->GetStringUTFChars(value, nullptr);
    if (!characters) {
        return {};
    }
    std::string result(characters);
    env->ReleaseStringUTFChars(value, characters);
    return result;
}

std::vector<std::string> string_array_from_java(JNIEnv* env, jobjectArray value) {
    std::vector<std::string> result;
    if (!value) {
        return result;
    }

    const jsize count = env->GetArrayLength(value);
    result.reserve(static_cast<size_t>(count));
    for (jsize index = 0; index < count; ++index) {
        jstring item = static_cast<jstring>(env->GetObjectArrayElement(value, index));
        if (!item) {
            clear_pending_exception(env);
            result.emplace_back();
            continue;
        }
        result.push_back(string_from_java(env, item));
        env->DeleteLocalRef(item);
    }
    return result;
}

bool clear_pending_exception(JNIEnv* env) {
    if (!env->ExceptionCheck()) {
        return true;
    }
    env->ExceptionDescribe();
    env->ExceptionClear();
    return false;
}

jclass find_class(JNIEnv* env, const char* name) {
    jclass result = env->FindClass(name);
    if (!result) {
        clear_pending_exception(env);
    }
    return result;
}

jmethodID find_method(JNIEnv* env, jclass type, const char* name, const char* signature) {
    jmethodID result = env->GetMethodID(type, name, signature);
    if (!result) {
        clear_pending_exception(env);
    }
    return result;
}

jobject create_error_object(
        JNIEnv* env,
        int code,
        const std::string& message,
        int frame_width,
        int frame_height) {
    jclass error_class = find_class(env, "com/rknn/engine/EngineError");
    if (!error_class) {
        return nullptr;
    }
    jmethodID constructor = find_method(
            env,
            error_class,
            "<init>",
            "(ILjava/lang/String;II)V");
    if (!constructor) {
        return nullptr;
    }
    jstring java_message = env->NewStringUTF(message.c_str());
    if (!java_message) {
        clear_pending_exception(env);
        return nullptr;
    }
    jobject error = env->NewObject(
            error_class,
            constructor,
            code,
            java_message,
            frame_width,
            frame_height);
    if (!error) {
        clear_pending_exception(env);
    }
    env->DeleteLocalRef(java_message);
    env->DeleteLocalRef(error_class);
    return error;
}

jobject create_result_object(JNIEnv* env, const rknn::RknnInferenceResult& result) {
    jclass tensor_output_class = find_class(env, "com/rknn/engine/TensorOutput");
    jclass inference_result_class = find_class(env, "com/rknn/engine/InferenceResult");
    jclass detection_class = find_class(env, "com/rknn/engine/Detection");
    jclass detection_result_class = find_class(env, "com/rknn/engine/DetectionResult");
    if (!tensor_output_class ||
        !inference_result_class ||
        !detection_class ||
        !detection_result_class) {
        return nullptr;
    }

    jmethodID tensor_output_constructor = find_method(
            env,
            tensor_output_class,
            "<init>",
            "(Ljava/lang/String;[I[FLjava/lang/String;)V");
    jmethodID inference_result_constructor = find_method(
            env,
            inference_result_class,
            "<init>",
            "(Ljava/lang/String;IIIIJJJJ[Lcom/rknn/engine/TensorOutput;"
            "Lcom/rknn/engine/DetectionResult;)V");
    jmethodID detection_constructor = find_method(
            env,
            detection_class,
            "<init>",
            "(ILjava/lang/String;FFFFF)V");
    jmethodID detection_result_constructor = find_method(
            env,
            detection_result_class,
            "<init>",
            "([Lcom/rknn/engine/Detection;)V");
    if (!tensor_output_constructor ||
        !inference_result_constructor ||
        !detection_constructor ||
        !detection_result_constructor) {
        return nullptr;
    }

    const size_t output_count = result.include_raw_outputs
            ? result.output_views.size()
            : 0;
    jobjectArray outputs = env->NewObjectArray(
            static_cast<jsize>(output_count),
            tensor_output_class,
            nullptr);
    if (!outputs) {
        clear_pending_exception(env);
        return nullptr;
    }
    for (size_t index = 0; index < output_count; ++index) {
        const rknn::RknnTensorView& view = result.output_views[index];
        if (!view.info ||
            view.element_count > static_cast<size_t>(std::numeric_limits<jsize>::max())) {
            return nullptr;
        }

        std::vector<jint> shape;
        shape.reserve(view.info->shape.size());
        for (uint32_t dimension : view.info->shape) {
            shape.push_back(static_cast<jint>(dimension));
        }
        jintArray java_shape = env->NewIntArray(static_cast<jsize>(shape.size()));
        if (!shape.empty()) {
            env->SetIntArrayRegion(
                    java_shape,
                    0,
                    static_cast<jsize>(shape.size()),
                    shape.data());
        }
        if (!java_shape) {
            clear_pending_exception(env);
            return nullptr;
        }

        jfloatArray java_values = env->NewFloatArray(static_cast<jsize>(view.element_count));
        if (!java_values) {
            clear_pending_exception(env);
            return nullptr;
        }
        if (view.element_count > 0 && view.values != nullptr) {
            env->SetFloatArrayRegion(
                    java_values,
                    0,
                    static_cast<jsize>(view.element_count),
                    view.values);
        }
        jstring java_name = env->NewStringUTF(view.info->name.c_str());
        const char* format = get_format_string(view.info->format);
        jstring java_format = env->NewStringUTF(format == nullptr ? "UNKNOWN" : format);
        if (!java_name || !java_format) {
            clear_pending_exception(env);
            return nullptr;
        }
        jobject java_output = env->NewObject(
                tensor_output_class,
                tensor_output_constructor,
                java_name,
                java_shape,
                java_values,
                java_format);
        if (!java_output) {
            clear_pending_exception(env);
            return nullptr;
        }
        env->SetObjectArrayElement(outputs, index, java_output);
        env->DeleteLocalRef(java_output);
        env->DeleteLocalRef(java_format);
        env->DeleteLocalRef(java_name);
        env->DeleteLocalRef(java_values);
        env->DeleteLocalRef(java_shape);
    }

    jobjectArray detections = env->NewObjectArray(
            static_cast<jsize>(result.detections.size()),
            detection_class,
            nullptr);
    if (!detections) {
        clear_pending_exception(env);
        return nullptr;
    }
    for (size_t index = 0; index < result.detections.size(); ++index) {
        const rknn::RknnDetection& detection = result.detections[index];
        jstring java_class_name = env->NewStringUTF(detection.class_name.c_str());
        if (!java_class_name) {
            clear_pending_exception(env);
            return nullptr;
        }
        jobject java_detection = env->NewObject(
                detection_class,
                detection_constructor,
                detection.class_id,
                java_class_name,
                detection.confidence,
                detection.left,
                detection.top,
                detection.right,
                detection.bottom);
        if (!java_detection) {
            clear_pending_exception(env);
            return nullptr;
        }
        env->SetObjectArrayElement(detections, static_cast<jsize>(index), java_detection);
        env->DeleteLocalRef(java_detection);
        env->DeleteLocalRef(java_class_name);
    }

    jobject java_detection_result = env->NewObject(
            detection_result_class,
            detection_result_constructor,
            detections);
    if (!java_detection_result) {
        clear_pending_exception(env);
        return nullptr;
    }

    jstring java_model_name = env->NewStringUTF(result.model_name.c_str());
    if (!java_model_name) {
        clear_pending_exception(env);
        return nullptr;
    }
    jobject java_result = env->NewObject(
            inference_result_class,
            inference_result_constructor,
            java_model_name,
            result.frame.source_width,
            result.frame.source_height,
            result.frame.input_width,
            result.frame.input_height,
            result.preprocess_millis,
            result.inference_millis,
            result.postprocess_millis,
            result.total_millis,
            outputs,
            java_detection_result);
    env->DeleteLocalRef(java_detection_result);
    env->DeleteLocalRef(detections);
    env->DeleteLocalRef(outputs);
    env->DeleteLocalRef(java_model_name);
    env->DeleteLocalRef(detection_result_class);
    env->DeleteLocalRef(detection_class);
    env->DeleteLocalRef(inference_result_class);
    env->DeleteLocalRef(tensor_output_class);
    return java_result;
}

void invoke_result(JNIEnv* env, jobject callback, jobject result) {
    if (!callback || !result) {
        return;
    }
    jclass callback_class = env->GetObjectClass(callback);
    jmethodID on_result = find_method(
            env,
            callback_class,
            "onResult",
            "(Lcom/rknn/engine/InferenceResult;)V");
    if (!on_result) {
        env->DeleteLocalRef(callback_class);
        return;
    }
    env->CallVoidMethod(callback, on_result, result);
    if (env->ExceptionCheck()) {
        env->ExceptionDescribe();
        env->ExceptionClear();
    }
    env->DeleteLocalRef(callback_class);
}

void invoke_error(
        JNIEnv* env,
        jobject callback,
        int code,
        const std::string& message,
        int frame_width,
        int frame_height) {
    if (!callback) {
        return;
    }
    jobject error = create_error_object(
            env,
            code,
            message,
            frame_width,
            frame_height);
    if (!error) {
        return;
    }
    jclass callback_class = env->GetObjectClass(callback);
    jmethodID on_error = find_method(
            env,
            callback_class,
            "onError",
            "(Lcom/rknn/engine/EngineError;)V");
    if (!on_error) {
        env->DeleteLocalRef(callback_class);
        env->DeleteLocalRef(error);
        return;
    }
    env->CallVoidMethod(callback, on_error, error);
    if (env->ExceptionCheck()) {
        env->ExceptionDescribe();
        env->ExceptionClear();
    }
    env->DeleteLocalRef(callback_class);
    env->DeleteLocalRef(error);
}

}

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_rknn_engine_RknnEngine_nativeCreate(JNIEnv*, jclass) {
    auto* handle = new (std::nothrow) RknnEngineHandle();
    return static_cast<jlong>(reinterpret_cast<intptr_t>(handle));
}

JNIEXPORT jint JNICALL
Java_com_rknn_engine_RknnEngine_nativeLoadModel(
        JNIEnv* env,
        jclass,
        jlong handle_value,
        jobject asset_manager,
        jstring model_name_value) {
    RknnEngineHandle* handle = handle_from(handle_value);
    if (!handle || !asset_manager || !model_name_value) {
        return kErrorInvalidArgument;
    }

    const std::string model_name = string_from_java(env, model_name_value);
    if (model_name.empty()) {
        return kErrorInvalidArgument;
    }

    AAssetManager* assets = AAssetManager_fromJava(env, asset_manager);
    if (!assets) {
        return kErrorModelUnavailable;
    }

    AAsset* asset = AAssetManager_open(assets, model_name.c_str(), AASSET_MODE_BUFFER);
    if (!asset) {
        return kErrorModelUnavailable;
    }

    const void* model_data = AAsset_getBuffer(asset);
    const size_t model_size = static_cast<size_t>(AAsset_getLength(asset));
    int return_code = kErrorModelUnavailable;
    if (model_data && model_size > 0) {
        std::lock_guard<std::mutex> lock(handle->mutex);
        return_code = handle->wrapper.load(model_data, model_size, model_name);
    }
    AAsset_close(asset);
    return return_code;
}

JNIEXPORT jint JNICALL
Java_com_rknn_engine_RknnEngine_nativeLoadModelWithConfig(
        JNIEnv* env,
        jclass,
        jlong handle_value,
        jobject asset_manager,
        jstring model_name_value,
        jfloat confidence_threshold,
        jfloat nms_threshold,
        jint max_detections,
        jboolean include_raw_outputs,
        jobjectArray class_names) {
    RknnEngineHandle* handle = handle_from(handle_value);
    if (!handle || !asset_manager || !model_name_value) {
        return kErrorInvalidArgument;
    }

    const std::string model_name = string_from_java(env, model_name_value);
    if (model_name.empty()) {
        return kErrorInvalidArgument;
    }

    AAssetManager* assets = AAssetManager_fromJava(env, asset_manager);
    if (!assets) {
        return kErrorModelUnavailable;
    }

    AAsset* asset = AAssetManager_open(assets, model_name.c_str(), AASSET_MODE_BUFFER);
    if (!asset) {
        return kErrorModelUnavailable;
    }

    const void* model_data = AAsset_getBuffer(asset);
    const size_t model_size = static_cast<size_t>(AAsset_getLength(asset));
    int return_code = kErrorModelUnavailable;
    if (model_data && model_size > 0) {
        rknn::RknnDetectionConfig config;
        config.confidence_threshold = confidence_threshold;
        config.nms_threshold = nms_threshold;
        config.max_detections = max_detections;
        config.include_raw_outputs = include_raw_outputs == JNI_TRUE;
        config.class_names = string_array_from_java(env, class_names);
        config.num_classes = static_cast<int>(config.class_names.size());

        std::lock_guard<std::mutex> lock(handle->mutex);
        return_code = handle->wrapper.load(
                model_data,
                model_size,
                model_name,
                config);
    }
    AAsset_close(asset);
    return return_code;
}

JNIEXPORT jint JNICALL
Java_com_rknn_engine_RknnEngine_nativeProcessFrame(
        JNIEnv* env,
        jclass,
        jlong handle_value,
        jobject bitmap,
        jobject callback) {
    RknnEngineHandle* handle = handle_from(handle_value);
    if (!handle || !bitmap) {
        return kErrorInvalidArgument;
    }

    AndroidBitmapInfo bitmap_info{};
    if (AndroidBitmap_getInfo(env, bitmap, &bitmap_info) != ANDROID_BITMAP_RESULT_SUCCESS ||
        bitmap_info.format != ANDROID_BITMAP_FORMAT_RGBA_8888 ||
        bitmap_info.width <= 0 ||
        bitmap_info.height <= 0) {
        invoke_error(
                env,
                callback,
                kErrorInvalidArgument,
                "Frame must be a non-empty RGBA_8888 bitmap",
                0,
                0);
        return kErrorInvalidArgument;
    }

    void* pixels = nullptr;
    if (AndroidBitmap_lockPixels(env, bitmap, &pixels) != ANDROID_BITMAP_RESULT_SUCCESS ||
        pixels == nullptr) {
        invoke_error(
                env,
                callback,
                kErrorInvalidArgument,
                "Failed to lock frame pixels",
                static_cast<int>(bitmap_info.width),
                static_cast<int>(bitmap_info.height));
        return kErrorInvalidArgument;
    }

    cv::Mat rgba(
            static_cast<int>(bitmap_info.height),
            static_cast<int>(bitmap_info.width),
            CV_8UC4,
            pixels,
            static_cast<size_t>(bitmap_info.stride));
    cv::Mat rgb;
    cv::cvtColor(rgba, rgb, cv::COLOR_RGBA2RGB);
    AndroidBitmap_unlockPixels(env, bitmap);

    rknn::RknnInferenceResult native_result;
    int return_code = kErrorInferenceFailed;
    jobject java_result = nullptr;
    {
        std::lock_guard<std::mutex> lock(handle->mutex);
        if (!handle->wrapper.is_loaded()) {
            return_code = kErrorEngineUnavailable;
        } else {
            return_code = handle->wrapper.infer(rgb, native_result);
            if (return_code == 0) {
                java_result = create_result_object(env, native_result);
                if (!java_result) {
                    clear_pending_exception(env);
                    return_code = kErrorCallbackFailed;
                }
            }
        }
    }

    if (return_code == 0 && java_result) {
        invoke_result(env, callback, java_result);
        return 0;
    }

    invoke_error(
            env,
            callback,
            return_code,
            "RKNN inference failed",
            static_cast<int>(bitmap_info.width),
            static_cast<int>(bitmap_info.height));
    return return_code;
}

JNIEXPORT void JNICALL
Java_com_rknn_engine_RknnEngine_nativeRelease(JNIEnv*, jclass, jlong handle_value) {
    delete handle_from(handle_value);
}

}
