/*
 * Copyright (C) 2017 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "BasePreparedModel.h"

#include <LegacyHalUtils.h>
#include <android-base/logging.h>
#include <android/log.h>
#include <cutils/properties.h>
#include <log/log.h>
#include <thread>
#include "ExecutionBurstServer.h"
#include "ValidateHal.h"

#undef LOG_TAG
#define DISABLE_ALL_QUANT
#define LOG_TAG "BasePreparedModel"

namespace android::hardware::neuralnetworks::nnhal {

using namespace android::nn;

static const Timing kNoTiming = {.timeOnDevice = UINT64_MAX, .timeInDriver = UINT64_MAX};

void BasePreparedModel::deinitialize() {
    ALOGV("Entering %s", __func__);
    mModelInfo->unmapRuntimeMemPools();

    ALOGV("Exiting %s", __func__);
}

template <typename T>
T getScalarData(const RunTimeOperandInfo& info) {
    // TODO: Check buffer is at least as long as size of data.
    T* data = reinterpret_cast<T*>(info.buffer);
    return data[0];
}

bool BasePreparedModel::initialize() {
    ALOGV("Entering %s", __func__);
    if (!mModelInfo->initRuntimeInfo()) {
        ALOGE("Failed to initialize Model runtime parameters!!");
        return false;
    }
    mNgraphNetCreator = std::make_shared<NgraphNetworkCreator>(mModelInfo, mTargetDevice);

    if (!mNgraphNetCreator->validateOperations()) return false;
    ALOGI("Generating IR Graph");
    auto ov_model = mNgraphNetCreator->generateGraph();
    if (ov_model == nullptr) {
        ALOGE("%s Openvino model generation failed", __func__);
        return false;
    }

    mPlugin = std::make_unique<IENetwork>(mTargetDevice, ov_model);
    mPlugin->loadNetwork();

    ALOGV("Exiting %s", __func__);
    return true;
}

static Return<void> notify(const sp<V1_0::IExecutionCallback>& callback, const ErrorStatus& status,
                           const hidl_vec<OutputShape>&, Timing) {
    return callback->notify(status);
}

static Return<void> notify(const sp<V1_2::IExecutionCallback>& callback, const ErrorStatus& status,
                           const hidl_vec<OutputShape>& outputShapes, Timing timing) {
    return callback->notify_1_2(status, outputShapes, timing);
}

static Return<void> notify(const sp<V1_3::IExecutionCallback>& callback, const ErrorStatus& status,
                           const hidl_vec<OutputShape>& outputShapes, Timing timing) {
    return callback->notify_1_3(convertToV1_3(status), outputShapes, timing);
}

namespace {
using time_point = std::chrono::steady_clock::time_point;
auto now() { return std::chrono::steady_clock::now(); };
auto microsecondsDuration(decltype(now()) end, decltype(now()) start) {
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
};
}  // namespace

template <typename T_IExecutionCallback>
Return<ErrorStatus> executeBase(const Request& request, MeasureTiming measure,
                                BasePreparedModel* preparedModel,
                                const sp<T_IExecutionCallback>& callback) {
    ALOGV("Entering %s", __func__);

    time_point driverStart;
    if (measure == MeasureTiming::YES) driverStart = now();

    if (callback.get() == nullptr) {
        ALOGE("invalid callback passed to execute");
        return ErrorStatus::INVALID_ARGUMENT;
    }
    if (!validateRequest(request, convertToV1_2(preparedModel->getModelInfo()->getModel()))) {
        notify(callback, ErrorStatus::INVALID_ARGUMENT, {}, kNoTiming);
        return ErrorStatus::INVALID_ARGUMENT;
    }

    // This thread is intentionally detached because the driver service
    // is expected to live forever.
    std::thread([preparedModel, request, measure, driverStart, callback] {
        asyncExecute(request, measure, preparedModel, driverStart, callback);
    }).detach();
    ALOGV("Exiting %s", __func__);
    return ErrorStatus::NONE;
}

template <typename T_IExecutionCallback>
void asyncExecute(const Request& request, MeasureTiming measure, BasePreparedModel* preparedModel,
                  time_point driverStart, const sp<T_IExecutionCallback>& callback) {
    ALOGV("Entering %s", __func__);
    auto modelInfo = preparedModel->getModelInfo();
    auto plugin = preparedModel->getPlugin();
    auto ngraphNw = preparedModel->getNgraphNwCreator();
    time_point driverEnd, deviceStart, deviceEnd;
    std::vector<RunTimePoolInfo> requestPoolInfos;
    auto errorStatus = modelInfo->setRunTimePoolInfosFromHidlMemories(request.pools);
    if (errorStatus != ErrorStatus::NONE) {
        ALOGE("Failed to set runtime pool info from HIDL memories");
        notify(callback, ErrorStatus::GENERAL_FAILURE, {}, kNoTiming);
        return;
    }

    for (size_t i = 0; i < request.inputs.size(); i++) {
        uint32_t len;
        auto inIndex = modelInfo->getModelInputIndex(i);
        void* srcPtr = modelInfo->getBlobFromMemoryPoolIn(request, i, len);

        const std::string& inputNodeName = ngraphNw->getNodeName(inIndex);
        if (inputNodeName == "") {
            ALOGD("Ignorning input at index(%d), since it is invalid", inIndex);
            continue;
        }
        ALOGD("Input index: %d layername : %s", inIndex, inputNodeName.c_str());
        auto destBlob = plugin->getInputBlob(i);
        auto inOperandType = modelInfo->getOperandType(inIndex);
        switch (inOperandType) {
            case OperandType::TENSOR_INT32: {
                int32_t* dest = destBlob.data<int32_t>();
                std::memcpy((uint8_t*)dest, (uint8_t*)srcPtr, len);
                break;
            }
            case OperandType::TENSOR_FLOAT16: {
                ov::float16* dest = destBlob.data<ov::float16>();
                std::memcpy((uint8_t*)dest, (uint8_t*)srcPtr, len);
                break;
            }
            case OperandType::TENSOR_FLOAT32: {
                uint8_t* dest = (uint8_t*)destBlob.data<float>();
                std::memcpy((uint8_t*)dest, (uint8_t*)srcPtr, len);
                break;
            }
            case OperandType::TENSOR_BOOL8: {
                uint8_t* dest = (uint8_t*)destBlob.data<bool>();
                std::memcpy((uint8_t*)dest, (uint8_t*)srcPtr, len);
                break;
            }
            case OperandType::TENSOR_QUANT8_ASYMM: {
                uint8_t* dest = (uint8_t*)destBlob.data<uint8_t>();
                std::memcpy((uint8_t*)dest, (uint8_t*)srcPtr, len);
                break;
            }
            case OperandType::TENSOR_QUANT8_SYMM:
            case OperandType::TENSOR_QUANT8_SYMM_PER_CHANNEL:
            case OperandType::TENSOR_QUANT8_ASYMM_SIGNED: {
                int8_t* dest = (int8_t*)destBlob.data<int8_t>();
                std::memcpy((int8_t*)dest, (int8_t*)srcPtr, len);
                break;
            }
            case OperandType::TENSOR_QUANT16_SYMM: {
                uint8_t* dest = (uint8_t*)destBlob.data<int16_t>();
                std::memcpy((uint8_t*)dest, (uint8_t*)srcPtr, len);
                break;
            }
            case OperandType::TENSOR_QUANT16_ASYMM: {
                uint8_t* dest = (uint8_t*)destBlob.data<uint16_t>();
                std::memcpy((uint8_t*)dest, (uint8_t*)srcPtr, len);
                break;
            }
            default:
                uint8_t* dest = (uint8_t*)destBlob.data<uint8_t>();
                std::memcpy((uint8_t*)dest, (uint8_t*)srcPtr, len);
                break;
        }
    }
    ALOGD("%s Run", __func__);

    if (measure == MeasureTiming::YES) deviceStart = now();
    try {
        plugin->infer();
    } catch (const std::exception& ex) {
        ALOGE("%s Exception !!! %s", __func__, ex.what());
        notify(callback, ErrorStatus::GENERAL_FAILURE, {}, kNoTiming);
        return;
    }
    if (measure == MeasureTiming::YES) deviceEnd = now();

    for (size_t i = 0; i < request.outputs.size(); i++) {
        auto outIndex = modelInfo->getModelOutputIndex(i);
        ALOGI("OutputIndex: %d", outIndex);
        const std::string& outputNodeName = ngraphNw->getNodeName(outIndex);
        if (outputNodeName == "") {
            ALOGD("Ignorning output at index(%d), since it is invalid", outIndex);
            continue;
        }
        ALOGD("Output index: %d layername : %s", outIndex, outputNodeName.c_str());
        auto srcBlob = plugin->getOutputBlob(i);
        auto operandType = modelInfo->getOperandType(outIndex);
        uint32_t actualLength = srcBlob.get_byte_size();
        uint32_t expectedLength = 0;
        void* destPtr = modelInfo->getBlobFromMemoryPoolOut(request, i, expectedLength);
        auto outputBlobDims = srcBlob.get_shape();

        bool outputSizeMismatch = false;
        if (actualLength != expectedLength) {
            ALOGE("%s Invalid length at outIndex(%d) Actual:%d Expected:%d", __func__, outIndex,
                  actualLength, expectedLength);
            outputSizeMismatch = true;
        }

        // TODO: bug identified with OV2021.4 where for Pad operation, if the output dimensions is 1
        // output dimension is coming as 0
        if ((outputBlobDims.size() == 0) && (actualLength != 0)) {
            std::vector<size_t> rdims = {1};
            modelInfo->updateOutputshapes(i, rdims, outputSizeMismatch ? false : true);
        } else
            modelInfo->updateOutputshapes(i, outputBlobDims, outputSizeMismatch ? false : true);

        if (outputSizeMismatch) {
            ALOGE(
                "Mismatch in actual and exepcted output sizes. Return with "
                "OUTPUT_INSUFFICIENT_SIZE error");
            notify(callback, ErrorStatus::OUTPUT_INSUFFICIENT_SIZE, modelInfo->getOutputShapes(),
                   kNoTiming);
            return;
        }

        switch (operandType) {
            case OperandType::TENSOR_INT32: {
                std::memcpy((uint8_t*)destPtr, (uint8_t*)srcBlob.data<int32_t>(),
                            srcBlob.get_byte_size());
                break;
            }
            case OperandType::TENSOR_FLOAT32: {
                std::memcpy((uint8_t*)destPtr, srcBlob.data<uint8_t>(), srcBlob.get_byte_size());
                break;
            }
            case OperandType::TENSOR_BOOL8: {
                std::memcpy((uint8_t*)destPtr, (uint8_t*)srcBlob.data<bool>(),
                            srcBlob.get_byte_size());
                break;
            }
            case OperandType::TENSOR_QUANT8_ASYMM: {
                std::memcpy((uint8_t*)destPtr, (uint8_t*)srcBlob.data<uint8_t>(),
                            srcBlob.get_byte_size());
                break;
            }
            case OperandType::TENSOR_QUANT8_SYMM:
            case OperandType::TENSOR_QUANT8_SYMM_PER_CHANNEL:
            case OperandType::TENSOR_QUANT8_ASYMM_SIGNED: {
                std::memcpy((int8_t*)destPtr, (int8_t*)srcBlob.data<int8_t>(),
                            srcBlob.get_byte_size());
                break;
            }
            case OperandType::TENSOR_FLOAT16: {
                std::memcpy((uint8_t*)destPtr, (uint8_t*)srcBlob.data<ov::float16>(),
                            srcBlob.get_byte_size());
                break;
            }
            case OperandType::TENSOR_QUANT16_SYMM: {
                std::memcpy((uint8_t*)destPtr, (uint8_t*)srcBlob.data<int16_t>(),
                            srcBlob.get_byte_size());
                break;
            }
            case OperandType::TENSOR_QUANT16_ASYMM: {
                std::memcpy((uint8_t*)destPtr, (uint8_t*)srcBlob.data<uint16_t>(),
                            srcBlob.get_byte_size());
                break;
            }
            default:
                std::memcpy((uint8_t*)destPtr, srcBlob.data<uint8_t>(), srcBlob.get_byte_size());
                break;
        }
    }

    if (!modelInfo->updateRequestPoolInfos()) {
        ALOGE("Failed to update the request pool infos");
    }

    Return<void> returned;
    if (measure == MeasureTiming::YES) {
        driverEnd = now();
        Timing timing = {.timeOnDevice = uint64_t(microsecondsDuration(deviceEnd, deviceStart)),
                         .timeInDriver = uint64_t(microsecondsDuration(driverEnd, driverStart))};
        returned = notify(callback, ErrorStatus::NONE, modelInfo->getOutputShapes(), timing);
    } else {
        returned = notify(callback, ErrorStatus::NONE, modelInfo->getOutputShapes(), kNoTiming);
    }
    if (!returned.isOk()) {
        ALOGE("hidl callback failed to return properly: %s", returned.description().c_str());
    }
    if (!modelInfo->unmapRuntimeMemPools()) {
        ALOGE("Failed to unmap the request pool infos");
    }
    ALOGV("Exiting %s", __func__);
}

static float avg_input_cp;
static float avg_output_cp;
static float avg_infer;
static float count;
static std::tuple<ErrorStatus, hidl_vec<V1_2::OutputShape>, Timing> executeSynchronouslyBase(
    const Request& request, MeasureTiming measure, BasePreparedModel* preparedModel,
    time_point driverStart) {
    ALOGV("Entering %s", __func__);
    auto modelInfo = preparedModel->getModelInfo();
    auto plugin = preparedModel->getPlugin();
    auto ngraphNw = preparedModel->getNgraphNwCreator();
    time_point driverEnd, deviceStart, deviceEnd;
    std::vector<RunTimePoolInfo> requestPoolInfos;
    auto errorStatus = modelInfo->setRunTimePoolInfosFromHidlMemories(request.pools);
    if (errorStatus != ErrorStatus::NONE) {
        ALOGE("Failed to set runtime pool info from HIDL memories");
        return {ErrorStatus::GENERAL_FAILURE, {}, kNoTiming};
    }
     
    //Measure Input/output copy 
    time_point tstartInMemcopy, tstopInMemcopy;
    time_point tstart_memcpy,tstop_memcpy;
    tstartInMemcopy = now();
    for (size_t i = 0; i < request.inputs.size(); i++) {
        uint32_t len;
        auto inIndex = modelInfo->getModelInputIndex(i);
        void* srcPtr = modelInfo->getBlobFromMemoryPoolIn(request, i, len);

        const std::string& inputNodeName = ngraphNw->getNodeName(inIndex);
        if (inputNodeName == "") {
            ALOGD("Ignorning input at index(%d), since it is invalid", inIndex);
            continue;
        }
        ALOGD("--------------Input index: %d layername : %s----------", inIndex, inputNodeName.c_str());
        auto destBlob = plugin->getInputBlob(i);
        auto inOperandType = modelInfo->getOperandType(inIndex);
        switch (inOperandType) {
            case OperandType::TENSOR_INT32: {
                int32_t* dest = destBlob.data<int32_t>();
                tstart_memcpy = now();
                std::memcpy((uint8_t*)dest, (uint8_t*)srcPtr, len);
                tstop_memcpy = now();
                ALOGV(" ---------------Input type TENSOR_INT32-----------------");
                break;
            }
            case OperandType::TENSOR_FLOAT16: {
                ov::float16* dest = destBlob.data<ov::float16>();
                tstart_memcpy = now();
                std::memcpy((uint8_t*)dest, (uint8_t*)srcPtr, len);
                tstop_memcpy = now();
                ALOGV(" ---------------Input type TENSOR_FLOAT16-----------------");
                break;
            }
            case OperandType::TENSOR_FLOAT32: {
                uint8_t* dest = (uint8_t*)destBlob.data<float>();
                tstart_memcpy = now();
                std::memcpy((uint8_t*)dest, (uint8_t*)srcPtr, len);
                tstop_memcpy = now();
                ALOGV(" ---------------Input type TENSOR_FLOAT32-----------------");
                break;
            }
            case OperandType::TENSOR_BOOL8: {
                uint8_t* dest = (uint8_t*)destBlob.data<bool>();
                tstart_memcpy = now();
                std::memcpy((uint8_t*)dest, (uint8_t*)srcPtr, len);
                tstop_memcpy = now();
                ALOGV(" ---------------Input type TENSOR_INT32-----------------");
                break;
            }
            case OperandType::TENSOR_QUANT8_ASYMM: {
                uint8_t* dest = (uint8_t*)destBlob.data<uint8_t>();
                tstart_memcpy = now();
                std::memcpy((uint8_t*)dest, (uint8_t*)srcPtr, len);
                tstop_memcpy = now();
                ALOGV(" ---------------Input type TENSOR_QUANT8_ASYMM-----------------");
                break;
            }
            case OperandType::TENSOR_QUANT8_SYMM:
            case OperandType::TENSOR_QUANT8_SYMM_PER_CHANNEL:
            case OperandType::TENSOR_QUANT8_ASYMM_SIGNED: {
                int8_t* dest = (int8_t*)destBlob.data<int8_t>();
                tstart_memcpy = now();
                std::memcpy((int8_t*)dest, (int8_t*)srcPtr, len);
                tstop_memcpy = now();
                ALOGV(" ---------------Input type TENSOR_QUANT8_ASYMM_SIGNED-----------------");
                break;
            }
            case OperandType::TENSOR_QUANT16_SYMM: {
                uint8_t* dest = (uint8_t*)destBlob.data<int16_t>();
                tstart_memcpy = now();
                std::memcpy((uint8_t*)dest, (uint8_t*)srcPtr, len);
                tstop_memcpy = now();
                ALOGV(" ---------------Input type TENSOR_QUANT16_SYMM-----------------");
                break;
            }
            case OperandType::TENSOR_QUANT16_ASYMM: {
                uint8_t* dest = (uint8_t*)destBlob.data<uint16_t>();
                tstart_memcpy = now();
                std::memcpy((uint8_t*)dest, (uint8_t*)srcPtr, len);
                tstop_memcpy = now();
                ALOGV(" ---------------Input type TENSOR_QUANT16_ASYMM-----------------");
                break;
            }
            default:
                uint8_t* dest = (uint8_t*)destBlob.data<uint8_t>();
                tstart_memcpy = now();
                std::memcpy((uint8_t*)dest, (uint8_t*)srcPtr, len);
                tstop_memcpy = now();
                ALOGV(" ---------------Input type defualt-----------------");
                break;
        }
        ALOGV("-------------Layer %d len=%d inputs.size()=%d - Input Memcpy time %d-----------------", i,len,request.inputs.size(), uint64_t(microsecondsDuration(tstop_memcpy, tstart_memcpy)));
    }
    tstopInMemcopy = now();
    float total_time = uint64_t(microsecondsDuration(tstopInMemcopy, tstartInMemcopy))/1000.0;
    ALOGV("---------------Input Memcpy time %d-----------------", uint64_t(microsecondsDuration(tstopInMemcopy, tstartInMemcopy)));
    count = count+1;
    avg_input_cp += total_time;
    float avg_time = (avg_input_cp)/count;
    ALOGV("---------------Input Memcpy Avg time %f %f %f-----------------",avg_time,count,avg_input_cp);
    ALOGD("%s Run", __func__);

    //Measure inference
    time_point tstart,tstop;//,tdeviceStart,tdeviceEnd,tdriverStart,tdriverEnd;
    //tdriverStart = driverStart;
    if (measure == MeasureTiming::YES) deviceStart = now();
    try {
        //ALOGV("Start ---------------Inference-----------------");
        tstart = now();
        plugin->infer();
        tstop = now();
        ALOGV("Exiting ---------------Inference time %d-----------------", uint64_t(microsecondsDuration(tstop, tstart)));
        total_time = uint64_t(microsecondsDuration(tstop, tstart))/1000.0;
        avg_infer += total_time;
        avg_time = (avg_infer)/count;
        ALOGV("---------------Inference Avg time %f-----------------",avg_time);
    } catch (const std::exception& ex) {
        ALOGE("%s Exception !!! %s", __func__, ex.what());
        return {ErrorStatus::GENERAL_FAILURE, {}, kNoTiming};
    }
    if (measure == MeasureTiming::YES) deviceEnd = now();

    //Measure output buffer copy
    tstart = now();
    for (size_t i = 0; i < request.outputs.size(); i++) {
        auto outIndex = modelInfo->getModelOutputIndex(i);
        ALOGI("OutputIndex: %d", outIndex);
        const std::string& outputNodeName = ngraphNw->getNodeName(outIndex);
        if (outputNodeName == "") {
            ALOGD("Ignorning output at index(%d), since it is invalid", outIndex);
            continue;
        }
        ALOGD("--------Output index: %d layername : %s------------", outIndex, outputNodeName.c_str());
        auto srcBlob = plugin->getOutputBlob(i);
        auto operandType = modelInfo->getOperandType(outIndex);
        uint32_t actualLength = srcBlob.get_byte_size();
        uint32_t expectedLength = 0;
        void* destPtr = modelInfo->getBlobFromMemoryPoolOut(request, i, expectedLength);
        auto outputBlobDims = srcBlob.get_shape();

        bool outputSizeMismatch = false;
        if (actualLength != expectedLength) {
            ALOGE("%s Invalid length at outIndex(%d) Actual:%d Expected:%d", __func__, outIndex,
                  actualLength, expectedLength);
            outputSizeMismatch = true;
        }

        // TODO: bug identified with OV2021.4 where for Pad operation, if the output dimensions is 1
        // output dimension is coming as 0
        if ((outputBlobDims.size() == 0) && (actualLength != 0)) {
            std::vector<size_t> rdims = {1};
            modelInfo->updateOutputshapes(i, rdims, outputSizeMismatch ? false : true);
        } else
            modelInfo->updateOutputshapes(i, outputBlobDims, outputSizeMismatch ? false : true);

        if (outputSizeMismatch) {
            ALOGE(
                "Mismatch in actual and exepcted output sizes. Return with "
                "OUTPUT_INSUFFICIENT_SIZE error");
            return {ErrorStatus::OUTPUT_INSUFFICIENT_SIZE, modelInfo->getOutputShapes(), kNoTiming};
        }

        switch (operandType) {
            case OperandType::TENSOR_INT32:
                tstart_memcpy = now(); 
                std::memcpy((uint8_t*)destPtr, (uint8_t*)srcBlob.data<int32_t>(),
                            srcBlob.get_byte_size());
                tstop_memcpy = now();
                ALOGV(" ---------------Output type TENSOR_INT32-----------------");
                break;
            case OperandType::TENSOR_FLOAT32: {
                tstart_memcpy = now();
                std::memcpy((uint8_t*)destPtr, (uint8_t*)srcBlob.data<float>(),
                            srcBlob.get_byte_size());
                tstop_memcpy = now();
                ALOGV(" ---------------Output type TENSOR_FLOAT32-----------------");
                break;
            }
            case OperandType::TENSOR_BOOL8: {
                tstart_memcpy = now();
                std::memcpy((uint8_t*)destPtr, (uint8_t*)srcBlob.data<bool>(),
                            srcBlob.get_byte_size());
                tstop_memcpy = now();
                ALOGV(" ---------------Output type TENSOR_BOOL8-----------------");
                break;
            }
            case OperandType::TENSOR_QUANT8_ASYMM: {
                tstart_memcpy = now();
                std::memcpy((uint8_t*)destPtr, (uint8_t*)srcBlob.data<uint8_t>(),
                            srcBlob.get_byte_size());
                tstop_memcpy = now();
                ALOGV(" ---------------Output type TENSOR_QUANT8_ASYMM-----------------");
                break;
            }
            case OperandType::TENSOR_QUANT8_SYMM:
            case OperandType::TENSOR_QUANT8_SYMM_PER_CHANNEL:
            case OperandType::TENSOR_QUANT8_ASYMM_SIGNED: {
                tstart_memcpy = now();
                std::memcpy((int8_t*)destPtr, (int8_t*)srcBlob.data<int8_t>(),
                            srcBlob.get_byte_size());
                tstop_memcpy = now();
                ALOGV(" ---------------Output type TENSOR_QUANT8-----------------");
                break;
            }
            case OperandType::TENSOR_FLOAT16: {
                tstart_memcpy = now();
                std::memcpy((uint8_t*)destPtr, (uint8_t*)srcBlob.data<ov::float16>(),
                            srcBlob.get_byte_size());
                tstop_memcpy = now();
                ALOGV(" ---------------Output type TENSOR_FLOAT16-----------------");
                break;
            }
            case OperandType::TENSOR_QUANT16_SYMM: {
                tstart_memcpy = now();
                std::memcpy((uint8_t*)destPtr, (uint8_t*)srcBlob.data<int16_t>(),
                            srcBlob.get_byte_size());
                tstop_memcpy = now();
                ALOGV(" ---------------Output type TENSOR_QUANT16_SYMM-----------------");
                break;
            }
            case OperandType::TENSOR_QUANT16_ASYMM: {
                tstart_memcpy = now();
                std::memcpy((uint8_t*)destPtr, (uint8_t*)srcBlob.data<uint16_t>(),
                            srcBlob.get_byte_size());
                tstop_memcpy = now();
                ALOGV(" ---------------Output type TENSOR_QUANT16_ASYMM-----------------");
                break;
            }
            default:
                tstart_memcpy = now();
                std::memcpy((uint8_t*)destPtr, srcBlob.data<uint8_t>(), srcBlob.get_byte_size());
                tstop_memcpy = now();
                ALOGV(" ---------------Output type default-----------------");
                break;
        }
        ALOGV("-------------Layer %d len=%d outputs.size=%d - Output Memcpy time %d -----------------", i, srcBlob.get_byte_size(), request.outputs.size(), uint64_t(microsecondsDuration(tstop_memcpy, tstart_memcpy)));
    }
    tstop = now();
    ALOGV("---------------Output Memcpy time %d-----------------",  uint64_t(microsecondsDuration(tstop, tstart)));
    total_time = uint64_t(microsecondsDuration(tstop, tstart))/1000.0;
    avg_output_cp += total_time;
    avg_time = (avg_output_cp)/count;
    ALOGV("---------------Output Memcpy Avg time %f-----------------",avg_time);

    if (!modelInfo->updateRequestPoolInfos()) {
        ALOGE("Failed to update the request pool infos");
        return {ErrorStatus::GENERAL_FAILURE, {}, kNoTiming};
    }

    if (measure == MeasureTiming::YES) {
        driverEnd = now();
        Timing timing = {.timeOnDevice = uint64_t(microsecondsDuration(deviceEnd, deviceStart)),
                         .timeInDriver = uint64_t(microsecondsDuration(driverEnd, driverStart))};
        return {ErrorStatus::NONE, modelInfo->getOutputShapes(), timing};
    }
    ALOGV("Exiting %s", __func__);
    if (!modelInfo->unmapRuntimeMemPools()) {
        ALOGE("Failed to unmap the request pool infos");
    }
    return {ErrorStatus::NONE, modelInfo->getOutputShapes(), kNoTiming};
}

Return<void> BasePreparedModel::executeSynchronously(const Request& request, MeasureTiming measure,
                                                     executeSynchronously_cb cb) {
    ALOGV("Entering %s", __func__);
    time_point driverStart;
    if (measure == MeasureTiming::YES) driverStart = now();

    if (!validateRequest(request, convertToV1_2(mModelInfo->getModel()))) {
        cb(ErrorStatus::INVALID_ARGUMENT, {}, kNoTiming);
        return Void();
    }
    auto [status, outputShapes, timing] =
        executeSynchronouslyBase(request, measure, this, driverStart);
    cb(status, std::move(outputShapes), timing);
    ALOGV("Exiting %s", __func__);
    return Void();
}

Return<void> BasePreparedModel::executeSynchronously_1_3(const V1_3::Request& request,
                                                         V1_2::MeasureTiming measure,
                                                         const V1_3::OptionalTimePoint&,
                                                         const V1_3::OptionalTimeoutDuration&,
                                                         executeSynchronously_1_3_cb cb) {
    ALOGV("Entering %s", __func__);
    time_point driverStart;
    if (measure == MeasureTiming::YES) driverStart = now();

    if (!validateRequest(request, mModelInfo->getModel())) {
        cb(V1_3::ErrorStatus::INVALID_ARGUMENT, {}, kNoTiming);
        return Void();
    }
    auto [status, outputShapes, timing] =
        executeSynchronouslyBase(convertToV1_0(request), measure, this, driverStart);
    cb(convertToV1_3(status), std::move(outputShapes), timing);
    ALOGV("Exiting %s", __func__);
    return Void();
}

Return<void> BasePreparedModel::configureExecutionBurst(
    const sp<V1_2::IBurstCallback>& callback,
    const MQDescriptorSync<V1_2::FmqRequestDatum>& requestChannel,
    const MQDescriptorSync<V1_2::FmqResultDatum>& resultChannel, configureExecutionBurst_cb cb) {
    ALOGV("Entering %s", __func__);
    const sp<V1_2::IBurstContext> burst =
        ExecutionBurstServer::create(callback, requestChannel, resultChannel, this);

    if (burst == nullptr) {
        cb(ErrorStatus::GENERAL_FAILURE, {});
        ALOGI("%s GENERAL_FAILURE", __func__);
    } else {
        cb(ErrorStatus::NONE, burst);
        ALOGI("%s burst created", __func__);
    }
    ALOGV("Exiting %s", __func__);
    return Void();
}

Return<ErrorStatus> BasePreparedModel::execute(const Request& request,
                                               const sp<V1_0::IExecutionCallback>& callback) {
    ALOGV("Entering %s", __func__);
    return executeBase(request, MeasureTiming::NO, this, callback);
}

Return<ErrorStatus> BasePreparedModel::execute_1_2(const Request& request, MeasureTiming measure,
                                                   const sp<V1_2::IExecutionCallback>& callback) {
    ALOGV("Entering %s", __func__);
    return executeBase(request, measure, this, callback);
}

Return<V1_3::ErrorStatus> BasePreparedModel::execute_1_3(
    const V1_3::Request& request, V1_2::MeasureTiming measure, const V1_3::OptionalTimePoint&,
    const V1_3::OptionalTimeoutDuration&, const sp<V1_3::IExecutionCallback>& callback) {
    ALOGV("Entering %s", __func__);
    return convertToV1_3(executeBase(convertToV1_0(request), measure, this, callback));
}

Return<void> BasePreparedModel::executeFenced(const V1_3::Request& request1_3,
                                              const hidl_vec<hidl_handle>& waitFor,
                                              V1_2::MeasureTiming measure,
                                              const V1_3::OptionalTimePoint& halDeadline,
                                              const V1_3::OptionalTimeoutDuration&,
                                              const V1_3::OptionalTimeoutDuration& duration,
                                              executeFenced_cb cb) {
    ALOGV("Entering %s", __func__);

    time_point driverStart, driverEnd;
    if (measure == MeasureTiming::YES) driverStart = now();

    if (!validateRequest(request1_3, mModelInfo->getModel(), /*allowUnspecifiedOutput=*/false)) {
        cb(V1_3::ErrorStatus::INVALID_ARGUMENT, hidl_handle(nullptr), nullptr);
        return Void();
    }

    const auto deadline = makeDeadline(halDeadline);
    if (hasDeadlinePassed(deadline)) {
        cb(V1_3::ErrorStatus::MISSED_DEADLINE_PERSISTENT, hidl_handle(nullptr), nullptr);
        return Void();
    }

    // Wait for the dependent events to signal
    for (const auto& fenceHandle : waitFor) {
        if (!fenceHandle.getNativeHandle()) {
            cb(V1_3::ErrorStatus::INVALID_ARGUMENT, hidl_handle(nullptr), nullptr);
            return Void();
        }
        const int syncFenceFd = fenceHandle.getNativeHandle()->data[0];
        if (syncWait(syncFenceFd, -1) != FenceState::SIGNALED) {
            ALOGV("%s syncWait failed", __func__);
            cb(V1_3::ErrorStatus::GENERAL_FAILURE, hidl_handle(nullptr), nullptr);
            return Void();
        }
    }

    auto errorStatus = mModelInfo->setRunTimePoolInfosFromHidlMemories(request1_3.pools);
    if (errorStatus != V1_3::ErrorStatus::NONE) {
        ALOGE("Failed to set runtime pool info from HIDL memories");
        cb(errorStatus, hidl_handle(nullptr), nullptr);
        return Void();
    }

    // rest of the interfaces are based on 1.0 request
    auto request = convertToV1_0(request1_3);

    time_point driverAfterFence;
    if (measure == MeasureTiming::YES) driverAfterFence = now();

    for (size_t i = 0; i < request.inputs.size(); i++) {
        uint32_t len;
        auto inIndex = mModelInfo->getModelInputIndex(i);
        void* srcPtr = mModelInfo->getBlobFromMemoryPoolIn(request, i, len);

        const std::string& inputNodeName = mNgraphNetCreator->getNodeName(inIndex);
        if (inputNodeName == "") {
            ALOGD("Ignorning input at index(%d), since it is invalid", inIndex);
            continue;
        }
        ALOGD("Input index: %d layername : %s", inIndex, inputNodeName.c_str());
        auto destBlob = mPlugin->getInputBlob(i);
        auto inOperandType = mModelInfo->getOperandType(inIndex);
        switch (inOperandType) {
            case OperandType::TENSOR_INT32: {
                int32_t* dest = destBlob.data<int32_t>();
                std::memcpy((uint8_t*)dest, (uint8_t*)srcPtr, len);
                break;
            }
            case OperandType::TENSOR_FLOAT16: {
                ov::float16* dest = destBlob.data<ov::float16>();
                std::memcpy((uint8_t*)dest, (uint8_t*)srcPtr, len);
                break;
            }
            case OperandType::TENSOR_FLOAT32: {
                uint8_t* dest = (uint8_t*)destBlob.data<float>();
                std::memcpy((uint8_t*)dest, (uint8_t*)srcPtr, len);
                break;
            }
            case OperandType::TENSOR_BOOL8: {
                uint8_t* dest = (uint8_t*)destBlob.data<bool>();
                std::memcpy((uint8_t*)dest, (uint8_t*)srcPtr, len);
                break;
            }
            case OperandType::TENSOR_QUANT8_ASYMM: {
                uint8_t* dest = (uint8_t*)destBlob.data<uint8_t>();
                std::memcpy((uint8_t*)dest, (uint8_t*)srcPtr, len);
                break;
            }
            case OperandType::TENSOR_QUANT8_SYMM:
            case OperandType::TENSOR_QUANT8_SYMM_PER_CHANNEL:
            case OperandType::TENSOR_QUANT8_ASYMM_SIGNED: {
                int8_t* dest = (int8_t*)destBlob.data<int8_t>();
                std::memcpy((int8_t*)dest, (int8_t*)srcPtr, len);
                break;
            }
            case OperandType::TENSOR_QUANT16_SYMM: {
                uint8_t* dest = (uint8_t*)destBlob.data<int16_t>();
                std::memcpy((uint8_t*)dest, (uint8_t*)srcPtr, len);
                break;
            }
            case OperandType::TENSOR_QUANT16_ASYMM: {
                uint8_t* dest = (uint8_t*)destBlob.data<uint16_t>();
                std::memcpy((uint8_t*)dest, (uint8_t*)srcPtr, len);
                break;
            }
            default:
                uint8_t* dest = (uint8_t*)destBlob.data<uint8_t>();
                std::memcpy((uint8_t*)dest, (uint8_t*)srcPtr, len);
                break;
        }
    }

    ALOGD("%s Run", __func__);

    time_point deviceStart, deviceEnd;
    if (measure == MeasureTiming::YES) deviceStart = now();
    try {
        mPlugin->infer();
    } catch (const std::exception& ex) {
        ALOGE("%s Exception !!! %s", __func__, ex.what());
        cb(V1_3::ErrorStatus::GENERAL_FAILURE, hidl_handle(nullptr), nullptr);
        return Void();
    }
    if (measure == MeasureTiming::YES) deviceEnd = now();

    for (size_t i = 0; i < request.outputs.size(); i++) {
        auto outIndex = mModelInfo->getModelOutputIndex(i);
        ALOGI("OutputIndex: %d", outIndex);
        const std::string& outputNodeName = mNgraphNetCreator->getNodeName(outIndex);
        if (outputNodeName == "") {
            ALOGD("Ignorning output at index(%d), since it is invalid", outIndex);
            continue;
        }
        ALOGD("Output index: %d layername : %s", outIndex, outputNodeName.c_str());
        auto srcBlob = mPlugin->getOutputBlob(i);
        auto operandType = mModelInfo->getOperandType(outIndex);
        uint32_t actualLength = srcBlob.get_byte_size();
        uint32_t expectedLength = 0;
        void* destPtr = mModelInfo->getBlobFromMemoryPoolOut(request, i, expectedLength);
        auto outDims = srcBlob.get_shape();

        if (actualLength != expectedLength) {
            ALOGE("%s Invalid length(%d) at outIndex(%d)", __func__, actualLength, outIndex);
            // Notify Insufficient Buffer Length to modelInfo
            mModelInfo->updateOutputshapes(i, outDims, false);
            cb(V1_3::ErrorStatus::OUTPUT_INSUFFICIENT_SIZE, hidl_handle(nullptr), nullptr);
            return Void();
        } else {
            mModelInfo->updateOutputshapes(i, outDims);
        }
        switch (operandType) {
            case OperandType::TENSOR_INT32: {
                std::memcpy((uint8_t*)destPtr, (uint8_t*)srcBlob.data<int32_t>(),
                            srcBlob.get_byte_size());
                break;
            }
            case OperandType::TENSOR_FLOAT32: {
                std::memcpy((uint8_t*)destPtr, (uint8_t*)srcBlob.data<float>(),
                            srcBlob.get_byte_size());
                break;
            }
            case OperandType::TENSOR_BOOL8: {
                std::memcpy((uint8_t*)destPtr, (uint8_t*)srcBlob.data<bool>(),
                            srcBlob.get_byte_size());
                break;
            }
            case OperandType::TENSOR_QUANT8_ASYMM: {
                std::memcpy((uint8_t*)destPtr, (uint8_t*)srcBlob.data<uint8_t>(),
                            srcBlob.get_byte_size());
                break;
            }
            case OperandType::TENSOR_QUANT8_SYMM:
            case OperandType::TENSOR_QUANT8_SYMM_PER_CHANNEL:
            case OperandType::TENSOR_QUANT8_ASYMM_SIGNED: {
                std::memcpy((int8_t*)destPtr, (int8_t*)srcBlob.data<int8_t>(),
                            srcBlob.get_byte_size());
                break;
            }
            case OperandType::TENSOR_FLOAT16: {
                std::memcpy((uint8_t*)destPtr, (uint8_t*)srcBlob.data<ov::float16>(),
                            srcBlob.get_byte_size());
                break;
            }
            case OperandType::TENSOR_QUANT16_SYMM: {
                std::memcpy((uint8_t*)destPtr, (uint8_t*)srcBlob.data<int16_t>(),
                            srcBlob.get_byte_size());
                break;
            }
            case OperandType::TENSOR_QUANT16_ASYMM: {
                std::memcpy((uint8_t*)destPtr, (uint8_t*)srcBlob.data<uint16_t>(),
                            srcBlob.get_byte_size());
                break;
            }
            default:
                std::memcpy((uint8_t*)destPtr, srcBlob.data<uint8_t>(), srcBlob.get_byte_size());
                break;
        }
    }

    if (!mModelInfo->updateRequestPoolInfos()) {
        ALOGE("Failed to update the request pool infos");
    }

    Timing timingSinceLaunch = {.timeOnDevice = UINT64_MAX, .timeInDriver = UINT64_MAX};
    Timing timingAfterFence = {.timeOnDevice = UINT64_MAX, .timeInDriver = UINT64_MAX};

    if (measure == MeasureTiming::YES) {
        driverEnd = now();
        timingSinceLaunch = {
            .timeOnDevice = uint64_t(microsecondsDuration(deviceEnd, deviceStart)),
            .timeInDriver = uint64_t(microsecondsDuration(driverEnd, driverStart))};
        timingAfterFence = {
            .timeOnDevice = uint64_t(microsecondsDuration(deviceEnd, deviceStart)),
            .timeInDriver = uint64_t(microsecondsDuration(driverEnd, driverAfterFence))};
    }

    sp<BaseFencedExecutionCallback> fencedExecutionCallback = new BaseFencedExecutionCallback(
        timingSinceLaunch, timingAfterFence, V1_3::ErrorStatus::NONE);
    cb(V1_3::ErrorStatus::NONE, hidl_handle(nullptr), fencedExecutionCallback);
    ALOGV("Exiting %s", __func__);
    return Void();
}

}  // namespace android::hardware::neuralnetworks::nnhal
