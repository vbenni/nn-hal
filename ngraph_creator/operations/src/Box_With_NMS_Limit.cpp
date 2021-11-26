#include <Box_With_NMS_Limit.hpp>
#define LOG_TAG "Box_With_NMS_Limit"

namespace android {
namespace hardware {
namespace neuralnetworks {
namespace nnhal {

Box_With_NMS_Limit::Box_With_NMS_Limit(int operationIndex) : OperationsBase(operationIndex) {
    mDefaultOutputIndex = sModelInfo->getOperationOutput(mNnapiOperationIndex, 0);
}

bool Box_With_NMS_Limit::validate() {
    ALOGV("%s Entering", __func__);
    // check output type
    if (!checkOutputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32)) {
        ALOGE("%s Output operand 0 is not of type FP32. Unsupported operation", __func__);
        return false;
    }

    // Check Input Type
    if (!checkInputOperandType(0, (int32_t)OperandType::TENSOR_FLOAT32)) {
        ALOGE("%s Input operand 0 is not of type FP32. Unsupported operation", __func__);
        return false;
    }

    ALOGV("%s PASSED", __func__);
    return true;
}

std::shared_ptr<ngraph::Node> Box_With_NMS_Limit::createNode() {
    ALOGV("-------%s Entering", __func__);

    std::shared_ptr<ngraph::Node> inputNode;
    bool useNchw = false;
    const auto& inputsSize = sModelInfo->getOperationInputsSize(mNnapiOperationIndex);
    ALOGD("%s inputsSize %lu", __func__, inputsSize);

    // Read inputs
    auto scores_node = getInputNode(0);  // score[num_rois,classes]  --> [batch,classes,num_rois]
    auto bboxes_node = getInputNode(1);  // shape[num_rois,classes*4] --> [batch,num_rois,4]
    auto bi_node = getInputNode(2); //batch[num_rois]
    auto score_threshold = sModelInfo->ParseOperationInput<float_t>(mNnapiOperationIndex, 3);
    auto score_threshold_node = createConstNode(ngraph::element::f32, {1}, convertToVector(score_threshold));
    auto max_bboxes = sModelInfo->ParseOperationInput<int32_t>(mNnapiOperationIndex, 4);
    auto max_bboxes_node = createConstNode(ngraph::element::i32, {1}, convertToVector(max_bboxes));
    auto nms_method = sModelInfo->ParseOperationInput<uint32_t>(mNnapiOperationIndex,
                                                                5);  // 0:hard, 1:linear, 2:gaussian
    auto nms_method_node = createConstNode(ngraph::element::i32, {1}, convertToVector(nms_method));
    auto iou_threshold = sModelInfo->ParseOperationInput<float_t>(mNnapiOperationIndex, 6);
    auto iou_threshold_node = createConstNode(ngraph::element::f32, {1}, convertToVector(iou_threshold));
    auto sigma = sModelInfo->ParseOperationInput<float_t>(mNnapiOperationIndex, 7);
    auto sigma_node = createConstNode(ngraph::element::f32, {1}, convertToVector(sigma));
    auto nms_score_threshold = sModelInfo->ParseOperationInput<float_t>(mNnapiOperationIndex, 8);
    auto nms_score_threshold_node = createConstNode(ngraph::element::f32, {1}, convertToVector(nms_score_threshold));

    ALOGD("------inputs parsed successfully---------\n");


    

    // const auto& score_thresholdOperandIndex =
    //     sModelInfo->getOperationInput(mNnapiOperationIndex, 3);
    // auto score_threshold_vec = sModelInfo->GetConstVecOperand<int32_t>(score_thresholdOperandIndex);
    // const auto score_threshold_node = createConstNode(
    //     ngraph::element::f32, ngraph::Shape{score_threshold_vec.size(), 1}, score_threshold_vec);
    // const auto& iou_thresholdOperandIndex = sModelInfo->getOperationInput(mNnapiOperationIndex, 6);
    // auto iou_threshold_vec = sModelInfo->GetConstVecOperand<int32_t>(iou_thresholdOperandIndex);
    // const auto iou_threshold_node = createConstNode(
    //     ngraph::element::f32, ngraph::Shape{iou_threshold_vec.size(), 1}, iou_threshold_vec);
    // const auto& nms_methodOperandIndex = sModelInfo->getOperationInput(mNnapiOperationIndex, 5);
    // auto nms_method_vec = sModelInfo->GetConstVecOperand<int32_t>(nms_methodOperandIndex);
    // const auto nms_method_node = createConstNode(
    //     ngraph::element::i32, ngraph::Shape{nms_method_vec.size(), 1}, nms_method_vec);
    // const auto& max_bboxesOperandIndex = sModelInfo->getOperationInput(mNnapiOperationIndex, 4);
    // auto max_bboxes_vec = sModelInfo->GetConstVecOperand<int32_t>(nms_methodOperandIndex);
    // const auto max_bboxes_node = createConstNode(
    //     ngraph::element::i32, ngraph::Shape{nms_method_vec.size(), 1}, max_bboxes_vec);

    // Hard: score_new = score_old * (1 if IoU < threshold else 0)
    // Linear: score_new = score_old * (1 if IoU < threshold else 1 - IoU)
    // Gaussian: score_new = score_old * exp(- IoU^2 / sigma)

    // Find the number of batches
    //  int32_t batch_axis = 0;
    //  auto batchIndex_node = getInputNode(2);
    //  auto k_node = createConstNode(ngraph::element::i32, {}, convertToVector(1));
    //  const auto topk = std::make_shared<ngraph::opset3::TopK>(
    //      input, k_node, batch_axis, ngraph::opset3::TopK::Mode::MAX,
    //      ngraph::opset3::TopK::SortType::VALUE);

    std::vector<ngraph::Output<ngraph::Node>> inputs;
    auto axis = 1;
    // add bi node to inputs for concat
    // reshape batchindex bi
    std::vector<uint32_t> shape(bi_node->get_shape().size(), 1);
    auto shapeNode = createConstNode(ngraph::element::i32, ngraph::Shape{shape.size()}, shape);
    bi_node = std::make_shared<ngraph::opset3::Reshape>(bi_node, shapeNode, true);
    ALOGD("------bi_node reshaped---------\n");

    // const auto& biOperandIndex = sModelInfo->getOperationInput(mNnapiOperationIndex, 2);
    // auto bi_vec = sModelInfo->GetConstVecOperand<int32_t>(biOperandIndex);
    // // Find the number of batches and max bbox
    // auto batches_num = max_element(std::begin(bi_vec), std::end(bi_vec));  // C++11
    // std::map<int, int> freqMap;
    // int maxFreq = 0;
    // int mostFreqElement = 0;
    // for (int x : bi_vec) {
    //     int f = ++freqMap[x];
    //     if (f > maxFreq) {
    //         maxFreq = f;
    //         mostFreqElement = x;
    //     }
    // }
    // int bbox_dim = mostFreqElement;
    // ALOGV("--------------------------bbox_dim = %d  mostFreqElement = %d----------------", bbox_dim,mostFreqElement);
    //bi_node = createConstNode(ngraph::element::i32, ngraph::Shape{bi_vec.size(), 1}, bi_vec);

    inputs.push_back(bi_node);

    
    // add score node to inputs for concat
    // transpose the score node first
    std::shared_ptr<ngraph::Node> order =
        createConstNode(ngraph::element::f32, {0}, convertToVector(0));
    std::shared_ptr<ngraph::Node> transposeScoreNode;
    transposeScoreNode = std::make_shared<ngraph::opset3::Transpose>(scores_node, order);
    inputs.push_back(transposeScoreNode);
    std::shared_ptr<ngraph::Node> scoreNode =
        std::make_shared<ngraph::opset3::Concat>(inputs, axis);
    ALOGV("%s Concatinated score_node created", __func__);

    std::vector<ngraph::Output<ngraph::Node>> inputs1;
    inputs1.push_back(bi_node);
    // add bboxes node to inputs for concat
    auto inputIndex1 = sModelInfo->getOperationInput(mNnapiOperationIndex, 1);
    auto inputOp1 = mNgraphNodes->getOperationOutput(inputIndex1);
    inputs1.push_back(inputOp1);
    std::shared_ptr<ngraph::Node> bbNode = std::make_shared<ngraph::opset3::Concat>(inputs1, axis);
    ALOGV("%s Concatinated bbnode created", __func__);

    std::shared_ptr<ngraph::Node> outputNode = std::make_shared<ngraph::op::v5::NonMaxSuppression>(
        bbNode, scoreNode, max_bboxes_node, iou_threshold_node, score_threshold_node,
        nms_method_node, ngraph::op::v5::NonMaxSuppression::BoxEncodingType::CORNER, true,
        ngraph::element::i64);
    ALOGV("%s PASSED", __func__);

    return outputNode;
}

}  // namespace nnhal
}  // namespace neuralnetworks
}  // namespace hardware
}  // namespace android
