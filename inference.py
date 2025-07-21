import cv2
import numpy as np
import onnxruntime

INPUT_W = 640
INPUT_H = 640

class OnnxModel(object):
    def __init__(self, model_path):
        sess_options = onnxruntime.SessionOptions()
        onnx_gpu = (onnxruntime.get_device() == 'GPU')
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if onnx_gpu else ['CPUExecutionProvider']
        self.sess = onnxruntime.InferenceSession(model_path, sess_options, providers=providers)
        self._input_names = [item.name for item in self.sess.get_inputs()]
        self._output_names = [item.name for item in self.sess.get_outputs()]
        
    @property
    def input_names(self):
        return self._input_names
        
    @property
    def output_names(self):
        return self._output_names
        
    def forward(self, inputs):
        to_list_flag = False
        if not isinstance(inputs, (tuple, list)):
            inputs = [inputs]
            to_list_flag = True
        input_feed = {name: input for name, input in zip(self.input_names, inputs)}
        outputs = self.sess.run(self.output_names, input_feed)
        if (len(self.output_names) == 1) and to_list_flag:
            return outputs[0]
        else:
            return outputs


def preprocess_image(image_path):
    """
    description: Read an image from image path, convert it to RGB,
                 resize and pad it to target size, normalize to [0,1],
                 transform to NCHW format.
    param:
        image_path: str, image path
    return:
        image:  the processed image
        image_raw: the original image
        h: original height
        w: original width
    """
    image_raw = cv2.imread(image_path)         # 1.opencv读入图片
    h, w, c = image_raw.shape                  # 2.记录图片大小
    image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)  # 3. BGR2RGB
    # Calculate widht and height and paddings
    r_w = INPUT_W / w  # INPUT_W=INPUT_H=640  # 4.计算宽高缩放的倍数 r_w,r_h
    r_h = INPUT_H / h
    if r_h > r_w:       # 5.如果原图的高小于宽(长边），则长边缩放到640，短边按长边缩放比例缩放
        tw = INPUT_W
        th = int(r_w * h)
        tx1 = tx2 = 0
        ty1 = int((INPUT_H - th) / 2)  # ty1=（640-短边缩放的长度）/2 ，这部分是YOLOv5为加速推断而做的一个图像缩放算法
        ty2 = INPUT_H - th - ty1       # ty2=640-短边缩放的长度-ty1
    else:
        tw = int(r_h * w)
        th = INPUT_H
        tx1 = int((INPUT_W - tw) / 2)
        tx2 = INPUT_W - tw - tx1
        ty1 = ty2 = 0
    # Resize the image with long side while maintaining ratio
    image = cv2.resize(image, (tw, th),interpolation=cv2.INTER_LINEAR)  # 6.图像resize,按照cv2.INTER_LINEAR方法
    # Pad the short side with (128,128,128)   
    image = cv2.copyMakeBorder(
        # image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (128, 128, 128)
        image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (114, 114, 114)

    )  # image:图像， ty1, ty2.tx1,tx2: 相应方向上的边框宽度，添加的边界框像素值为常数，value填充的常数值
    image = image.astype(np.float32)   # 7.unit8-->float
    # Normalize to [0,1]
    image /= 255.0    # 8. 逐像素点除255.0
    # HWC to CHW format:
    image = np.transpose(image, [2, 0, 1])   # 9. HWC2CHW
    # CHW to NCHW format
    image = np.expand_dims(image, axis=0)    # 10.CWH2NCHW
    # Convert the image to row-major order, also known as "C order":
    image = np.ascontiguousarray(image)  # 11.ascontiguousarray函数将一个内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快
    return image, image_raw, h, w  # 处理后的图像，原图， 原图的h,w


def yolo_postprocess(
    pred: np.ndarray,
    img: np.ndarray,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    num_classes: int = 80,
    xywh_to_xyxy: bool = True,
    with_conf: bool = True
) -> list:
    """
    YOLO 模型后处理函数，解析输出为检测框、类别、置信度等信息

    :param pred: 模型输出，shape = [1, num_anchors, 4 + 1 + num_classes]
    :param img: 原始输入图像，HWC or CHW 格式
    :param conf_threshold: 置信度阈值
    :param iou_threshold: NMS 的 IOU 阈值
    :param num_classes: 类别数量
    :param xywh_to_xyxy: 是否将坐标从 xywh 转换为 xyxy
    :param with_conf: 是否包含 objectness 置信度
    :return: list of detections, each detection is a tuple:
             (x1, y1, x2, y2, conf, class_id)
    """
    # 获取图像尺寸
    if img.shape[0] == 3 and img.ndim == 3:  # CHW -> HWC
        img = img.transpose(1, 2, 0)
    img_h, img_w = img.shape[:2]

    # 假设 pred 是 [1, num_anchors, 4 + 1 + num_classes]
    pred = pred[0]  # shape: [num_anchors, 4 + 1 + num_classes]

    # 提取 objectness 置信度
    if with_conf:
        obj_conf = pred[:, 4:5]  # shape: [num_anchors, 1]
        cls_conf = pred[:, 5:]  # shape: [num_anchors, num_classes]
        scores = obj_conf * cls_conf  # 合并 objectness 和 class 置信度
    else:
        scores = pred[:, 4:]  # 如果输出已经合并了置信度

    # 找出最大置信度的类别
    class_ids = np.argmax(scores, axis=1)
    confidences = np.max(scores, axis=1)

    # 过滤掉置信度低的预测
    keep = confidences > conf_threshold
    if not np.any(keep):
        return []

    boxes = pred[keep, :4]
    confidences = confidences[keep]
    class_ids = class_ids[keep]

    # 将坐标从 [0,1] 映射到图像实际尺寸
    boxes = boxes.copy()
    boxes[:, 0::2] *= img_w
    boxes[:, 1::2] *= img_h

    # xywh -> xyxy
    if xywh_to_xyxy:
        x_center, y_center, w, h = boxes.T
        x1 = x_center - w / 2
        y1 = y_center - h / 2
        x2 = x_center + w / 2
        y2 = y_center + h / 2
        boxes = np.stack([x1, y1, x2, y2], axis=1)

    # 转为整数坐标
    boxes = boxes.round().astype(np.int32).clip(0, [img_w - 1, img_h - 1] * 2)

    # 应用 NMS（非极大值抑制）
    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes.tolist(),
        scores=confidences.tolist(),
        score_threshold=conf_threshold,
        nms_threshold=iou_threshold
    )

    if len(indices) == 0:
        return []

    # 获取最终检测结果
    detections = []
    for i in indices.flatten():
        x1, y1, x2, y2 = boxes[i]
        conf = confidences[i]
        cls_id = class_ids[i]
        detections.append((x1, y1, x2, y2, conf, cls_id))

    return detections


if __name__ == "__main__":
    model = OnnxModel('runs/detect/train2/weights/best.onnx')

    image, image_raw, h, w = preprocess_image('datasets/coco8/images/val/000000000036.jpg')

    output = model.forward(image)
    
    yolo_postprocess(output, image)
    print(len(output))
