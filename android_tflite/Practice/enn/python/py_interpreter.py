import os
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

class_number_to_str_name = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "sofa", "potted plant", "bed", "dining table", "toilet",
    "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]

workspace = Path("D:\\workspace\\jupyter\MachineLearning\\android_tflite\\Practice\\enn\\python")

default_model = "default_yolo_v4_tiny_float.tflite"
samsung_model = "samsung_yolo_v4_tiny_float.tflite"


model_path = str(workspace / default_model)

interpreter = tf.lite.Interpreter(model_path=model_path)

interpreter.allocate_tensors()

# 모델 분석

# 입력 및 출력 정보 얻기
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 입력 텐서 정보 출력
print("== input tensor information ==")
for i, detail in enumerate(input_details):
    print(f"input tensor {i}:")
    print(f"\tname: {detail['name']}")
    print(f"\tshape: {detail['shape']}")
    print(f"\tdtype: {detail['dtype']}")
    print()

# 출력 텐서 정보 출력
print("== output tensor information ==")
for i, detail in enumerate(output_details):
    print(f"output tensor {i}:")
    print(f"\tname: {detail['name']}")
    print(f"\tshape: {detail['shape']}")
    print(f"\tdtype: {detail['dtype']}")
    print()

# 입력 데이터 (입력 이미지) 전처리
image_path = str(workspace / "bird.jpg") 

input_size = input_details[0]['shape'][1:3] # model shape 정보에서 size를 추출
input_width = input_details[0]['shape'][1]
input_height = input_details[0]['shape'][2]

input_data = Image.open(image_path)
input_data = input_data.resize(([input_width, input_height]))
input_data = np.array(input_data, dtype=np.float32)
input_data = input_data / 255.0
input_data = np.expand_dims(input_data, axis=0)  

# 입력 텐서 설정
interpreter.set_tensor(input_details[0]['index'], input_data)

# 추론 실행
interpreter.invoke()

# 출력 텐서 가져오기
output_boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # [1, 2535, 4] -> [2535, 4]
output_classes = interpreter.get_tensor(output_details[1]['index'])[0]  # [1, 2535, 80] -> [2535, 80]

print(output_boxes)

# 원본 이미지 로드 및 변환
original_image = cv2.imread(image_path)
original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

original_height = original_image.shape[0]
original_width = original_image.shape[1]
# original_width
# print(original_width, original_height)
# YOLOv4 Tiny 모델 후처리
def postprocess_boxes(boxes, classes, confidence_threshold=0.1):
    # 박스와 클래스 정보에서 유의미한 정보만 추출
    valid_boxes = []
    for i in range(len(boxes)):
        if np.max(classes[i] > confidence_threshold):  # confidence score
            valid_boxes.append({
                'box': boxes[i][:4],  # x, y, width, height
                'probability': max(classes[i]),
                'class': np.argmax(classes[i]),  # 가장 높은 클래스 확률  
                'name': class_number_to_str_name[np.argmax(classes[i])]
            })
    return valid_boxes

# 후처리된 박스 출력
detected_boxes = postprocess_boxes(output_boxes, output_classes)

for box in detected_boxes:
    x_min = min(box['box'][0], box['box'][2])
    x_max = max(box['box'][0], box['box'][2])
    y_min = min(box['box'][1], box['box'][3])
    y_max = max(box['box'][1], box['box'][3])
    box['box'][0] = x_min
    box['box'][2] = x_max
    box['box'][1] = y_min
    box['box'][3] = y_max
    print(box)


# 결과 이미지 시각화
def draw_boxes(image, boxes, x_ratio, y_ratio, class_names):
    for box in boxes:
        # x_min, y_min, x_max, y_max = map(int, box['box'])
        x_min, y_min, width, height = box['box']
        
        x_min = int(x_min * x_ratio)
        y_min = int(y_min * y_ratio)
        width = int(width * x_ratio)
        height = int(height * y_ratio)
        
        class_id = box['class']
        # confidence = box['score']
        color = (0, 255, 0)  # 녹색
        # label = f"Class {class_id} ({confidence:.2f})"
        cv2.rectangle(image, (x_min, y_min), (x_min + width, y_min + height), color, 2)
        # cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return image


print(original_width, original_height, input_width, input_height)
# 이미지에 박스 그리기

image_with_boxes = draw_boxes(original_image_rgb, detected_boxes, (original_width / input_width), (original_height / input_height), class_names=None)
cv2.imshow('Detected Objects', cv2.cvtColor(image_with_boxes, cv2.COLOR_RGB2BGR) )
cv2.waitKey(0)
cv2.destroyAllWindows()



