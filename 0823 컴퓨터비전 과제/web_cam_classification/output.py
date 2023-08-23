import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.nn import functional as F
from torch.utils.data import DataLoader
from PIL import Image
import os
import sys
import numpy as np
import cv2
import time

folder_path = os.path.join(os.getcwd(), 'output')

if not os.path.exists(folder_path):
    os.makedirs(folder_path)

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
}

image_datasets = {
    'train': datasets.ImageFolder(os.path.join(os.getcwd(), 'images')),
}

dataloaders = {
    'train': DataLoader(
        image_datasets['train'],
        batch_size=32,
        shuffle=True
    )
}

model = models.efficientnet_b4(weights='IMAGENET1K_V1')

for param in model.parameters():
  param.requires_gred = False


model.classifier = nn.Sequential(
    nn.Linear(1792, 512),
    nn.ReLU(),
    nn.Linear(512, len(image_datasets['train'].classes)),
)
model.load_state_dict(torch.load('cam.h5'))
model.eval()

cap = cv2.VideoCapture(0)


if not cap.isOpened():
    print('카메라를 열 수 없습니다')
    sys.exit()

print('카메라 연결 성공')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet 정규화
])
classes = image_datasets['train'].classes

while True:
    ret, frame = cap.read()
    if not ret:
        break
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    transformed_image = transform(image)
    transformed_image = transformed_image.unsqueeze(0)
    pred = model(transformed_image)
    predicted_classes = torch.argmax(pred, dim=1)
    # 변환된 이미지를 다시 OpenCV 형식으로 변환하여 화면에 표시
    #transformed_image_np = transformed_image.numpy().transpose(1, 2, 0)
    #transformed_image_np = (transformed_image_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
    #transformed_image_np = transformed_image_np.clip(0, 255).astype(np.uint8)
    #cv2.imshow('Transformed WebCam', transformed_image_np)
    result = classes[predicted_classes.item()]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (0, 0, 255)  # 빨간색
    thickness = 2
    text_size = cv2.getTextSize(result, font, font_scale, thickness)[0]
    x = (frame.shape[1] - text_size[0]) // 2  # 중앙 정렬
    y = text_size[1] + 10  # 상단에 텍스트가 나타나도록 y 좌표 설정
    cv2.putText(frame, result, (x, y), font, font_scale, color, thickness)

    cv2.imshow('frame', frame)
    temp = str(time.time()).split('.')
    cv2.imwrite(os.path.join(folder_path, temp[0] + temp[1] + '.jpg'), frame)

    if cv2.waitKey(10) == 27: #ESC키
        break