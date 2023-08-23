import sys
import cv2
import os
import time

name = input('사람 이름을 입력하세요: ')
folder_path = os.path.join(os.getcwd(), 'images',name)  # 원하는 폴더 경로로 변경
cap = cv2.VideoCapture(0)

if not os.path.exists(folder_path):
    os.makedirs(folder_path)

if not cap.isOpened():
    print('카메라를 열 수 없습니다')
    sys.exit()

print('카메라 연결 성공')

while True:
    ret, frame = cap.read()
    if not ret:
        break
    temp = str(time.time()).split('.')
    cv2.imwrite(os.path.join(folder_path, temp[0]+temp[1]+'.jpg'), frame)
    cv2.imshow('frame', frame)

    if cv2.waitKey(10) == 27: #ESC키
        break

