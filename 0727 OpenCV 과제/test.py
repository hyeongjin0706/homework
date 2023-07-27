import cv2
import numpy as np

img = cv2.imread('namecard.jpg')

img_shape = img.shape
w = img.shape[1]
h = img.shape[0]
img_copy = img.copy()
cv2.rectangle(img_copy, (15, 15), (w - 15, h - 15), (255, 0, 0), 2)

cv2.circle(img_copy, (15, 15), 10, (0, 255, 0), 2)
cv2.circle(img_copy, (w - 15, 15), 10, (0, 255, 0), 2)
cv2.circle(img_copy, (15, h-15), 10, (0, 255, 0), 2)
cv2.circle(img_copy, (w - 15, h - 15), 10, (0, 255, 0), 2)

v_1 = (15, 15)
v_2 = (w - 15, 15)
v_4 = (15, h-15)
v_3 = (w - 15, h - 15)
circle_list = [v_1, v_2, v_3, v_4]
w = h = 0
start_x = start_y = 0
check_circle = -1
def on_mouse(event, x, y, flags, param):
    global v_1, v_2, v_3, v_4, check_circle, img_copy, circle_list, img
    if event == cv2.EVENT_LBUTTONDOWN:
        for i, v in enumerate(circle_list):
            if v[0] - 7 <= x <= v[0] + 7 and v[1] - 7 <= y <= v[1] + 7:
                check_circle = i
                break
        cv2.imshow('img', img_copy)

    elif event == cv2.EVENT_MOUSEMOVE:
        img_copy = img.copy()
        if check_circle >= 0:
            circle_list[check_circle] = (x, y)
            for i in range(len(circle_list)):
                cv2.circle(img_copy, circle_list[i], 10, (0, 255, 0), 2)
                if i == 3:
                    cv2.line(img_copy, circle_list[i], circle_list[0], (255, 0, 0), 2)
                    continue
                cv2.line(img_copy, circle_list[i], circle_list[i+1], (255, 0, 0), 2)
            cv2.imshow('img', img_copy)
    elif event == cv2.EVENT_LBUTTONUP:
        if check_circle >= 0:
            check_circle = -1
            cv2.imshow('img', img_copy)
        else:
            cv2.imshow('img', img_copy)
            print('영역이 잘못됨')
# img = cv2.imread('sun.jpg')
# 창 이름 세팅
cv2.namedWindow('img')
cv2.setMouseCallback('img', on_mouse, img_copy)

space = ord(' ')
cv2.imshow('img', img_copy)
while True: # 무한 루프
    keycode = cv2.waitKey() # 키보드 입력 반환 값 저장
    if keycode == space:
        print('들어옴')
        srcQuad = np.array(circle_list, np.float32)
        dstQuad = np.array([[0, 0], [800, 0], [800, 400], [0, 400]], np.float32)
        pers = cv2.getPerspectiveTransform(srcQuad, dstQuad)
        dst = cv2.warpPerspective(img, pers, (800, 400))
        cv2.imshow('dst', dst)
    else:
        break