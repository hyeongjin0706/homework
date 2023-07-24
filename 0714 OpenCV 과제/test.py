import cv2
import time

def main():
    # 1. 두 동영상 파일 경로 설정
    video1_path = "wolf.mp4"
    video2_path = "wolf2.mp4"

    # 2. 동영상 읽어오기
    video1 = cv2.VideoCapture(video1_path)
    video2 = cv2.VideoCapture(video2_path)

    # 3. 동영상 크기 설정
    width = int(video1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 4. 동영상 저장용 VideoWriter 생성
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    output_path = "output_video.avi"
    out = cv2.VideoWriter(output_path, fourcc, 30, (width, height))

    # 5. 동영상 재생
    while True:
        ret1, frame1 = video1.read()
        ret2, frame2 = video2.read()

        if not ret1 or not ret2:
            break

        # 현재 시간 계산
        current_time = video1.get(cv2.CAP_PROP_POS_MSEC) / 1000

        # 2번 동영상이 시작되는 시간(1번 동영상이 끝나는 2초 전)
        start_time_video2 = video1.get(cv2.CAP_PROP_FRAME_COUNT) / video1.get(cv2.CAP_PROP_FPS) - 2

        if current_time < start_time_video2:
            # 1번 동영상 재생
            out.write(frame1)
        else:
            # 2번 동영상 위치 설정
            x_offset = int(width * (current_time - start_time_video2) / 2)
            y_offset = 0

            # 2번 동영상이 좌측에서 시작하도록 설정
            frame1[:, 0:x_offset] = frame2[:, 0:x_offset]
            out.write(frame1)
            if not frame1 or not ret2:
                break

        cv2.imshow("Combined Video", frame1)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    # 자원 해제
    video1.release()
    video2.release()
    out.release()

if __name__ == "__main__":
    main()