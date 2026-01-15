import cv2
import time

def main():
    video_path = 'stock-footage-high-angle-shot-of-the-famous-shibuya-pedestrian-scramble-crosswalk-with-crowds-of-peo.avi'
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"동영상을 찾을 수 없습니다: {video_path}")
        return

    # HOG Descriptor 초기화
    hog = cv2.HOGDescriptor()
    # 사전에 훈련된 보행자 탐지용 SVM 분류기 설정
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    print("보행자 검출을 시작합니다. (종료하려면 'q'를 누르세요)")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("동영상 종료")
            break
            
        start_time = time.time()
        
        # HOG 알고리즘은 연산량이 많아서 큰 이미지에서는 매우 느립니다.
        # 속도 향상을 위해 프레임 크기를 줄입니다 (너비 600~800px 권장)
        frame = cv2.resize(frame, (800, 450)) # 16:9 비율 유지 가정
        
        # 보행자 탐지
        # detectMultiScale(image, winStride, padding, scale)
        # - winStride: 윈도우 이동 간격 (클수록 빠르지만 검출율 하락, (8,8) 권장)
        # - padding: 입력 이미지 가장자리 패딩
        # - scale: 이미지 피라미드 스케일 (이미지를 줄여가며 탐색하는 비율, 1.05 권장)
        boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8), padding=(8, 8), scale=1.05)
        
        # 검출된 사람 박스 그리기
        for (x, y, w, h) in boxes:
            # 초록색 박스
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
        # FPS 계산 및 표시
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.2f} | People: {len(boxes)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow('Pedestrian Detection (HOG)', frame)
        
        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
