import cv2
import matplotlib.pyplot as plt
import numpy as np

def main():
    image_path = 'apple.jpg'
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"이미지를 찾을 수 없습니다: {image_path}")
        return

    # 원본 복사 (결과 그리기용)
    output = image.copy()
    
    # 1. 전처리: 그레이스케일 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. 전처리: 노이즈 제거 (필수)
    # 허프 변환은 노이즈에 민감하므로 Median Blur 등으로 부드럽게 만들어야 함
    gray_blurred = cv2.medianBlur(gray, 5)

    # 3. Hough Circle 변환
    # cv2.HoughCircles(image, method, dp, minDist, param1, param2, minRadius, maxRadius)
    # - dp: 해상도 비율 (1=원본과 동일, 2=절반)
    # - minDist: 검출된 원들 사이의 최소 거리 (너무 가까우면 같은 원으로 취급하지 않음)
    # - param1: Canny 엣지 검출기의 높은 임계값
    # - param2: 원 검출 임계값 (작을수록 원을 더 많이 찾지만 오검출 확률 높음)
    circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, 1, 
                               minDist=gray.shape[0]/8,
                               param1=100, param2=30,
                               minRadius=10, maxRadius=0) # 0이면 최대 반지름 제한 없음

    # 4. 결과 그리기
    if circles is not None:
        # 좌표를 정수로 변환
        circles = np.uint16(np.around(circles))
        
        print(f"검출된 원의 개수: {len(circles[0, :])}")

        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            
            # 원 둘레 그리기 (초록색, 두께 2)
            cv2.circle(output, center, radius, (0, 255, 0), 2)
            
            # 원 중심점 그리기 (빨간색, 두께 3)
            cv2.circle(output, center, 2, (0, 0, 255), 3)
    else:
        print("원을 찾지 못했습니다.")

    # --- 시각화 ---
    plt.figure(figsize=(10, 5))
    plt.suptitle("Hough Circle Detection", fontsize=16)

    # 원본 (Gray + Blur)
    plt.subplot(1, 2, 1)
    plt.imshow(gray_blurred, cmap='gray')
    plt.title("Preprocessed (Gray + Blur)")
    plt.axis('off')

    # 결과 (Circles Detected)
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    plt.title("Detected Circles")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
