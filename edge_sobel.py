import cv2
import matplotlib.pyplot as plt
import numpy as np

def main():
    # 엣지 검출은 모양이 중요하므로 pen.png 사용 권장
    image_path = 'pen.png'
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"이미지를 찾을 수 없습니다: {image_path}")
        return

    print(f"Original Shape: {image.shape}")
    
    # 전처리: 엣지 검출은 주로 그레이스케일에서 수행
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. Sobel X (dx=1, dy=0) : 세로선(수직 엣지) 검출
    # 값이 음수가 될 수 있으므로 cv2.CV_64F 사용 후 절대값 취함
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_x = cv2.convertScaleAbs(sobel_x)

    # 2. Sobel Y (dx=0, dy=1) : 가로선(수평 엣지) 검출
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_y = cv2.convertScaleAbs(sobel_y)

    # 3. Combined (Gradient Magnitude) : 전체 윤곽선
    # cv2.addWeighted로 x, y 성분 합치기
    sobel_combined = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

    # --- 시각화 ---
    plt.figure(figsize=(12, 10))
    plt.suptitle("Sobel Edge Detection (Differential Gradient)", fontsize=16)

    # 원본 (Gray)
    plt.subplot(2, 2, 1)
    plt.imshow(gray, cmap='gray')
    plt.title("Original (Grayscale)")
    plt.axis('off')

    # Sobel X
    plt.subplot(2, 2, 2)
    plt.imshow(sobel_x, cmap='gray')
    plt.title("Sobel X (Vertical Edges)")
    plt.axis('off')

    # Sobel Y
    plt.subplot(2, 2, 3)
    plt.imshow(sobel_y, cmap='gray')
    plt.title("Sobel Y (Horizontal Edges)")
    plt.axis('off')

    # Combined
    plt.subplot(2, 2, 4)
    plt.imshow(sobel_combined, cmap='gray')
    plt.title("Combined Edge Magnitude")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
