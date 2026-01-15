import cv2
import matplotlib.pyplot as plt
import numpy as np

def main():
    image_path = 'pngtree-high-resolution-almond-tree-png-image_16501052.png'
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"이미지를 찾을 수 없습니다: {image_path}")
        return

    print(f"Original Shape: {image.shape}")
    
    # BGR -> RGB 변환 (시각화용)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 1. Averaging Filter (Blur) 커널 정의
    # 5x5 크기의 모든 값을 1로 채우고 25로 나눔 (평균내기)
    kernel_size = 5
    kernel_blur = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    
    # 필터 적용
    blurred = cv2.filter2D(image_rgb, -1, kernel_blur)

    # 2. Sharpening Filter 커널 정의
    # 중심 픽셀을 강조하고 주변 픽셀을 뺌
    kernel_sharpen = np.array([[-1, -1, -1], 
                               [-1,  9, -1], 
                               [-1, -1, -1]])
    
    # 필터 적용
    sharpened = cv2.filter2D(image_rgb, -1, kernel_sharpen)

    # --- 시각화 ---
    plt.figure(figsize=(15, 6))
    plt.suptitle(f"Basic 2D Filtering (Kernel Convolution)", fontsize=16)

    # 원본
    plt.subplot(1, 3, 1)
    plt.imshow(image_rgb)
    plt.title("Original")
    plt.axis('off')

    # 블러 (흐릿하게)
    plt.subplot(1, 3, 2)
    plt.imshow(blurred)
    plt.title(f"Blurred (Average {kernel_size}x{kernel_size})")
    plt.axis('off')

    # 선명하게 (샤픈)
    plt.subplot(1, 3, 3)
    plt.imshow(sharpened)
    plt.title("Sharpened (Edge Enhacing)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
