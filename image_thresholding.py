import cv2
import matplotlib.pyplot as plt

def main():
    image_path = 'apple.jpg'
    # 이진화(Thresholding)는 기본적으로 그레이스케일 이미지에서 수행합니다.
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print(f"이미지를 찾을 수 없습니다: {image_path}")
        return

    print(f"Original Shape: {image.shape}")

    # 1. Global Thresholding (전역 임계처리)
    # 127보다 크면 255(흰색), 아니면 0(검은색)
    # 가장 단순하지만, 조명이 불균일하면 잘 안 됨.
    ret1, thresh_global = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    print(f"Global Threshold Value used: {ret1}")

    # 2. Otsu's Binarization (오츠 알고리즘)
    # 히스토그램을 분석해서 "최적의 임계값"을 컴퓨터가 자동으로 찾음.
    # 0을 넣고 cv2.THRESH_OTSU 옵션을 주면 됨.
    ret2, thresh_otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f"Otsu's Threshold Value calculated: {ret2}")

    # 3. Adaptive Thresholding (적응형 임계처리)
    # 이미지를 작은 구역으로 나누어 각 구역마다 다른 임계값을 적용.
    # 조명이 불균일하거나 그림자가 진 곳에서도 잘 됨.
    # thresh_adaptive = cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C)
    thresh_adaptive = cv2.adaptiveThreshold(image, 255, 
                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY, 
                                            11, 2)

    # --- 시각화 ---
    plt.figure(figsize=(16, 5))
    plt.suptitle("Image Thresholding (Binarization) Techniques", fontsize=16)

    # 원본
    plt.subplot(1, 4, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Original (Grayscale)")
    plt.axis('off')

    # Global
    plt.subplot(1, 4, 2)
    plt.imshow(thresh_global, cmap='gray')
    plt.title(f"Global Threshold (v=127)")
    plt.axis('off')

    # Otsu
    plt.subplot(1, 4, 3)
    plt.imshow(thresh_otsu, cmap='gray')
    plt.title(f"Otsu's Threshold (v={ret2:.0f})")
    plt.axis('off')

    # Adaptive
    plt.subplot(1, 4, 4)
    plt.imshow(thresh_adaptive, cmap='gray')
    plt.title("Adaptive Threshold\n(Shadow/Detail handled)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
