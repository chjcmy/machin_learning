import cv2
import matplotlib.pyplot as plt
import numpy as np

def main():
    image_path = 'skull__r1330156201.png'
    
    # 이미지 읽기 (BGR)
    image_bgr = cv2.imread(image_path)
    
    if image_bgr is None:
        print(f"이미지를 찾을 수 없습니다: {image_path}")
        return
        
    print(f"이미지 로드 성공! Shape: {image_bgr.shape}")

    # 1. 히스토그램 스트레칭 (Normalize)
    # 0~255 구간으로 정규화 (Min-Max Stretching 효과)
    # 컬러 이미지의 경우 각 채널별로 적용하거나 전체에 적용 가능
    # 여기서는 간단하게 전체 이미지 배열에 대해 적용
    dst = cv2.normalize(image_bgr, None, 0, 255, cv2.NORM_MINMAX)

    # 시각화를 위해 BGR -> RGB 변환
    src_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    dst_rgb = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

    # 2. 시각화 (이미지 & 히스토그램 비교)
    plt.figure(figsize=(12, 10))
    plt.suptitle("Histogram Stretching (Contrast Enhancement)", fontsize=16)

    # --- 원본 이미지 ---
    plt.subplot(2, 2, 1)
    plt.imshow(src_rgb)
    plt.title("Original Image")
    plt.axis('off')

    # --- 스트레칭 이미지 ---
    plt.subplot(2, 2, 2)
    plt.imshow(dst_rgb)
    plt.title("Stretched Image")
    plt.axis('off')

    # --- 원본 히스토그램 ---
    plt.subplot(2, 2, 3)
    colors = ('b', 'g', 'r')
    for i, color in enumerate(colors):
        hist = cv2.calcHist([image_bgr], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)
    plt.title("Original Histogram")
    plt.xlim([0, 256])

    # --- 스트레칭 히스토그램 ---
    plt.subplot(2, 2, 4)
    for i, color in enumerate(colors):
        hist = cv2.calcHist([dst], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)
    plt.title("Stretched Histogram")
    plt.xlim([0, 256])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
