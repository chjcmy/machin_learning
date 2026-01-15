import cv2
import matplotlib.pyplot as plt
import numpy as np
import random

def add_salt_and_pepper_noise(image, prob=0.05):
    """
    이미지에 소금(흰색)과 후추(검은색) 노이즈를 임의로 추가하는 함수
    prob: 노이즈가 생길 확률 (0.05 = 5%)
    """
    output = np.copy(image)
    
    # 1. Salt (White) noise
    # 전체 픽셀 수 * prob / 2 만큼 흰 점을 찍음
    num_salt = np.ceil(prob * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    output[tuple(coords)] = 255  # 흰색 점

    # 2. Pepper (Black) noise
    # 전체 픽셀 수 * prob / 2 만큼 검은 점을 찍음
    num_pepper = np.ceil(prob * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    output[tuple(coords)] = 0    # 검은색 점
    
    return output

def main():
    image_path = 'pngtree-high-resolution-almond-tree-png-image_16501052.png'
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"이미지를 찾을 수 없습니다: {image_path}")
        return

    print(f"Original Shape: {image.shape}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 1. 노이즈 추가 (이미지 훼손)
    print("Adding Salt & Pepper Noise...")
    noisy_image = add_salt_and_pepper_noise(image_rgb, prob=0.02) # 2% 확률로 노이즈

    # 2. Median Filter 적용 (노이즈 제거)
    # ksize는 반드시 홀수여야 함 (3, 5, 7 ...)
    median_ksize = 5
    median_blurred = cv2.medianBlur(noisy_image, median_ksize)

    # (비교용) Mean Filter (Average Blur) 도 같이 적용해봄
    # 같은 크기(5x5)의 커널로 블러링 했을 때 노이즈가 어떻게 되는지 비교
    mean_blurred = cv2.blur(noisy_image, (median_ksize, median_ksize))

    # --- 시각화 ---
    plt.figure(figsize=(16, 8))
    plt.suptitle(f"Median Blur vs Mean Blur (Noise Removal Comparison)", fontsize=16)

    # 원본 (노이즈 추가됨)
    plt.subplot(1, 3, 1)
    plt.imshow(noisy_image)
    plt.title("Noisy Image (Salt & Pepper)")
    plt.axis('off')

    # Mean Filter 결과
    plt.subplot(1, 3, 2)
    plt.imshow(mean_blurred)
    plt.title(f"Mean Filter (Blur {median_ksize}x{median_ksize})\nNoise smudged, not removed")
    plt.axis('off')

    # Median Filter 결과
    plt.subplot(1, 3, 3)
    plt.imshow(median_blurred)
    plt.title(f"Median Filter (ksize={median_ksize})\nCleanly removed!")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
