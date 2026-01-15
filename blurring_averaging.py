import cv2
import matplotlib.pyplot as plt

def main():
    image_path = 'pngtree-high-resolution-almond-tree-png-image_16501052.png'
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"이미지를 찾을 수 없습니다: {image_path}")
        return

    print(f"Original Shape: {image.shape}")
    
    # BGR -> RGB 변환 (시각화용)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 다양한 커널 크기로 블러링 적용
    # 커널 크기가 클수록 더 많이 뭉개짐 (더 흐릿해짐)
    k_sizes = [(5, 5), (10, 10), (20, 20)]
    
    blurred_images = []
    for k in k_sizes:
        # cv2.blur(src, ksize) -> 평균값 필터 (Normalized Box Filter)
        blurred = cv2.blur(image_rgb, k)
        blurred_images.append(blurred)

    # --- 시각화 ---
    plt.figure(figsize=(16, 5))
    plt.suptitle("Average Blurring (Mean Filter) with Varying Kernel Sizes", fontsize=16)

    # 1. 원본
    plt.subplot(1, 4, 1)
    plt.imshow(image_rgb)
    plt.title("Original")
    plt.axis('off')

    # 2. 5x5 Blur
    plt.subplot(1, 4, 2)
    plt.imshow(blurred_images[0])
    plt.title(f"Blur (5x5)")
    plt.axis('off')

    # 3. 10x10 Blur
    plt.subplot(1, 4, 3)
    plt.imshow(blurred_images[1])
    plt.title(f"Blur (10x10)")
    plt.axis('off')

    # 4. 20x20 Blur
    plt.subplot(1, 4, 4)
    plt.imshow(blurred_images[2])
    plt.title(f"Blur (20x20)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
