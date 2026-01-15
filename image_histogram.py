import cv2
import matplotlib.pyplot as plt

def main():
    # 이미지 파일 경로
    image_path = 'skull__r1330156201.png'
    
    # 이미지 읽기 (OpenCV는 BGR)
    image_bgr = cv2.imread(image_path)
    
    if image_bgr is None:
        print(f"이미지를 찾을 수 없습니다: {image_path}")
        return

    print(f"이미지 로드 성공! Shape: {image_bgr.shape}")

    # 히스토그램 계산 및 시각화 준비
    colors = ('b', 'g', 'r')
    labels = ('Blue', 'Green', 'Red')
    
    plt.figure(figsize=(12, 6))
    plt.suptitle(f"Color Histogram: {image_path}", fontsize=16)

    # 1. 왼쪽: 원본 이미지 표시
    plt.subplot(1, 2, 1)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.title("Original Image")
    plt.axis('off')

    # 2. 오른쪽: 히스토그램 그래프
    plt.subplot(1, 2, 2)
    
    for i, color in enumerate(colors):
        # cv2.calcHist(images, channels, mask, histSize, ranges)
        # i=0(B), 1(G), 2(R)
        hist = cv2.calcHist([image_bgr], [i], None, [256], [0, 256])
        plt.plot(hist, color=color, label=labels[i])
        plt.xlim([0, 256])
    
    plt.title("RGB Intensity Distribution")
    plt.xlabel("Pixel Value (0-255)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
