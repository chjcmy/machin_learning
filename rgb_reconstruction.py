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

    # 1. 채널 분리
    blue, green, red = cv2.split(image_bgr)

    # 2. 채널 합치기 (Merge) -> 원본 복구
    # [blue(0), green(1), red(2)] 순서로 다시 합침
    merged_bgr = cv2.merge([blue, green, red])
    
    # 3. 값 더하기 (Arithmetic Addition) -> 밝기 합산
    # 단순히 B + G + R 을 하면 255를 넘을 수 있으므로 cv2.add 사용 (saturation 연산)
    summed_image = cv2.add(blue, cv2.add(green, red))

    # 행렬 데이터 출력 (사용자 요청)
    import sys
    np.set_printoptions(threshold=sys.maxsize, linewidth=1000)

    print("\n" + "="*50)
    print("      [RECONSTRUCTED MATRIX DATA]")
    print("="*50)
    
    print("\n💀 Merged Image Matrix (Reconstructed Original) - [R, G, B]:\n")
    # 출력을 위해 RGB로 변환하여 출력 (사용자가 직관적으로 색상을 알 수 있게)
    print(cv2.cvtColor(merged_bgr, cv2.COLOR_BGR2RGB))
    
    print("\n✨ Summed Image Matrix (Intensity Sum):\n")
    print(summed_image)
    print("\n" + "="*50 + "\n")

    # 시각화를 위해 BGR -> RGB 변환
    original_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    merged_rgb = cv2.cvtColor(merged_bgr, cv2.COLOR_BGR2RGB)
    
    # Summed image는 1채널(Grayscale)이므로 colormap 적용 가능하지만, 
    # 여기서는 그레이스케일로 보여줌
    
    plt.figure(figsize=(15, 5))
    plt.suptitle("RGB Reconstruction vs Summation", fontsize=16)

    # 1. 원본
    plt.subplot(1, 3, 1)
    plt.imshow(original_rgb)
    plt.title("Original Image")
    plt.axis('off')

    # 2. Merged (복구된 이미지)
    plt.subplot(1, 3, 2)
    plt.imshow(merged_rgb)
    plt.title("Merged (B, G, R Combined)")
    plt.axis('off')

    # 3. Summed (값의 합)
    plt.subplot(1, 3, 3)
    plt.imshow(summed_image, cmap='gray')
    plt.title("Summed (B + G + R)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
