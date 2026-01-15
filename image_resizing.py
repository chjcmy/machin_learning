import cv2
import matplotlib.pyplot as plt

def main():
    image_path = 'pngtree-high-resolution-almond-tree-png-image_16501052.png'
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"이미지를 찾을 수 없습니다: {image_path}")
        return

    print(f"Original Shape: {image.shape}")

    # 확대 비율 (16x16 이미지를 10배 키워서 160x160으로 비교)
    scale = 10
    dim = (image.shape[1] * scale, image.shape[0] * scale)

    # 5가지 보간법 리스트
    interpolations = [
        ("Nearest", cv2.INTER_NEAREST),
        ("Linear", cv2.INTER_LINEAR),
        ("Cubic", cv2.INTER_CUBIC),
        ("Area", cv2.INTER_AREA),
        ("Lanczos4", cv2.INTER_LANCZOS4)
    ]

    plt.figure(figsize=(15, 8))
    plt.suptitle(f"Interpolation Methods Comparison (Scale x{scale})", fontsize=16)

    # 1. 원본 (가장 첫 번째 칸)
    plt.subplot(2, 3, 1)
    # 원본도 작지만 그냥 보여줌 (matplotlib이 알아서 늘려 보여줄 수 있으나, 픽셀 그대로 표현)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f"Original ({image.shape[1]}x{image.shape[0]})")
    plt.axis('off')

    # 2. 각 보간법 적용 결과
    for i, (name, method) in enumerate(interpolations):
        # 이미지 리사이즈
        resized = cv2.resize(image, dim, interpolation=method)
        
        # 서브플롯 위치 (2번째 칸부터 시작)
        plt.subplot(2, 3, i + 2)
        plt.imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        plt.title(f"{name}\n({dim[0]}x{dim[1]})")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
