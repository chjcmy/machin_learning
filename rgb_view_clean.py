import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys

def main():
    image_path = 'skull__r1330156201.png'
    
    image_bgr = cv2.imread(image_path)
    
    if image_bgr is None:
        print(f"이미지를 찾을 수 없습니다: {image_path}")
        return

    print(f"이미지 로드 성공! Shape: {image_bgr.shape}")
    print(f"Height: {image_bgr.shape[0]}, Width: {image_bgr.shape[1]}, Channels: {image_bgr.shape[2]}")
    print("-" * 30)

    np.set_printoptions(threshold=sys.maxsize, linewidth=1000)

    blue_channel = image_bgr[:, :, 0]
    green_channel = image_bgr[:, :, 1]
    red_channel = image_bgr[:, :, 2]

    print("\n" + "="*50)
    print("      [FULL MATRIX DATA VIEW]")
    print("="*50)
    
    print("\n🔵 Blue Channel Matrix (16x16):\n")
    print(blue_channel)
    
    print("\n🟢 Green Channel Matrix (16x16):\n")
    print(green_channel)
    
    print("\n🔴 Red Channel Matrix (16x16):\n")
    print(red_channel)

    image_rgb_data = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    print("\n🌈 Full RGB Matrix (3D Array - [R, G, B] per pixel):\n")
    print(image_rgb_data)
    
    print("\n" + "="*50 + "\n")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(8, 8))
    plt.suptitle(f"Original Image Check: {image_path}", fontsize=16)

    plt.imshow(image_rgb)
    plt.title("Original Image (RGB)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
