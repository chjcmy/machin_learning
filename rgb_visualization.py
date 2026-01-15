import cv2
import matplotlib.pyplot as plt
import numpy as np

def main():
    # 이미지 파일 경로
    image_path = 'skull__r1330156201.png'
    
    # 이미지 읽기 (OpenCV는 기본적으로 BGR로 읽음)
    image_bgr = cv2.imread(image_path)
    
    if image_bgr is None:
        print(f"이미지를 찾을 수 없습니다: {image_path}")
        return

    print(f"이미지 로드 성공! Shape: {image_bgr.shape}")
    print(f"Height: {image_bgr.shape[0]}, Width: {image_bgr.shape[1]}, Channels: {image_bgr.shape[2]}")
    print("-" * 30)

    import sys
    np.set_printoptions(threshold=sys.maxsize, linewidth=1000) # 줄바꿈 없이 최대한 넓게 출력

    # 1. Numpy Slicing으로 채널 분리 (BGR 순서)
    blue_channel = image_bgr[:, :, 0]
    green_channel = image_bgr[:, :, 1]
    red_channel = image_bgr[:, :, 2]

    # 2. 전체 행렬 데이터 출력 (16x16이므로 전체 출력 가능)
    print("\n" + "="*50)
    print("      [FULL MATRIX DATA VIEW]")
    print("="*50)
    
    print("\n🔵 Blue Channel Matrix (16x16):\n")
    print(blue_channel)
    
    print("\n🟢 Green Channel Matrix (16x16):\n")
    print(green_channel)
    
    print("\n🔴 Red Channel Matrix (16x16):\n")
    print(red_channel)

    # 3D RGB Matrix 출력 (사용자 요청 반영: "행렬 rgb 다 나오게")
    # BGR -> RGB 변환 이미지를 출력하여 [R, G, B] 순서로 보이게 함
    image_rgb_data = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    print("\n🌈 Full RGB Matrix (3D Array - [R, G, B] per pixel):\n")
    print(image_rgb_data)
    
    print("\n" + "="*50 + "\n")

    # 3. 시각화를 위한 준비 (각 채널을 해당 색상으로 보여주기)
    zeros = np.zeros_like(blue_channel)
    
    # Merge를 사용해 단일 채널 이미지를 3채널 컬러 이미지로 변환 (시각화용)
    # Blue만 있는 이미지 (B, 0, 0) -> OpenCV는 BGR이므로 (Blue, zeros, zeros)
    blue_img = cv2.merge([blue_channel, zeros, zeros])
    
    # Green만 있는 이미지 (0, G, 0)
    green_img = cv2.merge([zeros, green_channel, zeros])
    
    # Red만 있는 이미지 (0, 0, R)
    red_img = cv2.merge([zeros, zeros, red_channel])
    
    # 원본 이미지를 RGB로 변환 (Matplotlib 표시용)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # Matplotlib으로 시각화할 때는 각 단색 이미지도 RGB로 바꿔줘야 색이 제대로 보임 (plt는 RGB 기준)
    # 하지만 위에서 만든 blue_img는 BGR 기준 (B, 0, 0)이므로, 
    # plt.imshow로 볼 때:
    # blue_img (B, 0, 0) -> RGB로 해석하면 (Red=B, Green=0, Blue=0) -> 붉게 나옴 (잘못됨)
    # 따라서 plt용으로 RGB 순서 (0, 0, B)로 다시 만들어야 함.
    
    # 시각화용 RGB 이미지 생성
    blue_viz = np.zeros_like(image_bgr)
    blue_viz[:, :, 2] = blue_channel # RGB의 Blue는 2번 인덱스

    green_viz = np.zeros_like(image_bgr)
    green_viz[:, :, 1] = green_channel # RGB의 Green은 1번 인덱스

    red_viz = np.zeros_like(image_bgr)
    red_viz[:, :, 0] = red_channel # RGB의 Red는 0번 인덱스

    # 4. Matplotlib Plot
    plt.figure(figsize=(8, 8))
    plt.suptitle(f"Original Image Check: {image_path}", fontsize=16)

    # 원본만 크게 출력
    plt.imshow(image_rgb)
    plt.title("Original Image (RGB)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
