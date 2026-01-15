import cv2
import matplotlib.pyplot as plt
import numpy as np

def main():
    image_path = 'skull__r1330156201.png'
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"이미지를 찾을 수 없습니다: {image_path}")
        return

    # 이미지 크기
    rows, cols = image.shape[:2]

    # --- 변수 설정 (Variables) ---
    # 사용자가 값을 주듯이 여기서 변수를 변경하면 결과가 바뀜
    tx = 3.0   # x축 이동 (오른쪽으로 3픽셀)
    ty = 2.0   # y축 이동 (아래로 2픽셀)
    
    angle = 45.0 # 회전 각도 (45도 반시계 방향)
    scale = 1.0  # 크기 배율 (1.0 = 그대로)

    print(f"변수 설정: tx={tx}, ty={ty}, angle={angle}, scale={scale}")

    # 1. 이동 변환 (Translation)
    # 이동 행렬 M = [[1, 0, tx], [0, 1, ty]]
    M_trans = np.float32([[1, 0, tx], [0, 1, ty]])
    
    print("\n[이동 변환 행렬 M_trans]")
    print(M_trans)
    
    dst_trans = cv2.warpAffine(image, M_trans, (cols, rows))

    # 2. 회전 변환 (Rotation)
    # 중심점 기준으로 회전 (여기서는 이미지 중심)
    center = (cols / 2, rows / 2)
    M_rot = cv2.getRotationMatrix2D(center, angle, scale)
    
    print("\n[회전 변환 행렬 M_rot]")
    print(M_rot)

    dst_rot = cv2.warpAffine(image, M_rot, (cols, rows))

    # 3. 이동 + 회전 동시 적용 (복합)
    # 새로운 M을 만들거나, 여기서는 회전된 이미지를 다시 이동시켜봄 (순서 중요)
    dst_combined = cv2.warpAffine(dst_rot, M_trans, (cols, rows))


    # --- 시각화 ---
    plt.figure(figsize=(12, 4))
    plt.suptitle("Affine Transformations with Variables", fontsize=14)

    # 원본
    plt.subplot(1, 4, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original")
    plt.axis('off')

    # 이동
    plt.subplot(1, 4, 2)
    plt.imshow(cv2.cvtColor(dst_trans, cv2.COLOR_BGR2RGB))
    plt.title(f"Translated\n(tx={tx}, ty={ty})")
    plt.axis('off')

    # 회전
    plt.subplot(1, 4, 3)
    plt.imshow(cv2.cvtColor(dst_rot, cv2.COLOR_BGR2RGB))
    plt.title(f"Rotated\n(ang={angle}, sc={scale})")
    plt.axis('off')

    # 복합
    plt.subplot(1, 4, 4)
    plt.imshow(cv2.cvtColor(dst_combined, cv2.COLOR_BGR2RGB))
    plt.title("Rotated + Translated")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
