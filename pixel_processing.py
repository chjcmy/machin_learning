import cv2
import matplotlib.pyplot as plt

def main():
    # 이미지 읽기
    image = cv2.imread('apple.jpg')
    
    if image is None:
        print("이미지를 읽을 수 없습니다. 'apple.jpg' 경로를 확인해주세요.")
        return

    print("이미지 불러오기 성공")

    # 1. 화소 값 접근 (예: 100, 100 위치)
    px = image[100, 100]
    print(f"(100, 100) 위치의 화소 값 (BGR): {px}")

    # 2. 화소 값 변경 (특정 영역을 흰색으로 변경)
    # 이미지 복사 (원본 보존을 위해)
    modified_image = image.copy()
    # 세로 100~200, 가로 100~200 영역을 흰색(255, 255, 255)으로 채움
    modified_image[100:200, 100:200] = [255, 255, 255]
    print("특정 영역을 흰색으로 변경했습니다.")

    # 3. 이미지 반전 (Negative Image)
    # 모든 화소 값을 반전 (255 - 값)
    inverted_image = 255 - image
    print("이미지 색상을 반전시켰습니다.")

    # 시각화 (Matplotlib 사용)
    # BGR -> RGB 변환
    modified_rgb = cv2.cvtColor(modified_image, cv2.COLOR_BGR2RGB)
    inverted_rgb = cv2.cvtColor(inverted_image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 5))

    # 수정된 이미지 출력
    plt.subplot(1, 2, 1)
    plt.imshow(modified_rgb)
    plt.title('Modified Image (Region White)')
    plt.axis('off')

    # 반전된 이미지 출력
    plt.subplot(1, 2, 2)
    plt.imshow(inverted_rgb)
    plt.title('Inverted Image')
    plt.axis('off')

    plt.show()

if __name__ == "__main__":
    main()
