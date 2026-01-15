import cv2
import matplotlib.pyplot as plt

def main():
    # 1. 이미지 준비
    # 공통 크기 설정 (연산을 위해 크기가 같아야 함)
    target_size = (300, 300)

    # 이미지 로드 및 리사이즈
    apple_img = cv2.imread('apple.jpg')
    pineapple_img = cv2.imread('pineapple.jpg')
    pen_img = cv2.imread('pen.png')

    if apple_img is None or pineapple_img is None or pen_img is None:
        print("이미지 파일을 찾을 수 없습니다. (apple.jpg, pineapple.jpg, pen.png 확인 필요)")
        return

    apple_img = cv2.resize(apple_img, target_size)
    pineapple_img = cv2.resize(pineapple_img, target_size)
    pen_img = cv2.resize(pen_img, target_size)

    # 2. 산술 연산 (Image Arithmetic - Blending)
    
    # Apple + Pen = ApplePen (50% : 50% 합성)
    apple_pen = cv2.addWeighted(apple_img, 0.5, pen_img, 0.5, 0)
    
    # Pineapple + Pen = PineapplePen (50% : 50% 합성)
    pineapple_pen = cv2.addWeighted(pineapple_img, 0.5, pen_img, 0.5, 0)
    
    # 3. 시각화
    # Matplotlib 출력을 위해 BGR -> RGB 변환
    images = [pen_img, pineapple_img, apple_img, pen_img, pineapple_pen, apple_pen]
    titles = ['Pen', 'Pineapple', 'Apple', 'Pen', 'Pineapple Pen', 'Apple Pen']
    
    plt.figure(figsize=(12, 8))
    plt.suptitle("PPAP Arithmetic Operations", fontsize=20)

    for i in range(6):
        plt.subplot(2, 3, i+1)
        img_rgb = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
