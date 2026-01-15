import cv2
import sys
import matplotlib.pyplot as plt

def main():
    print("Hello OpenCV")
    print("OpenCV Version:", cv2.__version__)
    print("Python Version:", sys.version)

    # 간단한 이미지 창 띄우기
    image = cv2.imread('apple.jpg')
    
    if image is None:
        print("이미지를 읽을 수 없습니다. 경로를 확인해주세요.")
        return

    print("이미지를 성공적으로 불러왔습니다.")
    
    # 그레이스케일 변환
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 3. Canny Edge Detection
    canny_image = cv2.Canny(image, 100, 200)

    # 4. Otsu's Thresholding
    _, thresh_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # 이미지 리스트 및 타이틀 준비
    images = [image, gray_image, canny_image, thresh_image]
    titles = ['1. Original', '2. Gray', '3. Canny', '4. Otsu Threshold']

    # 전체 Figure 크기 설정 (각 subplot이 약 5x5인치가 되도록 10x10 설정)
    plt.figure(figsize=(10, 10))

    for i in range(4):
        plt.subplot(2, 2, i+1)
        
        target_img = images[i]
        
        # 이미지에 순서 텍스트 추가 (사본 생성하여 원본 데이터 보존)
        if len(target_img.shape) == 2: # 흑백/단일채널
            display_img = cv2.cvtColor(target_img, cv2.COLOR_GRAY2RGB)
            cv2.putText(display_img, str(i+1), (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 5)
        else: # 컬러 (BGR)
            display_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
            cv2.putText(display_img, str(i+1), (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 5)
            
        plt.imshow(display_img)
        plt.title(titles[i])
        plt.axis('off') # 테두리 제거

    plt.tight_layout()
    plt.show()

    # cv2.imshow('HelloCV', image)
    # cv2.imshow('HelloCV Gray', gray_image)
    
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
