import cv2
import matplotlib.pyplot as plt
import numpy as np

def detect_and_draw(img_gray, min_len, max_gap, color):
    """
    확률적 허프 변환을 수행하고 선을 그려서 반환하는 헬퍼 함수
    """
    # Canny Edge Detection (선 검출의 필수 전처리)
    edges = cv2.Canny(img_gray, 50, 150, apertureSize=3)
    
    # HoughLinesP(image, rho, theta, threshold, minLineLength, maxLineGap)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=min_len, maxLineGap=max_gap)
    
    # 결과 이미지 생성 (컬러)
    result_img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(result_img, (x1, y1), (x2, y2), color, 3) # 두께 3
            
    return result_img, edges

def main():
    image_path = 'pen.png'
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"이미지를 찾을 수 없습니다: {image_path}")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 사용자 요청: "옵션 쫌 다르게 해서"
    # 세 가지 다른 옵션으로 비교
    
    # 1. 짧은 선도 허용 (자잘한 직선 잡기)
    # minLineLength=10 (10픽셀만 넘으면 선으로 인정)
    res1, edges1 = detect_and_draw(gray, min_len=10, max_gap=5, color=(0, 0, 255)) # Red

    # 2. 긴 선만 허용 (메인 윤곽 잡기)
    # minLineLength=100 (100픽셀 이상이어야 선으로 인정)
    res2, edges2 = detect_and_draw(gray, min_len=100, max_gap=10, color=(0, 255, 0)) # Green

    # 3. 끊긴 선 이어주기 (Gap 허용치 증가)
    # maxLineGap=50 (50픽셀 떨어져 있어도 같은 선으로 간주하고 이음)
    res3, edges3 = detect_and_draw(gray, min_len=50, max_gap=50, color=(255, 0, 0)) # Blue

    # --- 시각화 ---
    plt.figure(figsize=(15, 10))
    plt.suptitle("Probabilistic Hough Transform (Different Options)", fontsize=16)

    # 원본 Canny (상단 중앙)
    plt.subplot(2, 3, 2)
    plt.imshow(edges1, cmap='gray')
    plt.title("Canny Edges (Input Feature)")
    plt.axis('off')

    # Option 1: Short Lines
    plt.subplot(2, 3, 4)
    plt.imshow(cv2.cvtColor(res1, cv2.COLOR_BGR2RGB))
    plt.title("Opt 1: Short Lines Allowed\n(min_len=10)")
    plt.axis('off')

    # Option 2: Long Lines Only
    plt.subplot(2, 3, 5)
    plt.imshow(cv2.cvtColor(res2, cv2.COLOR_BGR2RGB))
    plt.title("Opt 2: Long Lines Only\n(min_len=100)")
    plt.axis('off')

    # Option 3: Fill Gaps
    plt.subplot(2, 3, 6)
    plt.imshow(cv2.cvtColor(res3, cv2.COLOR_BGR2RGB))
    plt.title("Opt 3: Connect Wide Gaps\n(max_gap=50)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
