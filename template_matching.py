import cv2
import matplotlib.pyplot as plt
import numpy as np

def main():
    image_path = 'apple.jpg'
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"이미지를 찾을 수 없습니다: {image_path}")
        return
        
    # 1. 템플릿 만들기 (원본 이미지에서 일부분 잘라내기)
    # 예를 들어 1/3 지점에서 100x100 크기만큼 잘라서 템플릿으로 씀
    h, w = image.shape[:2]
    crop_h, crop_w = int(h/4), int(w/4) # 잘라낼 크기
    start_y, start_x = int(h/3), int(w/3) # 시작 위치
    
    template = image[start_y:start_y+crop_h, start_x:start_x+crop_w].copy()
    
    print(f"Original Size: {image.shape}")
    print(f"Template Size: {template.shape}")
    
    # 2. 템플릿 매칭 수행
    # cv2.TM_CCOEFF_NORMED: 정규화된 상관계수 매칭 (1에 가까울수록 일치)
    res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    
    # 3. 최댓값 위치 찾기 (가장 잘 맞는 곳)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    print(f"Max Match Score: {max_val:.4f} (1.0 is perfect)")
    print(f"Best Match Location: {max_loc}")
    
    # 4. 결과 그리기
    # max_loc은 템플릿의 좌상단(Top-Left) 좌표입니다.
    top_left = max_loc
    bottom_right = (top_left[0] + crop_w, top_left[1] + crop_h)
    
    # 원본에 사각형 그리기 (빨간색, 두께 3)
    # 주의: cv2.rectangle은 이미지를 직접 수정하므로 복사본에 그림
    result_img = image.copy()
    cv2.rectangle(result_img, top_left, bottom_right, (0, 0, 255), 3)

    # --- 시각화 ---
    plt.figure(figsize=(12, 6))
    plt.suptitle("Template Matching (Find the Crop!)", fontsize=16)

    # 템플릿 (우리가 찾는 작은 조각)
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(template, cv2.COLOR_BGR2RGB))
    plt.title("Template (Target)")
    plt.axis('off')

    # 매칭 히트맵 (어디가 가장 비슷한지)
    plt.subplot(1, 3, 2)
    plt.imshow(res, cmap='jet')
    plt.title("Matching Heatmap\n(Red = High Match)")
    plt.axis('off')

    # 결과 (찾은 위치)
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Detected Result\nScore: {max_val:.2f}")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
