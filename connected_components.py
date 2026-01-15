import cv2
import matplotlib.pyplot as plt
import numpy as np

def main():
    # 레이블링은 객체가 뚜렷하게 분리된 이미지가 좋습니다. (해골 아이콘 사용)
    image_path = 'skull__r1330156201.png'
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"이미지를 찾을 수 없습니다: {image_path}")
        return

    # 1. 전처리: 이진화 (Binary mask 생성)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 배경이 흰색이고 객체가 어두운 경우라면 반전(cv2.THRESH_BINARY_INV)이 필요할 수 있습니다.
    # 일반적으로 레이블링은 '흰색'을 객체로 인식합니다.
    # Otsu 알고리즘으로 자동 이진화
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 2. 레이블링 (Connected Components)
    # retval: 객체 갯수 (N) - 배경 포함
    # labels: 레이블 맵 (이미지와 같은 크기, 각 픽셀이 어떤 객체에 속하는지 번호가 매겨짐)
    # stats : 각 객체의 통계 (x, y, width, height, area)
    # centroids : 각 객체의 중심점 (x, y)
    cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)

    print(f"총 발견된 객체 수(배경 포함): {cnt}")

    # 3. 결과 그리기
    # 컬러링을 위해 원본 복사
    output = image.copy()
    
    # 각 객체마다 랜덤 색상 생성 (배경인 0번은 제외하고 1번부터 시작)
    for i in range(1, cnt):
        x, y, w, h, area = stats[i]
        
        # 작은 노이즈(너무 작은 점)는 무시하려면 area 조건 추가 가능
        if area < 10:
            continue
            
        # 랜덤 색상 (BGR)
        color = np.random.randint(0, 255, size=3).tolist()
        
        # 바운딩 박스 그리기
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 1)
        
        # 번호 매기기
        cv2.putText(output, str(i), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # 레이블 맵 자체를 컬러풀하게 보기 위한 변환
    # 0~N 범위를 0~255로 정규화해서 컬러맵 적용
    label_map = (labels * 255 / (cnt - 1)).astype(np.uint8)
    label_map_color = cv2.applyColorMap(label_map, cv2.COLORMAP_JET)
    # 배경(0번)은 검은색으로 처리
    label_map_color[labels == 0] = [0, 0, 0]

    # --- 시각화 ---
    plt.figure(figsize=(12, 5))
    plt.suptitle(f"Connected Component Labeling (Count: {cnt-1})", fontsize=16)

    # Binary Mask
    plt.subplot(1, 3, 1)
    plt.imshow(thresh, cmap='gray')
    plt.title("Binary Threshold (Input)")
    plt.axis('off')

    # Label Map (Colorized)
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(label_map_color, cv2.COLOR_BGR2RGB))
    plt.title("Label Map (Colorized)")
    plt.axis('off')

    # Final Result (Bounding Boxes)
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    plt.title("Result with Bounding Boxes")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
