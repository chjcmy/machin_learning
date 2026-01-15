import cv2
import matplotlib.pyplot as plt
import numpy as np

def split_image_into_grid(image, rows, cols):
    """
    이미지를 rows x cols 개수로 분할하여 리스트로 반환합니다.
    순서는 좌->우, 위->아래 입니다.
    """
    height, width = image.shape[:2]
    
    # 각 조각의 높이와 너비 계산 (반올림 오차 방지 위해 int 변환)
    # 마지막 조각이 조금 작거나 클 수 있음
    step_h = height // rows
    step_w = width // cols
    
    tiles = []
    
    for r in range(rows):
        for c in range(cols):
            # 슬라이싱 범위 계산
            y_start = r * step_h
            y_end = (r + 1) * step_h if r < rows - 1 else height # 마지막 줄은 끝까지
            
            x_start = c * step_w
            x_end = (c + 1) * step_w if c < cols - 1 else width # 마지막 칸은 끝까지
            
            # Numpy Slicing으로 이미지 잘라내기
            tile = image[y_start:y_end, x_start:x_end]
            tiles.append(tile)
            
    return tiles

def main():
    image_path = 'apple.jpg'
    image = cv2.imread(image_path)
    
    if image is None:
        print("이미지를 찾을 수 없습니다. 경로를 확인해주세요.")
        return

    # 분할 설정 (2행 2열)
    ROWS = 2
    COLS = 2
    
    print(f"이미지를 {ROWS}x{COLS} 그리드로 분할합니다.")
    
    tiles = split_image_into_grid(image, ROWS, COLS)
    
    # 시각화
    plt.figure(figsize=(10, 10))
    plt.suptitle(f"Image Split: {ROWS}x{COLS}", fontsize=20)
    
    for i, tile in enumerate(tiles):
        # subplot 인덱스는 1부터 시작
        plt.subplot(ROWS, COLS, i + 1)
        
        # BGR -> RGB 변환
        tile_rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
        
        plt.imshow(tile_rgb)
        plt.title(f"Part {i+1}")
        plt.axis('off')
        
        # 각 조각의 크기(shape) 출력
        print(f"Part {i+1} shape: {tile.shape}")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
