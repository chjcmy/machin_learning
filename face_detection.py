import cv2
import matplotlib.pyplot as plt

def main():
    image_path = 'kim.jpg'
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"이미지를 찾을 수 없습니다: {image_path}")
        return

    # Haar Cascade 분류기 불러오기
    # OpenCV 설치 시 기본으로 제공되는 haarcascade_frontalface_default.xml 파일을 사용합니다.
    # cv2.data.haarcascades는 xml 파일들이 있는 폴더 경로를 반환합니다.
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    if face_cascade.empty():
        print("Cascade 파일을 불러오는데 실패했습니다.")
        return

    # 얼굴 검출은 그레이스케일 이미지에서 수행해야 속도가 빠르고 정확함
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 얼굴 탐지 수행
    # detectMultiScale(image, scaleFactor, minNeighbors)
    # - scaleFactor: 이미지를 얼마나 줄여가며 찾을지 (1.1 = 10%씩 줄임)
    # - minNeighbors: 후보 영역이 몇 번 중복 검출되어야 실제 얼굴로 인정할지 (높을수록 엄격함)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    # 눈 탐지기 불러오기 (안경 쓴 눈 모델이 일반 성능도 더 좋은 경우가 많음)
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    if len(faces) > 0:
        print(f"검출된 얼굴 수: {len(faces)}")
        
        for (x, y, w, h) in faces:
            # 얼굴 영역 박스 그리기 (초록색)
            cv2.rectangle(img_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img_rgb, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # --- 눈 검출 (얼굴 영역 안에서만 찾기) ---
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img_rgb[y:y+h, x:x+w]
            
            # 디버깅: ROI 크기 출력
            print(f"  -> 얼굴 영역(ROI) 크기: {roi_gray.shape}")

            # 1. 원본 ROI에서 바로 찾으면 너무 작아서(52x52) 실패함 -> 3배 확대 (Super Resolution Simulation)
            scale_ratio = 3.0
            width = int(roi_gray.shape[1] * scale_ratio)
            height = int(roi_gray.shape[0] * scale_ratio)
            
            # 큐빅 보간법으로 화질 보전하며 확대
            roi_upscaled = cv2.resize(roi_gray, (width, height), interpolation=cv2.INTER_CUBIC)
            
            # 2. 확대된 이미지에서 명암비 향상 (Histogram Equalization)
            roi_upscaled = cv2.equalizeHist(roi_upscaled)
            
            print(f"  -> 확대된 검색 영역 크기: {roi_upscaled.shape}")

            # 3. 눈 탐지 (확대된 이미지에서 수행)
            # 조금 더 엄격한 기준을 적용해도 화질이 좋아져서 잘 잡힘
            eyes = eye_cascade.detectMultiScale(roi_upscaled, 
                                              scaleFactor=1.1, 
                                              minNeighbors=3,
                                              minSize=(20, 20)) # 확대했으므로 minSize도 키움
            
            print(f"  -> 확대 후 검출된 눈 개수: {len(eyes)}")

            for (ex, ey, ew, eh) in eyes:
                # 4. 좌표 복원 (확대된 좌표를 다시 원래 크기로 줄임)
                real_ex = int(ex / scale_ratio)
                real_ey = int(ey / scale_ratio)
                real_ew = int(ew / scale_ratio)
                real_eh = int(eh / scale_ratio)
                
                # 눈 영역 박스 그리기 (파란색)
                cv2.rectangle(roi_color, (real_ex, real_ey), (real_ex+real_ew, real_ey+real_eh), (0, 0, 255), 2)
    else:
        print("얼굴을 찾지 못했습니다.")

    # --- 시각화 ---
    plt.figure(figsize=(10, 6))
    plt.imshow(img_rgb)
    plt.title(f"Face & Eye Detection (Haar Cascade)\nFound {len(faces)} face(s)")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
