import os
import cv2
import string

# Cau hinh
classes = list(string.ascii_uppercase)
angles = ['front', 'left', 'right', 'close', 'far']
hands = ['left_hand', 'right_hand']
dataset_size = 10
DATA_DIR = './data'
BOX_SIZE = 300
MARGIN = 50  # Khoang cach le khung tu bien trai/phai

# Tao thu muc
for class_name in classes:
    for hand in hands:
        for angle in angles:
            path = os.path.join(DATA_DIR, class_name, hand, angle)
            os.makedirs(path, exist_ok=True)

# Mo webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Khong mo duoc webcam.")
    exit()

# Vong lap chinh
for class_name in classes:
    for hand in hands:
        for angle in angles:
            folder_path = os.path.join(DATA_DIR, class_name, hand, angle)
            print(f'\nSan sang cho: {class_name} | {hand} | {angle}')
            print(">>> Dat tay vao khung. Bam Q de bat dau chup...")

            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Loi webcam.")
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()

                h, w, _ = frame.shape

                # Xac dinh toa do khung dua vao ben tay
                if hand == 'right_hand':
                    x1 = MARGIN
                else:  # left_hand
                    x1 = w - BOX_SIZE - MARGIN
                y1 = h // 2 - BOX_SIZE // 2
                x2 = x1 + BOX_SIZE
                y2 = y1 + BOX_SIZE

                # Ve khung chu nhat
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text = f'{class_name} | {hand} | {angle} - Bam Q de chup'
                cv2.putText(frame, text, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow('frame', frame)

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

            # Chup anh
            start_index = len(os.listdir(folder_path))
            counter = 0
            while counter < dataset_size:
                ret, frame = cap.read()
                if not ret:
                    print("Loi khi chup anh.")
                    break

                h, w, _ = frame.shape
                # Xac dinh lai khung chup
                if hand == 'right_hand':
                    x1 = MARGIN
                else:
                    x1 = w - BOX_SIZE - MARGIN
                y1 = h // 2 - BOX_SIZE // 2
                x2 = x1 + BOX_SIZE
                y2 = y1 + BOX_SIZE

                crop = frame[y1:y2, x1:x2]
                filename = f'{start_index + counter}.jpg'
                save_path = os.path.join(folder_path, filename)
                cv2.imwrite(save_path, crop)
                print(f'✅ Da luu: {save_path}')

                cv2.imshow('frame', frame)
                counter += 1
                cv2.waitKey(100)

cap.release()
cv2.destroyAllWindows()
print("\nHoan tat thu thap du lieu.")
