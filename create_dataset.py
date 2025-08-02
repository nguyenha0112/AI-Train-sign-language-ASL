import os
import pickle
import cv2
import mediapipe as mp

# Khởi tạo MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'
data = []
labels = []

for class_name in os.listdir(DATA_DIR):
    class_path = os.path.join(DATA_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    for hand_type in os.listdir(class_path):
        hand_path = os.path.join(class_path, hand_type)
        for angle in os.listdir(hand_path):
            angle_path = os.path.join(hand_path, angle)

            for img_name in os.listdir(angle_path):
                img_path = os.path.join(angle_path, img_name)
                img = cv2.imread(img_path)
                if img is None:
                    continue

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands.process(img_rgb)

                if results.multi_hand_landmarks:
                    landmarks = results.multi_hand_landmarks[0]

                    x_, y_, data_aux = [], [], []
                    for lm in landmarks.landmark:
                        x_.append(lm.x)
                        y_.append(lm.y)

                    for lm in landmarks.landmark:
                        data_aux.append(lm.x - min(x_))
                        data_aux.append(lm.y - min(y_))

                    if len(data_aux) == 42:
                        data.append(data_aux)
                        labels.append(class_name)  # 🟢 Chỉ ghi chữ cái (A-Z)

hands.close()

# Lưu pickle
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("✅ Đã tạo data.pickle với nhãn là chữ cái A-Z.")
