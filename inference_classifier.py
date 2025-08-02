import cv2
import pickle
import numpy as np
from collections import deque
import mediapipe as mp
import time

# --- Load model ---
with open('model.p', 'rb') as f:
    model_data = pickle.load(f)
    model = model_data['model']
    labels_dict = model_data['labels_dict']

# --- MediaPipe ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5
)

# --- Sliding Window ---
recent_predictions = deque(maxlen=30)  # ~2 giây nếu 15 FPS
stable_letter = None
predicted_letter = ""
current_text = ""
THRESHOLD = 20  # Số lần lặp để xác nhận ký tự (~2 giây)

# --- Webcam ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Không mở được webcam.")
    exit()

print("👋 Nhận diện cử chỉ tay bắt đầu...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    data_aux = []
    x_, y_ = [], []

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        for lm in hand_landmarks.landmark:
            x, y = lm.x, lm.y
            x_.append(x)
            y_.append(y)

        for lm in hand_landmarks.landmark:
            data_aux.append(lm.x - min(x_))
            data_aux.append(lm.y - min(y_))

        if len(data_aux) == 42:
            prediction = model.predict([data_aux])[0]
            recent_predictions.append(prediction)

            most_common = max(set(recent_predictions), key=recent_predictions.count)
            count = recent_predictions.count(most_common)

            if count >= THRESHOLD and most_common != stable_letter:
                stable_letter = most_common
                predicted_letter = stable_letter

                if stable_letter == 'DEL':
                    current_text = current_text[:-1]
                elif stable_letter == 'SPACE':
                    current_text += ' '
                else:
                    current_text += stable_letter

                print(f"📌 Nhận ký tự: {stable_letter}")

        # Vẽ khung quanh bàn tay
        x_list = [int(lm.x * w) for lm in hand_landmarks.landmark]
        y_list = [int(lm.y * h) for lm in hand_landmarks.landmark]
        x_min, x_max = min(x_list), max(x_list)
        y_min, y_max = min(y_list), max(y_list)

        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

        if predicted_letter:
            cv2.putText(frame, f"Kytu: {predicted_letter}", (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        mp_drawing.draw_landmarks(
            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
        )

    # Hiển thị văn bản hiện tại + hướng dẫn
    cv2.rectangle(frame, (0, 0), (w, 90), (0, 0, 0), -1)
    cv2.putText(frame, f"Text: {current_text}", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Hướng dẫn người dùng
    cv2.putText(frame, "DEL=s | Reset=c | Save=s | Xoa 1 ky tu=Backspace | Thoat=ESC", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("Sign Language Recognition", frame)

    key = cv2.waitKey(1)
    if key == ord('c'):
        current_text = ""
        print("🧹 Đã xóa toàn bộ văn bản.")
    elif key == ord('s'):
        with open("output_text.txt", "w", encoding="utf-8") as f:
            f.write(current_text)
        print("💾 Đã lưu văn bản ra output_text.txt")
    elif key == 8:  # Backspace
        current_text = current_text[:-1]
        print("⌫ Đã xóa 1 ký tự.")
    elif key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
