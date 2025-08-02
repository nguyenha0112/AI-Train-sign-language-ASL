import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Kiểm tra file data.pickle
if not os.path.exists('data.pickle'):
    print("Lỗi: File data.pickle không tồn tại!")
    exit()

# Đọc dữ liệu
with open('data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])  # labels là ['A', 'B', ...]

# Kiểm tra dữ liệu
if len(data) == 0 or len(labels) == 0:
    print("Lỗi: Dữ liệu hoặc nhãn rỗng!")
    exit()

# Chia dữ liệu train/test
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)

# Train model
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(x_train, y_train)

# Đánh giá
y_pred = model.predict(x_test)
score = accuracy_score(y_pred, y_test)
print('{:.2f}% mẫu được phân loại đúng!'.format(score * 100))

# Tạo labels_dict: {'A': 'A', 'B': 'B', ...} (không chuyển về số nữa)
unique_labels = sorted(set(labels))
labels_dict = {label: label for label in unique_labels}

# Lưu mô hình
with open('model.p', 'wb') as f:
    pickle.dump({'model': model, 'labels_dict': labels_dict}, f)

print("✅ Đã lưu model và labels_dict vào file model.p")
