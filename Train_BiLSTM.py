import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional, Dropout
import numpy as np
from tensorflow import keras
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers
from keras.optimizers import Adam


# Đọc dữ liệu từ file CSV
# df = pd.read_csv('dataC3.csv')
# Với cột cuối là nhãn

import numpy as np
import pandas as pd


vui1 = np.loadtxt("/Users/nguyentrithanh/Documents/20232/MachineLearning/CapstoneProjectML/CollectedData/new_data/ThaiVui.txt")
vui2 = np.loadtxt("/Users/nguyentrithanh/Documents/20232/MachineLearning/CapstoneProjectML/CollectedData/new_data/ThaiVui2.txt")
vui3 = np.loadtxt("/Users/nguyentrithanh/Documents/20232/MachineLearning/CapstoneProjectML/CollectedData/new_data/QuangVui3.txt")
vui4 = np.loadtxt("/Users/nguyentrithanh/Documents/20232/MachineLearning/CapstoneProjectML/CollectedData/new_data/SonVui4.txt")
vui5 = np.loadtxt("/Users/nguyentrithanh/Documents/20232/MachineLearning/CapstoneProjectML/CollectedData/new_data/ThanhfVui2.txt")
vui6 = np.loadtxt("/Users/nguyentrithanh/Documents/20232/MachineLearning/CapstoneProjectML/CollectedData/new_data/ThanhfVui.txt")

vui = np.concatenate((vui1, vui2, vui3, vui4, vui5))
print(vui.shape)

buon1 = np.loadtxt("/Users/nguyentrithanh/Documents/20232/MachineLearning/CapstoneProjectML/CollectedData/new_data/BachBuon.txt")
buon2 = np.loadtxt("/Users/nguyentrithanh/Documents/20232/MachineLearning/CapstoneProjectML/CollectedData/new_data/ThaiBuon.txt")
buon3 = np.loadtxt("/Users/nguyentrithanh/Documents/20232/MachineLearning/CapstoneProjectML/CollectedData/new_data/SonBuon3.txt")
buon4 = np.loadtxt("/Users/nguyentrithanh/Documents/20232/MachineLearning/CapstoneProjectML/CollectedData/new_data/SonBuon4.txt")
buon5 = np.loadtxt("/Users/nguyentrithanh/Documents/20232/MachineLearning/CapstoneProjectML/CollectedData/new_data/ThanhfBuon.txt")


buon = np.concatenate((buon1, buon2, buon3, buon4, buon5))

calm1 = np.loadtxt("/Users/nguyentrithanh/Documents/20232/MachineLearning/CapstoneProjectML/CollectedData/new_data/ThaiCalm.txt")
calm2 = np.loadtxt("/Users/nguyentrithanh/Documents/20232/MachineLearning/CapstoneProjectML/CollectedData/new_data/BachCalm.txt")
calm3 = np.loadtxt("/Users/nguyentrithanh/Documents/20232/MachineLearning/CapstoneProjectML/CollectedData/new_data/QuangCalm3.txt")
calm4 = np.loadtxt("/Users/nguyentrithanh/Documents/20232/MachineLearning/CapstoneProjectML/CollectedData/new_data/ThaiCalm2.txt")
calm5 = np.loadtxt("/Users/nguyentrithanh/Documents/20232/MachineLearning/CapstoneProjectML/CollectedData/new_data/ThanhfCalm.txt")


calm = np.concatenate((calm1, calm2, calm3, calm4, calm5))




def SplitArray(arr):

    # Kích thước của các mảng con và số phần tử trùng lặp
    subarray_size = 15*512
    overlap = 14*512

    # Khởi tạo danh sách để lưu các mảng con
    subarrays = []

    # Sử dụng vòng lặp để tạo các mảng con
    for i in range(0, len(arr) - subarray_size + 1, subarray_size - overlap):
        subarray = arr[i:i + subarray_size]
        subarrays.append(subarray)

    # Chuyển danh sách các mảng con thành mảng NumPy
    subarrays = np.array(subarrays)

    return subarrays

calm = SplitArray(calm)
vui = SplitArray(vui)
buon = SplitArray(buon)


calm_df = pd.DataFrame(calm)
vui_df = pd.DataFrame(vui)
buon_df = pd.DataFrame(buon)
labels = [0] * 586 + [1] * 586 + [2] * 586
df = pd.concat((vui_df, buon_df, calm_df))
df['labels'] = labels
print(df)



X = df.iloc[:, :-1].values

y = df['labels'].values



# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X = scaler.fit_transform(X)

# One-hot encoding cho nhãn
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y.reshape(-1, 1))

# K-fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
fold_scores = []

# Tạo thư mục để lưu các plot và mô hình
model_dir = 'models'
os.makedirs(model_dir, exist_ok=True)

# Danh sách để lưu tên file của các mô hình đã train
model_filenames = []

# Tạo DataFrame để lưu history của loss và accuracy
history_df = pd.DataFrame(columns=['fold', 'epoch', 'train_loss', 'test_loss', 'train_accuracy', 'test_accuracy'])

for fold_index, (train_index, test_index) in enumerate(kfold.split(X), 1):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Chia thành tập huấn luyện và tập kiểm tra
    time_steps = 128
    n_features = X.shape[1]

    # Chuẩn bị đầu vào và đầu ra cho mô hình Bi-LSTM
    def prepare_data(data, labels, time_steps):
        X = []
        y = []
        for i in range(len(data) - time_steps):
            X.append(data[i:i+time_steps])
            y.append(labels[i+time_steps])
        return np.array(X), np.array(y)
    X_train, y_train = prepare_data(X_train, y_train, time_steps)
    X_test, y_test = prepare_data(X_test, y_test, time_steps)

    # Xây dựng mô hình Bi-LSTM
    model = Sequential()
    model.add(Bidirectional(LSTM(units=16, activation='relu', kernel_regularizer=regularizers.l2(0.01)), input_shape=(time_steps, n_features)))
    # model.add(Bidirectional(LSTM(units=16, activation='relu'), input_shape=(time_steps, n_features)))
    model.add(Dropout(0.1))
    model.add(Dense(units=16, activation='relu'))
    model.add(Dropout(0.1)) #0.2
    model.add(Dense(units=3, activation='softmax'))


    # model = Sequential()
    # # model.add(Input((1000,1)))
    # model.add(LSTM(128))
    # model.add(Dense(128, 'relu'))
    # model.add(Dense(3, 'softmax'))
    # model.summary()

    #learning rate
    lr = 0.0001
    optimizer = Adam(learning_rate=lr)

    # Biên dịch mô hình
    model.compile(optimizer=optimizer , loss='categorical_crossentropy', metrics=['accuracy'])

    # Đặt callbacks để dừng sau 10 epochs
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=10)

    # Huấn luyện mô hình và lưu history
    history = model.fit(X_train, y_train, epochs=200, batch_size=96, validation_data=(X_test, y_test), callbacks=[early_stopping])

    # Lưu mô hình
    model_filename = f"BiLSTM_Brainwave_{fold_index}_accuracy_{history.history['val_accuracy'][-1]*100:.2f}.h5"
    model.save(os.path.join(model_dir, model_filename))
    model_filenames.append(model_filename)

    # Đánh giá mô hình trên tập kiểm tra
    _, accuracy = model.evaluate(X_test, y_test)
    fold_scores.append(accuracy)

    # Vẽ đồ thị loss và accuracy
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')
    plt.savefig(f"{model_dir}/{os.path.splitext(model_filename)[0]}_loss.png")
    plt.clf()

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='lower right')
    plt.savefig(f"{model_dir}/{os.path.splitext(model_filename)[0]}_accuracy.png")
    plt.clf()

    # Lưu accuracy và loss của fold hiện tại vào DataFrame
    for epoch, (train_loss, test_loss, train_accuracy, test_accuracy) in enumerate(zip(
        history.history['loss'],
        history.history['val_loss'],
        history.history['accuracy'],
        history.history['val_accuracy']
    ), 1):
        history_df = pd.concat([history_df, pd.DataFrame({
            'fold': [fold_index],
            'epoch': [epoch],
            'train_loss': [train_loss],
            'test_loss': [test_loss],
            'train_accuracy': [train_accuracy],
            'test_accuracy': [test_accuracy]
        })], ignore_index=True)

    # Lưu thông tin của fold hiện tại vào file txt
    with open(f'{model_dir}/{os.path.splitext(model_filename)[0]}.txt', 'w') as file:
        file.write(f"Train Loss: {history.history['loss'][-1]}\n")
        file.write(f"Train Accuracy: {history.history['accuracy'][-1]}\n")
        file.write(f"Validation Loss: {history.history['val_loss'][-1]}\n")
        file.write(f"Validation Accuracy: {history.history['val_accuracy'][-1]}\n")

# In kết quả
for i, score in enumerate(fold_scores, 1):
    print(f"Fold {i}: Accuracy = {score}")

print("Average Accuracy:", np.mean(fold_scores))

# Lưu DataFrame vào file CSV
history_df.to_csv('history_bi_lstm.csv', index=False)
