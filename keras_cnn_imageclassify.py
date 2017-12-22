# -*- coding: utf-8 -*-
import os
import cv2
import time
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Dense, Activation, Dropout

#再現性を持たせるためrandom seed指定
np.random.seed(0)


#ディレクトリ内画像読み込み関数
#inputpathのディレクトリ内にある全ての画像ファイルを読み込みimgsdataに格納して返す
def load_images(inputpath):
    imglist = []

    for root, dirs, files in os.walk(inputpath):
        for fn in sorted(files):
            bn, ext = os.path.splitext(fn)
            if ext not in [".bmp", ".BMP", ".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"]:
                continue

            filename = os.path.join(root, fn)
            
            #画像の読み込み
            #カラー画像の場合
            #testimage = cv2.imread(filename, cv2.IMREAD_COLOR)
            # サイズ変更
            #height, width = testimage.shape[:2]
            #testimage = cv2.resize(testimage, (38, 38), interpolation = cv2.INTER_AREA)  #主に縮小するのでINTER_AREA使用
            #testimage = np.asarray(testimage, dtype=np.float64)
            # チャンネル，高さ，幅に入れ替え
            #testimage = testimage.transpose(2, 0, 1)
            # チャンネルをbgrの順からrgbの順に変更
            #testimage = testimage[::-1]

            
            # グレースケールの場合
            testimage = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            # サイズ変更
            height, width = testimage.shape[:2]
            testimage = cv2.resize(testimage, (20, 20), interpolation = cv2.INTER_AREA)  #主に縮小するのでINTER_AREA使用
            # チャンネルの次元がないので1次元追加する
            testimage = np.asarray([testimage], dtype=np.float64)
            testimage = np.asarray(testimage, dtype=np.float64).reshape((1, 20, 20))


            # 高さ，幅，チャンネルに入れ替え．data_format="channels_last"を使うとき必要
            testimage = testimage.transpose(1, 2, 0)
            
            imglist.append(testimage)
    imgsdata = np.asarray(imglist, dtype=np.float32)

    return imgsdata


#画像読み込みとラベル作成
data_class0 = load_images(".\\images_class0\\")
label_class0 = []
for i in range(len(data_class0)):
    label_class0.append(0)

data_class1 = load_images(".\\images_class1\\")
label_class1 = []
for i in range(len(data_class1)):
    label_class1.append(1)
    
data_all = np.vstack((data_class0, data_class1))
label_all = label_class0 + label_class1

print("Loaded image number: " + repr(len(data_all)))

#読み込んだ画像を学習データとテストデータに分割（8:2）
data_train, data_test, label_train, label_test = train_test_split(data_all, label_all, test_size=0.2)

data_train = np.asarray(data_train, np.float32)
data_test = np.asarray(data_test, np.float32)
label_train = np.asarray(label_train, np.int32)
label_test = np.asarray(label_test, np.int32)

#画素値正規化
#とりあえず255で割る
data_train /= 255.0
data_test /= 255.0
#画素値の最大値で割る方法もある
#train_maxvalue = np.max(data_train)
#test_maxvalue = np.max(data_test)
#data_train /= train_maxvalue
#data_test /= test_maxvalue

label_train_binary = to_categorical(label_train)
label_test_binary = to_categorical(label_test)


#%%
#ネットワーク構築
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='valid', data_format="channels_last", input_shape=(20, 20, 1)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(3, 3), padding='same'))

model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='valid'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

#model.add(Conv2D(128, (3, 3), padding='valid'))
#model.add(Activation("relu"))
#model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

model.add(Flatten())

model.add(Dense(256))
model.add(Activation("relu"))
model.add(Dropout(0.5))

model.add(Dense(128))
model.add(Activation("relu"))
model.add(Dropout(0.5))

model.add(Dense(2))
model.add(Activation("softmax"))

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

print(model.summary())

#%%
#学習
start = time.time()  # 処理時間の計測開始
training = model.fit(data_train, label_train_binary,
                     epochs=50, batch_size=100, verbose=1)
training_time = time.time() - start


#評価
start = time.time()
result = model.predict_classes(data_test, verbose=1)
predict_time = time.time() - start


#%%
#認識率等結果を表示
score = accuracy_score(label_test, result)
print()
print("Training time: " + str(training_time) + "s, Prediction time: " + str(predict_time) + "s")
print("Classification accuracy on test data: " + str(score))
cmatrix = confusion_matrix(label_test, result)
print(cmatrix)


#学習結果をファイルに保存
#モデル
json_string = model.to_json()
open("model_json", "w").write(json_string)
#重み
model.save_weights('weight_hdf5')
