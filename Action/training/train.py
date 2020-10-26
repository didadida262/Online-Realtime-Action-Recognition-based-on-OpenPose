import pandas as pd
from enum import Enum
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.models import load_model

import matplotlib.pyplot as plt
from keras.callbacks import Callback
import itertools
from sklearn.metrics import confusion_matrix


class Actions(Enum):
    # framewise_recognition.h5
    # squat = 0
    # stand = 1
    # walk = 2
    # wave = 3

    # framewise_recognition_under_scene.h5
    # stand = 0
    # walk = 1
    # operate = 2
    # fall_down = 3
    # run = 4


    # framewise_recognition_lalal.h5
    stand = 0
    hug = 1
    salute = 2
    smoke = 3
    bow = 4
    crane = 5



# Callback class to visialize training progress
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('accuracy'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_accuracy'))

    def loss_plot(self, loss_type):
        print('self.losses:',self.losses[loss_type])
        print('self.accuracy:',self.accuracy[loss_type])
        print('self.val_loss:',self.val_loss[loss_type])
        print('self.val_acc:',self.val_acc[loss_type])
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# load data
raw_data = pd.read_csv('data_first_dealed.csv', header=0)
dataset = raw_data.values
X = dataset[:, 0:35].astype(float)
Y = dataset[:, 35]
# X = dataset[0:3289, 0:36].astype(float)  # 忽略run数据
# Y = dataset[0:3289, 36]

# 将类别编码为数字
# encoder = LabelEncoder()
# encoder_Y = encoder.fit_transform(Y)
# print(encoder_Y[0], encoder_Y[900], encoder_Y[1800], encoder_Y[2700])
# encoder_Y = [0]*744 + [1]*722 + [2]*815 + [3]*1008 + [4]*811
# encoder_Y = [0]*744 + [1]*722 + [2]*815 + [3]*1008
# encoder_Y = [0]*745 + [1]*722 + [2]*668 + [3]*691

#   训练数据六个动作共计12000条数据，每个动作2000条 
#   encoder_Y就是个一维数组 
encoder_Y = [0]*2000 + [1]*2000 + [2]*2000 + [3]*2000 + [4]*2000 + [5]*2000
# one hot 编码,返回标签数组表示化
dummy_Y = np_utils.to_categorical(encoder_Y)

# train test split
# print('encoder_Y',encoder_Y)
# print('dummy_Y',dummy_Y)
# print('dummy_Y',dummy_Y.shape)
# print('Y',Y)
# print('Y',Y.shape)
# X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_Y, test_size=0.1, random_state=4)
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=1)

X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_Y, test_size=0.1, random_state=0)

# build keras model
# vgg默认卷积核3x3
model = Sequential()
# 定义一个有128个节点的神经层，激活函数是relu
model.add(Dense(units=128, activation='relu'))
# 加速训练过程
model.add(BatchNormalization())
model.add(Dense(units=64, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(units=16, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(units=6, activation='softmax'))  # units = nums of classes

# training
his = LossHistory()
model.compile(loss='categorical_crossentropy', optimizer=Adam(0.0001), metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=32, epochs=10, verbose=1, validation_data=(X_test, Y_test), callbacks=[his])
model.summary()
his.loss_plot('epoch')
model.save('framewise_recognition_test_lalal.h5')

# # evaluate and draw confusion matrix
print('Test:')
score, accuracy = model.evaluate(X_test,Y_test,batch_size=32)
print('Test Score:{:.3}'.format(score))
print('Test accuracy:{:.3}'.format(accuracy))
# # confusion matrix
Y_pred = model.predict(X_test)
cfm = confusion_matrix(np.argmax(Y_test,axis=1), np.argmax(Y_pred, axis=1))
np.set_printoptions(precision=2)
#
plt.figure()
# class_names = ['squat', 'stand', 'walk', 'wave']
class_names = ['stand','hug','salute','smoke','bow','crane']
plot_confusion_matrix(cfm, classes=class_names, title='Confusion Matrix')
plt.show()

# # test
model = load_model('framewise_recognition_test_lalal.h5')

test_input = [0.46, 0.43, 0.52, 0.4, 0.52, 0.39, 0.61, 0.4,
              0.67, 0.46, 0.52, 0.46, 0.61, 0.46, 0.67, 0.42, 0.67,
              0.42, 0.81, 0.43, 0.91, 0.45, 0.67, 0.45, 0.81, 0.45,
              0.91, 0.42, 0.44, 0.43, 0.44, 0.42, 0.46, 0.44, 0.46]
test_np = np.array(test_input)
test_np = test_np.reshape(-1, 36)

test_np = np.array(X[1033]).reshape(-1, 36)
print('test_np:',test_np)
if test_np.size > 0:
    pred = np.argmax(model.predict(test_np))
    init_label = Actions(pred).name
    print(init_label)
