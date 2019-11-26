# coding utf-8
import numpy as np
import random
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D, normalization
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import load_model
from keras import backend as K
from keras.callbacks import TensorBoard as tb
from load_face_dataset import load_dataset, resize_image, IMAGE_SIZE
import matplotlib.pyplot as plt


class Dataset:
    def __init__(self, path_name):
        # 训练集
        self.train_images = None
        self.train_labels = None

        # 验证集
        self.valid_images = None
        self.valid_labels = None

        # 测试集
        self.test_images = None
        self.test_labels = None

        # 数据集加载路径
        self.path_name = path_name

        # 当前库采用的维度顺序
        self.input_shape = None

    # 加载数据集并按照交叉验证的原则划分数据集并进行相关预处理工作
    def load(self, img_rows=IMAGE_SIZE, img_cols=IMAGE_SIZE,
             img_channels=3, nb_classes=2):
        # 加载数据集到内存
        images, labels = load_dataset(self.path_name)

        train_images, valid_images, train_labels, valid_labels = train_test_split(images, labels, test_size=0.3,
                                                                                  random_state=random.randint(0, 100))
        _, test_images, _, test_labels = train_test_split(images, labels, test_size=0.5,
                                                          random_state=random.randint(0, 100))

        # 通道 行 列顺序
        # theano 作为后端：channels,rows,cols  TensorFlow作为后端:rows,cols,channels
        if K.image_dim_ordering() == 'th':
            train_images = train_images.reshape(train_images.shape[0], img_channels, img_rows, img_cols)
            valid_images = valid_images.reshape(valid_images.shape[0], img_channels, img_rows, img_cols)
            test_images = test_images.reshape(test_images.shape[0], img_channels, img_rows, img_cols)
            self.input_shape = (img_channels, img_rows, img_cols)
        else:
            train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, img_channels)
            valid_images = valid_images.reshape(valid_images.shape[0], img_rows, img_cols, img_channels)
            test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, img_channels)
            self.input_shape = (img_rows, img_cols, img_channels)

            # 输出训练集、验证集、测试集的数量
            print(train_images.shape[0], '训练数据')
            print(valid_images.shape[0], '验证数据')
            print(test_images.shape[0], '测试数据')

            # 将类别标签进行one-hot编码使其向量化，两种类别转为二维数组
            train_labels = np_utils.to_categorical(train_labels, nb_classes)
            valid_labels = np_utils.to_categorical(valid_labels, nb_classes)
            test_labels = np_utils.to_categorical(test_labels, nb_classes)

            # 像素数据浮点化以便归一化
            train_images = train_images.astype('float32')
            valid_images = valid_images.astype('float32')
            test_images = test_images.astype('float32')

            # 将其归一化,图像的各像素值归一化到0~1区间
            train_images /= 255
            valid_images /= 255
            test_images /= 255

            self.train_images = train_images
            self.valid_images = valid_images
            self.test_images = test_images
            self.train_labels = train_labels
            self.valid_labels = valid_labels
            self.test_labels = test_labels


# CNN网络模型类
class Model:
    def __init__(self):
        self.model = None

    # 建立模型
    def build_model(self, dataset, nb_classes=2):
        # 构建一个空的网络模型
        self.model = Sequential()
        # 第一层
        self.model.add(Convolution2D(filters=16, kernel_size=(3, 3), padding='same', input_shape=dataset.input_shape,name='Conv_1'))
        # self.model.add(BatchNormalization(input_shape=dataset.input_shape))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='Pool_1'))
        # 第二层
        self.model.add(Convolution2D(filters=32, kernel_size=(3, 3), padding='same', name='Conv_2'))
        # self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='Pool_2'))
        # 第三层
        self.model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same', name='Conv_3'))
        # self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        # 第四层
        self.model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same', name='Conv4'))
        # self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='Pool4'))
        # 第六层
        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        # 第七层
        self.model.add(Dense(nb_classes))
        self.model.add(Activation('softmax'))

        # 输出模型概况
        self.model.summary()

        # 训练模型

    def train(self, dataset, batch_size=20, epochs=10, data_up=True):
        # 生成SGD优化器进行训练
        sgd = SGD(lr=0.01, decay=1e-6,momentum=0.9, nesterov=True)
        # 完成实际的模型配置工作
        self.model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

        # 数据提升
        if not data_up:
            self.model.fit(dataset.train_images,
                           dataset.train_labels,
                           batch_size=batch_size,
                           epochs=epochs,
                           validation_data=(dataset.valid_images, dataset.valid_labels),
                           shuffle=True)
        # 使用实时数据提升
        else:
            # 定义数据生成器
            datagen = ImageDataGenerator(
                rotation_range=20,                     # 数据提升时图片随机转动的角度
                width_shift_range=0.2,                 # 数据提升时图片水平偏移的幅度，百分比
                height_shift_range=0.2,                # 垂直平移幅度
                horizontal_flip=True,                  # 随机水平翻转
                vertical_flip=False)                   # 随机垂直翻转
            # 计算整个训练样本集的数量
            datagen.fit(dataset.train_images)

            # 利用生成器开始训练模型
            self.model.fit_generator(datagen.flow(dataset.train_images, dataset.train_labels,
                                     batch_size=batch_size),
                                     samples_per_epoch=dataset.train_images.shape[0],
                                     epochs=epochs,
                                     validation_data=(dataset.valid_images, dataset.valid_labels))

            # 构造TensorBoard
            tbCallBack = tb(log_dir="/Users/heyiheng/Desktop/biyelunwen/LastDemo/logs",
                            histogram_freq=1,
                            batch_size=batch_size,
                            write_grads=True)
            history = self.model.fit(dataset.train_images, dataset.train_labels,
                                     batch_size=batch_size,
                                     epochs=epochs,
                                     shuffle=True,
                                     verbose=2,
                                     validation_split=0.2,
                                     callbacks=[tbCallBack])
            return model, history

        # 识别人脸


    MODEL_PATH = './my_face_model.h5'

    def save_model(self, file_path=MODEL_PATH):
        self.model.save(file_path)

    def load_model(self, file_path=MODEL_PATH):
        self.model = load_model(file_path)

    def evaluate(self, dataset):
        score = self.model.evaluate(dataset.test_images, dataset.test_labels, verbose=1)
        print("%s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))


if __name__ == '__main__':
    dataset = Dataset(path_name='/Users/heyiheng/Desktop/biyelunwen/LastDemo/data')
    dataset.load()

    # 训练模型
    model = Model()
    model.build_model(dataset)
    model.train(dataset)
    model.save_model(file_path='/Users/heyiheng/Desktop/biyelunwen/LastDemo/model/my_face_model.h5')

    # # 评估模型
    # model = Model()
    # model.load_model(file_path='/Users/heyiheng/Desktop/biyelunwen/LastDemo/model/my_face_model.h5')
    # model.evaluate(dataset)
