# -*- coding: utf-8 -*- 
# @Time : 2020/12/23 10:12 
# @Author : Digger
# @File : 图像分割FCN.py

import tensorflow as tf
import glob
import matplotlib.pyplot as plt
import numpy as np


def show(img_path, annotation_path):
    """
    展示图片和图像分割效果
    :param img_path: 原图存储地址
    :param annotation_path: 分割图像存储地址
    """
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img)
    annotation = tf.io.read_file(annotation_path)
    annotation = tf.image.decode_png(annotation)
    # print(annotation.shape)
    # 对矩阵进行压缩
    annotation = tf.squeeze(annotation)
    # print(annotation.shape)
    plt.subplot(1, 2, 1)
    plt.imshow(img.numpy())
    plt.subplot(1, 2, 2)
    plt.imshow(annotation.numpy())
    plt.show()


def data_process():
    """
    处理数据，创建训练数据集和测试数据集
    :return: 训练数据集、测试数据集、STEPS_PER_EPOCH、VALIDATION_STEPS
    """
    # 获取所有图片的存储地址
    images = glob.glob("F:\\dataset\\图片定位与分割\\images\\*.jpg")
    # 对图片地址按照图片名称进行排序
    images.sort(key=lambda x: x.split('/')[-1])
    # 获取所有图像分割后的图片
    annotations = glob.glob("F:\\dataset\\图片定位与分割\\annotations\\trimaps\\*.png")
    # 对分割图像按照名称排序
    annotations.sort(key=lambda x: x.split('/')[-1])
    # 生成长度与图片个数相同的随机序列
    np.random.seed(2020)
    index = np.random.permutation(len(images))
    # 将图片和分割图像地址一起随机
    images = np.array(images)[index]
    annotations = np.array(annotations)[index]
    # 创建dataset数据集
    dataset = tf.data.Dataset.from_tensor_slices((images, annotations))
    # 划分train数据集和test数据集（20%作为测试数据）
    test_count = int(len(images) * 0.2)
    train_count = len(images) - test_count
    train_ds = dataset.skip(test_count)
    test_ds = dataset.take(train_count)
    # 对训练数据和测试数据进行处理
    train_ds = train_ds.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_ds = test_ds.map(load_image)
    BATCH_SIZE = 8
    BUFFER_SIZE = 100
    STEPS_PER_EPOCH = train_count // BATCH_SIZE
    VALIDATION_STEPS = test_count // BATCH_SIZE
    # 对训练数据和测试数据做随机、重复、分批次
    train_ds = train_ds.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    train_ds = train_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    test_ds = test_ds.batch(BATCH_SIZE)
    return train_ds, test_ds, STEPS_PER_EPOCH, VALIDATION_STEPS


def read_jpg(path):
    """
    将图片解码成JPG格式
    :param path: 图片的存储地址
    :return: JPG格式的图片
    """
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    return img


def read_png(path):
    """
    将图片解码成PNG格式
    :param path: 图片的存储地址
    :return: PNG格式的图片
    """
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=1)
    return img


def normalize(input_image, input_mask):
    """
    对图片进行归一化处理，分割信息编号从0开始，将[1，2，3]变为[0，1，2]
    :param input_image: 输入图片
    :param input_mask: 输入的分割图片信息
    :return: 归一化的图片和改变编码的分割图片信息
    """
    input_image = tf.cast(input_image, tf.float32) / 127.5 - 1
    # 让定位信息中的分类从0开始编号
    input_mask -= 1
    return input_image, input_mask


def load_image(input_image_path, input_mask_path):
    """
    加载图片和分割图像，以及进行预处理
    :param input_image_path: 图片的存储路径
    :param input_mask_path: 分割图像的存储路径
    :return: 处理之后的图片和分割图像
    """
    input_image = read_jpg(input_image_path)
    input_image = tf.image.resize(input_image, (224, 224))
    input_mask = read_png(input_mask_path)
    input_mask = tf.image.resize(input_mask, (224, 224))
    input_image, input_mask = normalize(input_image, input_mask)
    return input_image, input_mask


def create_model():
    """
    创建训练模型
    :return:创建好的模型
    """
    # 使用预训练网络VGG16
    conv_base = tf.keras.applications.VGG16(weights='imagenet', input_shape=(224, 224, 3), include_top=False)
    print(conv_base.summary())
    # 创建一个元组记录要提取的网络层名
    layer_names = ['block5_conv3',  # 14*14
                   'block4_conv3',  # 28*28
                   'block3_conv3',  # 56*56
                   'block5_pool']
    # 根据网络层名获取对应网络层的输出
    layers_output = [conv_base.get_layer(name).output for name in layer_names]
    # 创建特征提取模型
    down_stack = tf.keras.Model(inputs=conv_base.inputs, outputs=layers_output)
    # 将网络设为不可训练状态
    down_stack.trainable = False
    # 设置输入格式
    inputs = tf.keras.layers.Input(shape=(224, 224, 3))
    # 获取特征提取模型的输出
    out_block5, out_block4, out_block3, out_pool = down_stack(inputs)
    # 将VGG16的最后一个pool层进行反卷积
    x1 = tf.keras.layers.Conv2DTranspose(512, 3, padding='same', strides=2, activation='relu')(out_pool)  # 14*14
    # 使用一层卷积进行特征提取
    x1 = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu')(x1)  # 14*14
    # 将进行特征提取后的卷积层与block5_conv3相加
    c1 = tf.add(x1, out_block5)
    # 将相加所得的结果进行反卷积
    x2 = tf.keras.layers.Conv2DTranspose(512, 3, padding='same', strides=2, activation='relu')(c1)  # 28*28
    # 特征提取
    x2 = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu')(x2)  # 28*28
    # 将进行特征提取后的卷积层与block4_conv3相加
    c2 = tf.add(x2, out_block4)
    # 将相加所得的结果进行反卷积
    x3 = tf.keras.layers.Conv2DTranspose(256, 3, padding='same', strides=2, activation='relu')(c2)  # 56*56
    # 特征提取
    x3 = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(x3)  # 56*56
    # 将进行特征提取后的卷积层与block3_conv3相加
    c3 = tf.add(x3, out_block3)
    x4 = tf.keras.layers.Conv2DTranspose(128, 3, padding='same', strides=2, activation='relu')(c3)  # 112*112
    x4 = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x4)  # 112*112
    # 预测输出层
    predictions = tf.keras.layers.Conv2DTranspose(3, 3, padding='same', strides=2, activation='softmax')(x4)  # 224*224
    # 建立模型
    model = tf.keras.models.Model(inputs=inputs, outputs=predictions)
    return model


def train(model, train_ds, test_ds, epochs, STEPS_PER_EPOCH, VALIDATION_STEPS):
    """
    模型训练
    :param model: 模型
    :param train_ds: 训练数据集
    :param test_ds: 验证数据集
    :param epochs:
    :param STEPS_PER_EPOCH:
    :param VALIDATION_STEPS:
    :return: 训练结果
    """
    # 模型配置编译
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # 模型训练
    history = model.fit(train_ds, epochs=epochs, steps_per_epoch=STEPS_PER_EPOCH, validation_data=test_ds,
                        validation_steps=VALIDATION_STEPS)
    return history


def show_history(history):
    """
    绘制训练结果loss
    :param history: 训练结果
    """
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(EPOCHS)
    plt.figure()
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'bo', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.Xlabel('Epoch')
    plt.Ylabel('Loss Value')
    plt.Ylim([0, 1])
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # show()
    train_ds, test_ds, STEPS_PER_EPOCH, VALIDATION_STEPS = data_process()
    model = create_model()
    EPOCHS = 20
    history = train(model, train_ds, test_ds, EPOCHS, STEPS_PER_EPOCH, VALIDATION_STEPS)
    show_history(history)
