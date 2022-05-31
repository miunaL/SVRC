#coding=GBK

import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler
import struct
from sklearn import preprocessing
from sklearn.preprocessing import normalize
from Logger import *
# ѵ�����ļ�
train_images_idx3_ubyte_file = 'train-images.idx3-ubyte'
# ѵ������ǩ�ļ�
train_labels_idx1_ubyte_file = 'train-labels.idx1-ubyte'

def decode_idx3_ubyte(idx3_ubyte_file):
    """
    ����idx3�ļ���ͨ�ú���
    :param idx3_ubyte_file: idx3�ļ�·��
    :return: ���ݼ�
    """
    # ��ȡ����������
    bin_data = open(idx3_ubyte_file, 'rb').read()

    # �����ļ�ͷ��Ϣ������Ϊħ����ͼƬ������ÿ��ͼƬ�ߡ�ÿ��ͼƬ��
    offset = 0
    fmt_header = '>iiii'   #'>IIII'��˵ʹ�ô�˷���ȡ4��unsinged int32
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    print('ħ��:%d, ͼƬ����: %d��, ͼƬ��С: %d*%d' % (magic_number, num_images, num_rows, num_cols))

    # �������ݼ�
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    print("offset: ",offset)
    fmt_image = '>' + str(image_size) + 'B'   # '>784B'����˼�����ô�˷���ȡ784��unsigned byte
    images = np.empty((num_images, num_rows*num_cols))
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('�ѽ��� %d' % (i + 1) + '��')
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows*num_cols))
        offset += struct.calcsize(fmt_image)
    return images.T


def decode_idx1_ubyte(idx1_ubyte_file):
    """
    ����idx1�ļ���ͨ�ú���
    :param idx1_ubyte_file: idx1�ļ�·��
    :return: ���ݼ�
    """
    # ��ȡ����������
    bin_data = open(idx1_ubyte_file, 'rb').read()

    # �����ļ�ͷ��Ϣ������Ϊħ���ͱ�ǩ��
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print('ħ��:%d, ͼƬ����: %d��' % (magic_number, num_images))

    # �������ݼ�
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('�ѽ��� %d' % (i + 1) + '��')
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels


def load_train_images(idx_ubyte_file=train_images_idx3_ubyte_file):
    """
    TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  60000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).
    :param idx_ubyte_file: idx�ļ�·��
    :return: n*row*colάnp.array����,nΪͼƬ����
    """
    return decode_idx3_ubyte(idx_ubyte_file)


def load_train_labels(idx_ubyte_file=train_labels_idx1_ubyte_file):
    """
    TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  60000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    The labels values are 0 to 9.
    :param idx_ubyte_file: idx�ļ�·��
    :return: n*1άnp.array����,nΪͼƬ����
    """
    return decode_idx1_ubyte(idx_ubyte_file)

def getLoadBinaryClassData():
    path = "E:\\WorkStation\\LR\\"
    dataX = pd.read_csv(path+"trainX.csv")
    dataY = pd.read_csv(path+"trainY.csv", header=None)
    min_max_scaler = MinMaxScaler()
    X = min_max_scaler.fit_transform(dataX).T
    Y = np.concatenate((np.mat(dataY).T, 1-np.mat(dataY).T), axis=0)
#     Y = mat([mat(dataY).T, 1 - mat(dataY).T])
    return (X, Y)

def getMnist49(path):
    # def testRun():
    train_images = load_train_images(path+"train-images.idx3-ubyte").T #(num_rows*num_cols,num_images)
    b = np.ones(train_images.shape[0])
    train_images = np.column_stack((train_images,b))
    train_labels = load_train_labels(path+"train-labels.idx1-ubyte")
    print("train_images shape", train_images.shape)
    print("train+labels shape", train_labels.shape)
    selectIndex = np.logical_or(train_labels==4, train_labels==9)
    train_49_images = train_images[selectIndex,:]
    train_49_labels = train_labels[selectIndex]
    print(train_49_images.shape)
    print(train_49_labels.shape)
    # X = np.mat(scale(256-train_49_images)).T
    X = np.mat(train_49_images).T
    # print("sum > 300", (X.sum(axis=1)).mean(axis=0))
    X = np.mat(normalize(X, axis=1, norm='l2'))
    Y = np.mat(train_49_labels==4)
    return (X, Y)

def getMnist(path):
    # def testRun():
    train_images = load_train_images(path+"train-images.idx3-ubyte").T #(num_rows*num_cols,num_images)
    # b = np.ones(train_images.shape[0])
    # train_images = np.column_stack((train_images,b))
    train_labels = load_train_labels(path+"train-labels.idx1-ubyte")
    print("train_images shape", train_images.shape)
    print("train_labels shape", train_labels.shape)
    X = np.mat(train_images).T
    X = np.mat(normalize(X, axis=0, norm='l2'))
    # X = np.mat(preprocessing.scale(X, axis=1))
    Y = pd.DataFrame(train_labels)
    Y = np.mat(pd.get_dummies(Y[:][0]).astype(int)).T
    return (X[:,:1000], Y[:,:1000])

def getMnistWithNumber(path, number):
    # def testRun():
    train_images = load_train_images(path+"train-images.idx3-ubyte").T #(num_rows*num_cols,num_images)
    # b = np.ones(train_images.shape[0])
    # train_images = np.column_stack((train_images,b))
    train_labels = load_train_labels(path+"train-labels.idx1-ubyte")
    print("train_images shape", train_images.shape)
    print("train+labels shape", train_labels.shape)
    selectIndex = np.zeros((train_labels.shape))
    for i in number:
        selectIndex = np.logical_or(train_labels == i, selectIndex)
    # selectIndex = np.logical_or(train_labels==4, train_labels==9)
    train_images = train_images[selectIndex,:]
    train_labels = train_labels[selectIndex]
    Logger.log("cate number :"+str(number))
    Logger.log("train data shape :" + str(train_images.shape))
    X = np.mat(train_images).T#/256
    X = np.mat(normalize(X, axis=0, norm='l2'))
    print(train_labels[:20])
    Y = pd.DataFrame(train_labels)
    Y = np.mat(pd.get_dummies(Y[:][0]).astype(int)).T
    # return (X, Y)
    num = 60000
    X = np.mat(X[:,:num])
    Y = np.mat(Y[:,:num])
    Logger.log("here wo use "+str(num)+" data points")
    return (X, Y)