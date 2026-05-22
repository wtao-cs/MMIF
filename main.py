import os
import pickle
import random

import keras
import numpy as np
from Fancy_aggregations.owas import OWA_generic
from keras.layers import Input, Conv1D, Activation, Dense, Dropout, Concatenate, \
    BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.utils import to_categorical
from sklearn.model_selection import KFold
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

from depression.paper.MMIF.GitHubUpload.utils import summayResults

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

ProjectRootPath = 'D:/ai_python/EEG_based_depression_detection'


def butter_bandpass(lowcut, highcut, fs, order=5):
    low = 2 * lowcut / fs
    high = 2 * highcut / fs
    from scipy.signal import butter
    b, a = butter(order, [low, high], btype='band')

    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5, axis=0):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    from scipy.signal import lfilter
    y = lfilter(b, a, data, axis=axis)
    return y

def loadEDRAFeaturesDatasetSource(filePath='EDRA_new_source.pkl'):
    filePath =  ProjectRootPath + "/depression/depressionDetectionDataset/"+filePath
    with open(filePath, 'rb') as file:
        dataset = pickle.load(file)

    train_x1 = dataset['features']
    train_y1 = dataset['label']
    train_x = []
    train_y = []
    for index in range(0, len(train_y1)):
        label = train_y1[index]
        temp = train_x1[index]
        train_x.append(temp)
        train_y.append([label] * temp.shape[0])
    train_x = np.vstack(train_x)
    train_y = np.hstack(train_y)

    return train_x, train_y


def ResBlock(x, filters, kernel_size, dilation_rate, F2):
    r = Conv1D(filters, kernel_size, padding='same', kernel_initializer='he_uniform', dilation_rate=dilation_rate,
               activation='relu'
               )(
        x)
    r = BatchNormalization()(r)

    r = Activation(activation='relu')(r)

    r = Dropout(0.1)(r)

    r = Conv1D(filters, kernel_size, padding='same', kernel_initializer='he_uniform', dilation_rate=dilation_rate,
               activation='relu'
               )(
        r)
    r = BatchNormalization()(r)
    r = Activation(activation='relu', name='deepFeature_output')(r)

    return r


def MIFFModal(x_train, y_train, x_test, y_test, x_features, y_features, return_sequences=False):
    inputs = Input(shape=(x_train.shape[1], x_train.shape[2]), name='input_1')

    x = ResBlock(inputs, filters=32, kernel_size=3, dilation_rate=1, F2=32)

    inputs_2 = Input(shape=(x_features.shape[1]), name='input_2')

    x = keras.layers.GlobalAvgPool1D(name='global_avg')(x)

    x = Concatenate(axis=-1)([x, inputs_2])
    x = Dense(2, activation='softmax', name='final_output')(x)
    return Model(inputs=[inputs, inputs_2], outputs=x)


def doEDRATest(x_train, y_train, x_test, y_test222, trn_dfRaw, test_Raw, task, folder):
    global is_power_of_two, AttentionLayer, ResBlock
    num_classes = 2
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test222, num_classes)

    model = MIFFModal(trn_dfRaw, y_train, test_Raw, y_test, x_train, x_test)
    optim = Adam(lr=0.001)
    model.summary()

    plot_model(model, to_file="cnn_model.png", show_shapes=True, show_layer_names=True)
    from tensorflow.keras.callbacks import EarlyStopping

    model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])

    from tensorflow.keras.callbacks import LearningRateScheduler

    def lr_schedule(epoch, lr):
        if epoch < 50:
            return lr
        elif epoch < 100:
            return 0.0001
        else:
            return 0.00001

    lr_scheduler = LearningRateScheduler(lr_schedule)

    # Define the EarlyStopping callback
    early_stopping = EarlyStopping(
        monitor='accuracy',  # Monitor the loss function
        patience=30,  # Number of consecutive epochs with no improvement to stop training
        verbose=1,  # Verbosity mode (1 = display messages, 0 = silent)
        restore_best_weights=False  # Restore the model weights from the epoch with the best value
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for i, (trn_index, val_index) in enumerate(kf.split(trn_dfRaw, y_train)):
        trn_df1 = x_train[trn_index]
        val_df1 = x_train[val_index]

        trn_dfRaw1 = trn_dfRaw[trn_index]
        val_dfRaw1 = trn_dfRaw[val_index]

        trn_y1 = y_train[trn_index]
        val_y1 = y_train[val_index]
        break


    model.fit([trn_dfRaw1, trn_df1], trn_y1, batch_size=128, epochs=150, verbose=1,
              callbacks=[lr_scheduler, early_stopping],
              validation_data=([val_dfRaw1, val_df1], val_y1))

    pre = model.evaluate([test_Raw, x_test], y_test, batch_size=128, verbose=2)
    print('test_loss:', pre[0], '- test_acc:', pre[1])
    y_test1 = [np.argmax(item) for item in y_test]  # 将onehot编码转成一般编码
    # for item in y_test:
    y_pred = model.predict([test_Raw, x_test])
    aa = [np.argmax(item) for item in y_test]  # 将onehot编码转成一般编码
    # # TCN(x_train,y_train,x_test,y_test)
    y_pred1 = [np.argmax(item) for item in y_pred]  # 将onehot编码转成一般编码

    df_metrics = summayResults(y_test1, y_pred1, np.array([]))

    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_test222, y_pred[:, -1])
    print(auc)

    df_metrics['auc'] = auc

    df_metrics['y_pred'] = y_pred
    df_metrics['val_y'] = y_test222

    return df_metrics


def setEnvSeedNMT(seed):
    # 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
    os.environ['PYTHONHASHSEED'] = str(seed)
    # 2. Set the `python` built-in pseudo-random generator at a fixed value
    random.seed(seed)
    # 3. Set the `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed)

    import tensorflow as tf
    # 4. Set the `tensorflow` pseudo-random generator at a fixed value
    tf.random.set_seed(seed)
    # 5. Configure a new global `tensorflow` session
    config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
    tf.compat.v1.set_random_seed(seed)

    # 控制 TensorFlow 的确定性（需要 tf 2.1+）
    os.environ['TF_DETERMINISTIC_OPS'] = '1'


if __name__ == '__main__':
    setEnvSeedNMT(43)
    allowTestTasks = [
        # "HUSM",
        # "MODMA",
        "EDRA",
    ]

    testResultList = []
    testResultList.append([''])
    resultDict = {}
    for task in allowTestTasks:
        if task == 'EDRA':
            train_xRaw, train_y = loadEDRAFeaturesDatasetSource()
            train_xRaw = butter_bandpass_filter(train_xRaw, 0.5, 50, 500, order=4, axis=1)

        taskStr = str(task)
        resultDict[taskStr] = {}
        row = [task]

        mat_ = task + "_Entropy_Features.mat"
        from scipy.io import loadmat

        mat = loadmat(mat_)
        distribution_entropy = mat['distribution_entropy']
        entropy_permutation = mat['entropy_permutation']
        sample_entropy = mat['sample_entropy']

        train_x = np.array([entropy_permutation, sample_entropy, distribution_entropy])
        train_x = OWA_generic(train_x, a=0.1, b=1, axis=0, keepdims=False)

        print(train_x.shape)

        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        row = []
        resVal1 = []
        for i, (trn_index, val_index) in enumerate(kf.split(train_x, train_y)):
            print(">>>>>\n\n")
            trn_df = train_x[trn_index]
            val_df = train_x[val_index]

            trn_dfRaw = train_xRaw[trn_index]
            val_dfRaw = train_xRaw[val_index]

            trn_y = train_y[trn_index]
            val_y = train_y[val_index]

            res = doEDRATest(trn_df, trn_y, val_df, val_y, trn_dfRaw, val_dfRaw, task, i)
            resVal1.append(res)

        resVal11 = []
        for item in resVal1:
            accuracy = item['accuracy'] * 100
            sensitivity = item['sensitivity'] * 100
            specificity = item['specificity'] * 100
            g_mean = item['g_mean'] * 100
            f1_score = item['f1_score'] * 100
            auc = item['auc'] * 100

            resVal11.append([accuracy, sensitivity, specificity, f1_score, auc])

        resVal11 = np.array(resVal11)

        print(np.mean(resVal11, axis=0))
        print(np.std(resVal11, axis=0))

        resVal1 = np.array(resVal11)

        row.append(str(np.mean(resVal1[:, 0], axis=0))[:5] + '$\pm$' + str(np.std(resVal1[:, 0], axis=0))[:4])
        row.append(str(np.mean(resVal1[:, 1], axis=0))[:5] + '$\pm$' + str(np.std(resVal1[:, 1], axis=0))[:4])
        row.append(str(np.mean(resVal1[:, 2], axis=0))[:5] + '$\pm$' + str(np.std(resVal1[:, 2], axis=0))[:4])
        row.append(str(np.mean(resVal1[:, 3], axis=0))[:5] + '$\pm$' + str(np.std(resVal1[:, 3], axis=0))[:4])
        row.append(str(np.mean(resVal1[:, 4], axis=0))[:5] + '$\pm$' + str(np.std(resVal1[:, 4], axis=0))[:4])

        testResultList.append(row)

    # 手动格式化输出每个元素
    for row in testResultList:
        print(" | ".join([f"{elem:10}" for elem in row]))  #

    print(testResultList)
