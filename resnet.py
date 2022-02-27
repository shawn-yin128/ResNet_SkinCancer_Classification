"""
0. import packages
"""
import matplotlib.pyplot as plt
import seaborn as sns
from keras.utils.np_utils import to_categorical
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
from keras.layers import Input, Add, Dense, Activation, BatchNormalization, Flatten, Conv2D, MaxPool2D, \
    AveragePooling2D, Dropout
from keras.models import Model
from keras.metrics import Precision, Recall
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

epochs = 3
batch_size = 64
"""
1. data preparation
"""
skin_df = pd.read_csv("D:/py_project/deep_learning_project/py_cnn_skin_cancer_paper/HAM10000_metadata.csv")
print("successfully load csv!\n")
# get image path
image_id_path_dict = {}
for i in range(24306, 34321):
    image_id_path_dict["ISIC_00{}".format(i)] = "D:/py_project/deep_learning_project/py_cnn_skin_cancer_paper/image/ISIC_00{}.jpg".format(i)
print("image id path generated!\n")
# rename lesions type
lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}
print("type dict generated!\n")
# prepare dataframe column
skin_df['path'] = skin_df['image_id'].map(image_id_path_dict.get)
skin_df["cell_type"] = skin_df["dx"].map(lesion_type_dict.get)
skin_df["cell_type_idx"] = pd.Categorical(skin_df["cell_type"]).codes
print("dataframe column ready!\n")
# get image matrix
skin_df['image'] = skin_df['path'].map(lambda x: np.asarray(Image.open(x).resize((300, 300))))
print("image ready with shape and counts: {}\n".format(skin_df['image'].map(lambda x: x.shape).value_counts()))
# get target dataframe
target = skin_df["cell_type_idx"]
features = skin_df.drop(columns=["cell_type_idx"], axis=1)
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.20, random_state=1)
X_train = np.asarray(x_train['image'].tolist())
X_test = np.asarray(x_test['image'].tolist())
Y_train = to_categorical(y_train, num_classes=7)
Y_test = to_categorical(y_test, num_classes=7)
print("train test data ready!\n")

"""
2. data preprocessing
"""
# normalization
X_train_mean = np.mean(X_train)
X_train_std = np.std(X_train)
X_test_mean = np.mean(X_test)
X_test_std = np.std(X_test)
X_train = (X_train - X_train_mean) / X_train_std
X_test = (X_test - X_test_mean) / X_test_std
print("normalization done!\n")
# train validation split and reshape
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=1)
X_train = X_train.reshape(X_train.shape[0], *(300, 300, 3))
X_test = X_test.reshape(X_test.shape[0], *(300, 300, 3))
X_val = X_val.reshape(X_val.shape[0], *(300, 300, 3))
print("data splited and reshaped!\n")

"""
3. model define
"""


# define identity block
def iden_block(X, f):
    X_res = X

    X = Conv2D(f, kernel_size=(3, 3), strides=1, padding="same")(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation("relu")(X)

    X = Conv2D(f, kernel_size=(3, 3), strides=1, padding="same")(X)
    X = BatchNormalization(axis=3)(X)

    X = Add()([X, X_res])
    X = Activation("relu")(X)

    return X


# define convolutional block
def con_block(X, f):
    X_res = X

    X = Conv2D(f, kernel_size=(3, 3), strides=1, padding="same")(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation("relu")(X)

    X = Conv2D(f, kernel_size=(3, 3), strides=2, padding="same")(X)
    X = BatchNormalization(axis=3)(X)

    X_res = Conv2D(f, kernel_size=(3, 3), strides=2, padding="same")(X_res)
    X_res = BatchNormalization(axis=3)(X_res)

    X = Add()([X, X_res])
    X = Activation("relu")(X)
    X = Dropout(0.5)(X)

    return X


# define bottleneck block
def bot_block(X, f):
    X_res = X

    X = Conv2D(f / 2, kernel_size=(1, 1), strides=1, padding="same")(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation("relu")(X)

    X = Conv2D(f / 2, kernel_size=(3, 3), strides=1, padding="same")(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation("relu")(X)

    X = Conv2D(f, kernel_size=(1, 1), strides=1, padding="same")(X)
    X = BatchNormalization(axis=3)(X)

    X = Add()([X, X_res])
    X = Activation("relu")(X)

    return X


# define resnet framework
def resnet(input_size, clas):
    X_input = Input(input_size)

    X = Conv2D(64, (7, 7), strides=2, padding="same")(X_input)
    X = BatchNormalization(axis=3)(X)
    X = Activation("relu")(X)
    X = MaxPool2D((2, 2))(X)

    X = iden_block(X, 64)
    X = iden_block(X, 64)
    X = iden_block(X, 64)
    X = iden_block(X, 64)
    X = con_block(X, 128)

    X = iden_block(X, 128)
    X = iden_block(X, 128)
    X = iden_block(X, 128)
    X = iden_block(X, 128)
    X = con_block(X, 256)

    X = iden_block(X, 256)
    X = iden_block(X, 256)
    X = iden_block(X, 256)
    X = iden_block(X, 256)
    X = con_block(X, 512)

    X = iden_block(X, 512)
    X = iden_block(X, 512)
    X = iden_block(X, 512)
    X = iden_block(X, 512)
    X = con_block(X, 1024)

    X = iden_block(X, 1024)
    X = iden_block(X, 1024)
    X = iden_block(X, 1024)

    X = AveragePooling2D((2, 2), padding="same")(X)

    X = Flatten()(X)
    X = Dense(clas, activation="softmax")(X)

    model = Model(inputs=X_input, outputs=X, name="resnet")

    return model


# define bottleneck resnet framework
def bot_resnet(input_size, clas):
    X_input = Input(input_size)

    X = Conv2D(64, (7, 7), strides=2, padding="same")(X_input)
    X = BatchNormalization(axis=3)(X)
    X = Activation("relu")(X)
    X = MaxPool2D((2, 2))(X)

    X = bot_block(X, 64)
    X = bot_block(X, 64)
    X = bot_block(X, 64)
    X = con_block(X, 128)

    X = bot_block(X, 128)
    X = bot_block(X, 128)
    X = bot_block(X, 128)
    X = con_block(X, 256)

    X = bot_block(X, 256)
    X = bot_block(X, 256)
    X = bot_block(X, 256)
    X = con_block(X, 512)

    X = bot_block(X, 512)
    X = bot_block(X, 512)
    X = bot_block(X, 512)
    X = bot_block(X, 512)
    X = con_block(X, 1024)

    X = AveragePooling2D((2, 2), padding="same")(X)

    X = Flatten()(X)
    X = Dense(clas, activation="softmax")(X)

    model = Model(inputs=X_input, outputs=X, name="resnet")

    return model


"""
4. compile setup
"""
model = resnet(input_size=(300, 300, 3), clas=7)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=["accuracy", Precision(), Recall()])
learning_rate_reduction = ReduceLROnPlateau(monitor='accuracy',
                                            patience=2,
                                            verbose=1,
                                            factor=0.3,
                                            min_lr=0.00000001)
print("model compile setup ready!\n")

"""
5. train model
"""
# train process
fit_model = model.fit(X_train, Y_train, batch_size=batch_size,
                      epochs=epochs, callbacks=learning_rate_reduction, validation_data=(X_val, Y_val))
print("model trained!\n")
# train process visualization
fig, ax = plt.subplots(2, 1)
ax[0].plot(fit_model.history['loss'], label="Training loss")
ax[0].plot(fit_model.history['val_loss'], label="Validation loss")
legend = ax[0].legend(loc='best', shadow=True)
ax[1].plot(fit_model.history['accuracy'], label="Training accuracy")
ax[1].plot(fit_model.history['val_accuracy'], label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)
plt.show()
print("loss and accuracy curve drew!\n")

"""
6. test model
"""
# test set
scores = model.evaluate(X_test, Y_test, verbose=1)
print("model one test data has a loss of {}".format(scores[0]) + ", and accuracy of {}\n".format(scores[1]))
# confusion matrix
labels_names = ["AK", "BCC", "BKL", "DF", "NV", "MEL", "VASC"]
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(Y_test, axis=1)
conf_mat2 = confusion_matrix(y_true, y_pred_classes)
f, ax = plt.subplots(figsize=(10, 9))
sns.heatmap(conf_mat2, annot=True, fmt=".0f")
ax.set_xticklabels(labels_names)
ax.set_yticklabels(labels_names)
plt.show()
print("confusion matrix drew!\n")
print("basic resnet done!\n")

"""
7. compile setup - bot
"""
model = bot_resnet(input_size=(300, 300, 3), clas=7)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=["accuracy", Precision(), Recall()])
learning_rate_reduction = ReduceLROnPlateau(monitor='accuracy',
                                            patience=2,
                                            verbose=1,
                                            factor=0.3,
                                            min_lr=0.00000001)
print("model - bot compile setup ready!\n")

"""
8. train model - bot
"""
# train process
fit_model = model.fit(X_train, Y_train, batch_size=batch_size,
                      epochs=epochs, callbacks=learning_rate_reduction, validation_data=(X_val, Y_val))
print("model - bot trained!\n")
# train process visualization
fig, ax = plt.subplots(2, 1)
ax[0].plot(fit_model.history['loss'], label="Training loss")
ax[0].plot(fit_model.history['val_loss'], label="Validation loss")
legend = ax[0].legend(loc='best', shadow=True)
ax[1].plot(fit_model.history['accuracy'], label="Training accuracy")
ax[1].plot(fit_model.history['val_accuracy'], label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)
plt.show()
print("loss and accuracy curve drew - bot!\n")

"""
6. test model - bot
"""
# test set
scores = model.evaluate(X_test, Y_test, verbose=1)
print("model - bot one test data has a loss of {}".format(scores[0]) + ", and accuracy of {}\n".format(scores[1]))
# confusion matrix
labels_names = ["AK", "BCC", "BKL", "DF", "NV", "MEL", "VASC"]
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(Y_test, axis=1)
conf_mat2 = confusion_matrix(y_true, y_pred_classes)
f, ax = plt.subplots(figsize=(10, 9))
sns.heatmap(conf_mat2, annot=True, fmt=".0f")
ax.set_xticklabels(labels_names)
ax.set_yticklabels(labels_names)
plt.show()
print("confusion matrix drew - bot!\n")
print("project done!\n")