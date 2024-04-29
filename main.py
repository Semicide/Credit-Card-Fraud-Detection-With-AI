import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')  # Or any other backend you prefer
df = pd.read_csv("card_transdata.csv")
df=df[:50000]
df.drop_duplicates(inplace=True)

scaler = MinMaxScaler()
columns_to_scale = ['distance_from_home','distance_from_last_transaction', 'ratio_to_median_purchase_price']
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

X = df.drop('fraud', axis=1)
y = df['fraud']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size = 0.4, random_state = 12)

mean = np.mean(X_train)
std = np.std(X_train)


X_train -= mean
X_train /= std

X_test -= mean
X_test /= std

from keras import models, layers
from keras.callbacks import EarlyStopping

model = models.Sequential()

model.add(layers.Dense(10, input_shape=(X_train.shape[1],), activation='relu'))
model.add(layers.Dense(8, activation='relu'))
model.add(layers.Dense(6, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

from sklearn.metrics import classification_report
model.compile(optimizer='rmsprop', loss='binary_crossentropy',metrics=['accuracy'])

from sklearn.utils.class_weight import compute_class_weight
class_weight = compute_class_weight(class_weight="balanced",classes=np.unique(y_train),y=y_train)

weights = {
    0:0.5,
    1:300
}

callback = EarlyStopping(monitor='loss', patience=3)

history = model.fit(X_train,
                    y_train,
                    epochs=100,
                    batch_size=64,
                    validation_data=(X_test, y_test),
                    class_weight = weights,
                   callbacks=[callback])

print(history.history.keys())


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


val_predictions = model.predict(X_test)
print(val_predictions)
preds = np.around(val_predictions)

from sklearn.metrics import classification_report
print(classification_report(y_test, preds))

