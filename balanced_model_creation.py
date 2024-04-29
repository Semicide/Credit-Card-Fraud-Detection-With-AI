import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.models import Sequential
from matplotlib import pyplot as plt
from sklearn import preprocessing, model_selection
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from warnings import filterwarnings
filterwarnings(action='ignore')

pd.set_option("display.width",500)
pd.set_option("display.max.columns",50)
pd.set_option("display.max_rows",50)

df = pd.read_csv("card_transdata.csv")
df.drop_duplicates(inplace=True)


# Eksik değerlere sahip sütunları kontrol etme
print(df.isnull().sum())

fraud_1_count = df[df["fraud"] == 1].shape[0] # fraud == 1 olanların sayısını bulma

# fraud == 1 olan verilere eşit miktarda rastgele fraud == 0 olan verileri seçme
fraud_0_data = df[df["fraud"] == 0].sample(n=fraud_1_count, random_state=42)

fraud_1_data = df[df["fraud"] == 1] # fraud == 1 olan verileri ayrı bir DataFrame'e dönüştürme

balanced_data = pd.concat([fraud_0_data, fraud_1_data]) # Hem fraud == 0 hem de fraud == 1 olan verileri birleştirme

print(balanced_data["fraud"].value_counts()) # Sonuçları kontrol etme.


X = np.array(balanced_data.drop(columns = "fraud"))
y = np.array(balanced_data["fraud"])



X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.4, random_state = 123, shuffle = True)


# y_train içindeki fraud (1) ve non-fraud (0) değerlerinin sayısını bulma
y_train_fraud_count = np.sum(y_train == 1)
y_train_non_fraud_count = np.sum(y_train == 0)

# y_test içindeki fraud (1) ve non-fraud (0) değerlerinin sayısını bulma
y_test_fraud_count = np.sum(y_test == 1)
y_test_non_fraud_count = np.sum(y_test == 0)



# Sonuçları yazdırma
print("Eğitim verisi için fraud (1) sayısı:", y_train_fraud_count)
print("Eğitim verisi için non-fraud (0) sayısı:", y_train_non_fraud_count)
print("Test verisi için fraud (1) sayısı:", y_test_fraud_count)
print("Test verisi için non-fraud (0) sayısı:", y_test_non_fraud_count)

# Eğitim verisi için yüzdelik oranları hesaplama
y_train_fraud_percent = (y_train_fraud_count / len(y_train)) * 100
y_train_non_fraud_percent = (y_train_non_fraud_count / len(y_train)) * 100

# Test verisi için yüzdelik oranları hesaplama
y_test_fraud_percent = (y_test_fraud_count / len(y_test)) * 100
y_test_non_fraud_percent = (y_test_non_fraud_count / len(y_test)) * 100

# Sonuçları yazdırma
print("Eğitim verisi için fraud (1) yüzdesi: {:.2f}%".format(y_train_fraud_percent))
print("Eğitim verisi için non-fraud (0) yüzdesi: {:.2f}%".format(y_train_non_fraud_percent))
print("Test verisi için fraud (1) yüzdesi: {:.2f}%".format(y_test_fraud_percent))
print("Test verisi için non-fraud (0) yüzdesi: {:.2f}%".format(y_test_non_fraud_percent))


"""
# Data scaling to produce good results
scale = MinMaxScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.transform(X_test)
"""

for col in df.columns:
    print(df.groupby("fraud")[col].mean(),"\n")

for col in df.columns:
    print(str(col),df[col].nunique(),"values :", df[col].unique())


model = Sequential()
model.add(Dense(64, activation="relu", input_dim=(X_train.shape[1])))
model.add(Dense(32, activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(4, activation="relu"))
model.add(Dense(2, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.compile(optimizer= "Adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()


history = model.fit(X_train,
                    y_train,
                    epochs=10,
                    validation_data=(X_test, y_test),
                    callbacks=[EarlyStopping(patience=5, verbose=1)])

model.save('balanced_model.keras')

# Plotting accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Epoch'a göre value_loss ve value_accuracy değişimini gösterme
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Değişimi')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Değişimi')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


#korelasyon analizi
"""
def high_correlated_cols (df, plot= False, corr_th= 0.1):
    corr = df.corr()
    corr_matrix = corr.abs()
    upper_triangle_matrix = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]

    if plot :
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize':(15,15)})
        sns.heatmap(corr, cmap="magma")
        plt.show()

high_correlated_cols(df,plot=True)
"""




# Belirlenen özelliklerin histogramlarını gösterme
selected_features = ['ratio_to_median_purchase_price']

for col in selected_features:
    plt.figure(figsize=(8, 6))
    plt.hist(balanced_data[col], bins=30, range=(0, 50), color='blue', alpha=0.7)
    plt.title('Histogram of {}'.format(col))
    plt.xlabel('Değerler')
    plt.ylabel('Frekans')
    plt.grid(True)
    plt.show()

# 4968.31547687355 gibi bir aykırı değeri var. (Bunu silebiliriz.)
selected_feature = 'ratio_to_median_purchase_price'

plt.figure(figsize=(8, 6))
sns.boxplot(x=balanced_data[selected_feature], color='blue')
plt.title('Boxplot of {}'.format(selected_feature))
plt.xlabel('Değerler')
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(x=balanced_data["distance_from_last_transaction"], color='blue')
plt.title('Boxplot of {}'.format(selected_feature))
plt.xlabel('Değerler')
plt.grid(True)
plt.show()


plt.figure(figsize=(8, 6))
sns.boxplot(x=balanced_data["distance_from_home"], color='blue')
plt.title('Boxplot of {}'.format(selected_feature))
plt.xlabel('Değerler')
plt.grid(True)
plt.show()