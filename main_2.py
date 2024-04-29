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
df = df[0:50000]
df.drop_duplicates(inplace=True)

print(df[df["fraud"]==1])
X = np.array(df.drop(columns = "fraud"))
y = np.array(df["fraud"])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.4, random_state = 123, shuffle = True)

# Data scaling to produce good results
scale = MinMaxScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.transform(X_test)

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

model.save('my_model.keras')

# Plotting accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.title('Model Accuracy')
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