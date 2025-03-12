
pip install django djangorestframework tensorflow pandas numpy scikit-learn pickle-mixin matplotlib seaborn
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
df = pd.read_csv("brain_stroke.csv")
df.head()
df.tail()
pd=df.isnull().sum()
print(pd)
label_encoders = {}
for col in ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(x='stroke',data=df)
plt.title("Distribution of stroke")
df.columns
df_cat=df[['gender', 'hypertension', 'heart_disease', 'ever_married','work_type', 'Residence_type', 
                 'smoking_status', 'stroke']]
for i in df_cat.columns:
    print(df_cat[i].unique())
for i in df_cat.columns:
    print(df_cat[i].value_counts())
for i in df_num.columns:
    plt.figure(figsize = (15,6))
    sns.histplot(df_num[i], palette = 'hls')
    plt.xticks(rotation = 90)
    plt.show()
x = df.drop(columns=['stroke'])
y = df['stroke']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
from sklearn.metrics import classification_report , confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
model = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(x_train.shape[1],)),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=20, batch_size=16)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
model = Sequential([
    Dense(32, activation='relu', input_shape=(x_train.shape[1],)),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification (stroke or no stroke)
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) 
model.fit(x_train, y_train, epochs=50, batch_size=8, validation_data=(x_test, y_test))
model.save("stroke_ann_model.h5")
import joblib
joblib.dump("scaler","scaler.pkl")


