#!/usr/bin/env python
# coding: utf-8

# In[2]:


print("Prediksi Harga Mobil Bekas dengan Algoritma KNN")


# In[9]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

data = pd.read_csv("https://raw.githubusercontent.com/imamkoo/KNN_algorithm_prediction_car_price/main/bmw.csv")
data.head()


# In[8]:


print(data.shape)
print(data.describe)


# In[13]:


#Bersihkan data yang tidak lengkap
data_drop = data.dropna(axis=0)
data_drop.describe()


# In[14]:


print(data.dtypes)


# In[15]:


#input output data yang menggunakan tipe data integer dan float
features = ['year', 'mileage', 'tax', 'mpg', 'engineSize']
x = data_drop[features]
y = data_drop['price']
x.shape, y.shape


# In[16]:


#Membagi data untuk training dan validasi (test)
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(x, y, random_state = 70)

test_y.shape


# In[17]:


#Machine learning model dengan algoritma K-Nearest Neighbors
from sklearn.neighbors import KNeighborsRegressor as KNN_Reg
from sklearn.metrics import mean_squared_error as mse

model = KNN_Reg(n_neighbors = 80)

# training the model:
model.fit(train_x, train_y)
acc1 = model.score(test_x, test_y)

# test for prediction
test_predict = model.predict(test_x)
score = mse(test_predict, test_y)
print(' MSE: ', score, '\n', 'Accuracy: ', acc1)  


# In[18]:


#Elbow method untuk menentukan nilak K terbaik
def Elbow(K):
  #initiating an empy list
  test_mse =[]

  #train model for every value of K
  for i in K:
    model = KNN_Reg(n_neighbors=i)
    model.fit(train_x, train_y)
    tmp = model.predict(test_x)
    tmp = mse(tmp, test_y)
    test_mse.append(tmp)
  
  return test_mse


# In[19]:


#Menampilkan grafik nilai k berdasarkan MSE
K = range(80, 100)
test = Elbow(K)

#plotting
plt.plot(K,test)
plt.xlabel('K Neighbors')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Elbow Curve for Test')


# In[29]:


#Melakukan koreksi nilai K dengan menggunakan nilai K terbaik berdasarkan hasil dari Elbow method
#Nilai K terbaik adalah nilai K yang menghasilkan MSE minimum, dalam kasus ini k=86
new_model = KNN_Reg(n_neighbors=86)

# Train model
new_model.fit(train_x, train_y)
acc2 = new_model.score(test_x, test_y)

# Prediction test
print(' Accuracy of new model (%):', acc2*100, '\n', 'Accuracy of old model (%):', acc1*100, '\n Improvement (%):', (acc2-acc1)*100)


# In[39]:


#Percobaan 1
#Data mobil bekas: year=2019, mileage=5000, tax=145, mpg=30.2, engineSize=2
#Format input data:
#data_mobil_bekas = ['year', 'mileage', 'tax', 'mpg', 'engineSize']

data_mobil_bekas = np.array([[2019,5000,145,30.2,2]])
prediction_old = model.predict(data_mobil_bekas)
prediction_new = new_model.predict(data_mobil_bekas)
print(f"Hasil Prediksi harga rumah dengan old model: £{prediction_old}, atau jika dirupiahkan yaitu: Rp{prediction_old* 16259*1e-6} Juta.")
print(f"Hasil Prediksi harga rumah dengan new model: £{prediction_new}. atau jika dirupiahkan yaitu: Rp{prediction_new* 16259*1e-6} Juta.")


# In[ ]:




