#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


# In[2]:


df1 = pd.read_csv('TypeOfVehicleStateData.csv')
df1.head()


# In[3]:


df2 = pd.read_excel('ChargingStations.xlsx', sheet_name='Table 4', header=1)
df2.head()


# In[5]:


df3 = pd.read_excel('ElectricVehiclesInIndia.xlsx')
df3.head()


# ## Analysis of the data

# In[5]:


# checking the shape (# of rows and columns) of the datasets
print('DF1 Shape: ', df1.shape)
print('DF2 Shape: ', df2.shape)
print('DF3 Shape: ', df3.shape)


# In[6]:


# checking the info (columns, datatypes, nulls) of the datasets
print()
print(df1.info())
print()
print(df2.info())
print()
print(df3.info())


# In[7]:


# getting a statistical summary of the datasets
d1 = df1.describe()
d2 = df2.describe()
d3 = df3.describe()
display('<<< DATASET 1 >>>', d1, '<<< DATASET 2 >>>', d2, '<<< DATASET 3 >>>', d3)


# In[12]:


# 2 wheelers data visualization from dataset 1
plt.figure(figsize=(6, 6))
sns.barplot(data=df1, y=df1['Region'].sort_values(ascending=True), x='2W', palette='dark:salmon_r')
plt.ylabel('State', fontsize=14, family='serif')
plt.xlabel('Number of EV: 2 Wheelers', family='serif', fontsize=14, labelpad=10)
plt.xticks(family='serif')
plt.yticks(family='serif')
plt.title(label='Statewise Electric Vehicles (2 Wheelers) in India', weight=200, family='serif', size=15, pad=12)
plt.show()


# In[11]:


# 3 wheelers data visualization from dataset 1
plt.figure(figsize=(6, 6))
sns.barplot(data=df1, y=df1['Region'].sort_values(ascending=True), x='3W', palette='dark:salmon_r')
plt.ylabel('State', fontsize=14, family='serif')
plt.xlabel('Number of EV: 3 Wheelers', family='serif', fontsize=14, labelpad=10)
plt.xticks(family='serif')
plt.yticks(family='serif')
plt.title(label='Statewise Electric Vehicles (3 Wheelers) in India', weight=200, family='serif', size=15, pad=12)
plt.show()


# In[13]:


# 4 wheelers data visualization from dataset 1
plt.figure(figsize=(6, 6))
sns.barplot(data=df1, y=df1['Region'].sort_values(ascending=True), x='4W', palette='dark:salmon_r')
plt.ylabel('State', fontsize=14, family='serif')
plt.xlabel('Number of EV: 4 Wheelers', family='serif', fontsize=14, labelpad=10)
plt.xticks(family='serif')
plt.yticks(family='serif')
plt.title(label='Statewise Electric Vehicles (4 Wheelers) in India', weight=200, family='serif', size=15, pad=12)
plt.show()


# In[14]:


# charging stations sanctioned visualization from dataset 1
plt.figure(figsize=(6, 6))
sns.barplot(data=df1, y=df1['Region'].sort_values(ascending=True), x='Chargers', palette='viridis')
plt.ylabel('State', fontsize=14, family='serif')
plt.xlabel('Number of Charging Stations', family='serif', fontsize=14, labelpad=10)
plt.xticks(family='serif')
plt.yticks(family='serif')
plt.title(label='Number of Charging Stations Sanctioned in India', weight=200, family='serif', size=15, pad=12)
plt.show()


# In[16]:


# brand-wise count of EV models
sns.catplot(data=df3, x='Brand', kind='count', palette='crest', height=6, aspect=2)
sns.despine(right=False, top=False)
plt.tick_params(axis='x', rotation=40)
plt.xlabel('Brand',family='serif', size=12)
plt.ylabel('Count', family='serif', size=12)
plt.xticks(family='serif')
plt.yticks(family='serif')
plt.title('Number of EV Models Manufactured by a Brand', family='serif', size=15)
plt.show()


# In[18]:


# analysis of different segments of EVs from dataset 3
x = df3['Segment'].value_counts().plot.pie(radius=2, cmap='crest', startangle=0, textprops=dict(family='serif'), pctdistance=.5)
plt.pie(x=[1], radius=1.2, colors='white')
plt.title(label='Electric Vehicles of Different Segments in India', family='serif', size=15, pad=100)
plt.ylabel('')
plt.show()


# In[19]:


# brand-wise analysis of the number of seats
sns.catplot(kind='bar', data=df3, x='Brand', y='Seats', palette='crest', ci=None, height=6, aspect=2)
sns.despine(right=False, top=False)
plt.tick_params(axis='x', rotation=40)
plt.xlabel('Brand',family='serif', size=12)
plt.ylabel('Number of Seats', family='serif', size=12)
plt.xticks(family='serif')
plt.yticks(family='serif')
plt.title('Brand-wise Analysis of the Number of Seats', family='serif', size=15);


# In[20]:


# plug types visualization from dataset 3
df3['PlugType'].value_counts().sort_values(ascending=True).plot.barh()
plt.xlabel('Count', family='serif', size=12)
plt.ylabel('Plug Type', family='serif', size=12)
plt.xticks(family='serif')
plt.yticks(family='serif')
plt.title('Available Plug Types of EVs in India', family='serif', size=15)
plt.show()


# In[26]:


# plotting the price from dataset 3
plt.plot(df3['PriceEuro'], color='black')
plt.xlabel('Number of Samples', family='serif', size=12)
plt.ylabel('Price', family='serif', size=12)
plt.title('Price Comparison', family='serif', size=15, pad=12);


# In[22]:


# accleration visualization from dataset 3
plt.figure(figsize=(6, 8))
sns.barplot(data=df3, y='Brand', x='AccelSec', ci=None, palette='viridis')
plt.xticks(family='serif')
plt.yticks(family='serif')
plt.xlabel('Accleration', family='serif', size=12)
plt.ylabel('Brand', family='serif', size=12)
plt.title(label='Accleration of EVs in India', family='serif', size=15, pad=12)
plt.show()


# In[23]:


# speed visualization from dataset 3
plt.figure(figsize=(6, 8))
sns.barplot(data=df3, x='TopSpeed_KmH', y='Brand', ci=None, palette='viridis')
plt.xticks(family='serif')
plt.yticks(family='serif')
plt.xlabel('Max Speed', family='serif', size=12)
plt.ylabel('Brand', family='serif', size=12)
plt.title(label='Brand-wise Speed Comparison of EVs in India', family='serif', size=15, pad=12)
plt.show()


# In[25]:


# brand-wise analysis of the range parameter
sns.catplot(kind='bar', data=df3, x='Brand', y='Range_Km', palette='crest', ci=None, height=6, aspect=2)
sns.despine(right=False, top=False)
plt.tick_params(axis='x', rotation=40)
plt.xlabel('Brand',family='serif', size=12)
plt.ylabel('Range', family='serif', size=12)
plt.xticks(family='serif')
plt.yticks(family='serif')
plt.title('Brand-wise Analysis of the Range Parameter', family='serif', size=15);


# Model Building Using K-Means Clusteing

# In[28]:


# encoding the categorical features

# PowerTrain feature
df3['PowerTrain'].replace(to_replace=['RWD','FWD','AWD'],value=[0, 1, 2],inplace=True)

# RapidCharge feature
df3['RapidCharge'].replace(to_replace=['No','Yes'],value=[0, 1],inplace=True)

# selecting features for building a model
X = df3[['AccelSec','TopSpeed_KmH','Efficiency_WhKm','FastCharge_KmH', 'Range_Km', 'RapidCharge', 'Seats', 'PriceEuro','PowerTrain']]

# feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# applying Principle Component Analysis (PCA)
pca = PCA(n_components=9)
X_pca = pca.fit_transform(X_scaled)
df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9'])
df_pca.head()


# In[29]:


# plotting the results of Elbow

wcss = []

for i in range(1, 11):
  kmean = KMeans(n_clusters=i, init='k-means++', random_state=90)
  kmean.fit(X_pca)
  wcss.append(kmean.inertia_)

plt.figure(figsize=(8,6))
plt.title('Plot of the Elbow Method', size=15, family='serif')
plt.plot(range(1, 11), wcss)
plt.xticks(range(1, 11), family='serif')
plt.yticks(family='serif')
plt.xlabel('Number of Custers (K)', family='serif')
plt.ylabel('WCSS', family='serif')
plt.grid()
plt.tick_params(axis='both', direction='inout', length=6, color='purple', grid_color='lightgray', grid_linestyle='--')
plt.show()


# In[30]:


# training the model using k=4 as rendered by the above plot
kmean = KMeans(n_clusters=4, init='k-means++', random_state=90)
kmean.fit(X_pca)


# In[31]:


# check the labels assigned to each data point
print(kmean.labels_)


# In[32]:


# check the size of clusters
pd.Series(kmean.labels_).value_counts()


# In[34]:


# adding a new feature of cluster labels to the dataset 3
df3['clusters'] = kmean.labels_

# visualizing clusters
plt.figure(figsize=(7,5))
sns.scatterplot(data=df_pca, x='PC1', y='PC9', s=70, hue=kmean.labels_, palette='viridis', zorder=2, alpha=.9)
plt.scatter(x=kmean.cluster_centers_[:,0], y=kmean.cluster_centers_[:,1], c="r", s=80, label="centroids")
plt.xlabel('PC1', family='serif', size=12)
plt.ylabel('PC9', family='serif', size=12)
plt.xticks(family='serif')
plt.yticks(family='serif')
plt.grid()
plt.tick_params(grid_color='lightgray', grid_linestyle='--', zorder=1)
plt.legend(title='Labels', fancybox=True, shadow=True)
plt.title('K-Means Clustering Results', family='serif', size=15)
plt.show()


# In[3]:


df4=pd.read_csv("EV_Customer_Feedback.csv")
df4.head()


# In[6]:


df3['INR']=df3['PriceEuro']*85.06
df3.head()


# In[7]:


plt.rcParams['figure.figsize']=(18,12)
plt.subplot(221)
sns.countplot(data=df4,x='rating',hue='Used it for')
plt.subplot(222)
sns.countplot(data=df4,x='rating',hue='Value for Money')
plt.subplot(223)
sns.countplot(data=df4,x='rating',hue='Visual Appeal')
plt.subplot(224)
sns.countplot(data=df4,x='rating',hue='Comfort')


# In[9]:


df4[['Value for Money','Visual Appeal','Comfort','rating']].corr().style.background_gradient(cmap='RdBu_r')


# In[10]:


plt.rcParams['figure.figsize']=(18,12)
LIST=[i for i in range(103)]
plt.subplot(321)
plt.scatter(LIST,df3['AccelSec'])
plt.ylabel("Acceleration(Sec)")
plt.subplot(322)
plt.scatter(LIST,df3['TopSpeed_KmH'])
plt.ylabel("Top Speed(KmH)")
plt.subplot(323)
plt.scatter(LIST,df3['Efficiency_WhKm'])
plt.ylabel("Efficiency(WhKm)")
plt.subplot(324)
plt.scatter(LIST,df3['Range_Km'])
plt.ylabel("Range(Km)")
plt.subplot(325)
sns.countplot(data=df3,x='Seats')
plt.subplot(326)
plt.scatter(LIST,df3['INR'])
plt.ylabel("Price")


# In[ ]:




