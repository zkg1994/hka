#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# # reading files

# In[2]:


data = pd.read_csv("C:/Users/kaige/Desktop/t1.csv")


# In[3]:


data.head(5)


# In[4]:


data.shape


# # group  by date

# In[5]:


data = data.drop(['GenerationCurrent'], axis = 1)
data = data.drop(['GenerationImportEnergy'], axis = 1)
data = data.drop(['GenerationExportEnergy'], axis = 1)


# In[6]:


data.head(5)


# In[7]:


data['VNC'] = data['GridVoltage'] * data['GridCurrent']
data.head(5)


# In[8]:


type(data['Hour'])


# In[9]:


group_date = data.groupby(['Id','Date','Hour']).mean()
group_date = group_date.reset_index()
group_date.head(10)


# In[10]:


train_data = group_date[['Id','Date','Hour','VNC']]


# In[11]:


type(train_data)


# In[12]:


train_data.shape


# In[13]:


train_data.head(5)


# ### write to excel

# In[53]:


data_date = plot_date.drop(['Date'], axis = 1).values.copy()


# In[54]:


data_date


# In[16]:


type(train_data['Date'])


# # plot figure grouped by date

# In[17]:


plot_date = train_data.groupby(['Id', 'Date']).mean()
plot_date = plot_date.reset_index()
plot_date = plot_date.drop(['Hour'], axis = 1)
plot_date.shape


# In[18]:


plot_date.head(5)


# In[19]:


id_list = plot_date['Id'].drop_duplicates().tolist()
fig, ax = plt.subplots(1, 1, figsize = (30, 12))
for i in range(len(id_list)):
    rows = plot_date[plot_date['Id'] == id_list[i]]
    rows.plot('Date', 'VNC', kind = 'line',color = 'blue', ax = ax, legend = False, figsize = (30, 12), grid = True, fontsize = 30)

plt.title('Grouped By Date', fontsize = 30)
plt.xlabel('Date', fontsize = 30)
plt.ylabel('Voltage', fontsize = 30)
plt.show()


# # grouped by hour

# In[20]:


plot_hour = train_data.groupby(['Id', 'Hour']).mean()
plot_hour = plot_hour.reset_index()
plot_hour.shape


# In[21]:


plot_hour.head(5)


# In[22]:


data_hour = plot_hour.values.copy()


# # plot figure grouped by hour

# In[23]:


plot_hour.plot('Hour', 'VNC', kind = 'line', legend = False, grid = True, figsize = (30, 12), xticks = range(0,24), fontsize = 30)
plt.title('Grouped By Hour', fontsize = 30)
plt.xlabel('Hour', fontsize = 30)
plt.ylabel('VNC', fontsize = 30)
plt.show()


# # train Date, k-means

# In[71]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics


# In[72]:


date = plot_date['Date']


# In[73]:


arrs = date.values
arrs = [arrs]


# In[74]:


arrs


# In[75]:


# using one-hot-encoding
from sklearn.preprocessing import OneHotEncoder

# character: date 
encoder = OneHotEncoder()
encoder.fit(arrs)
encoder_vector = encoder.transform(arrs).toarray()


# In[76]:


clinski_harabaz_scores = []
cluster_num = range(2, 6)

for i in cluster_num:
    kmeans = KMeans(n_clusters = i, n_jobs = 4)
    data_train = np.concatenate((data_date, encoder_vector.T), axis = 1)
    predict = kmeans.fit_predict(data_train)
    clinski_harabaz_scores.append(metrics.calinski_harabaz_score(data_train, predict))
    


# In[77]:


clinski_harabaz_scores


# In[78]:


fig = plt.figure()
x = range(2,6)
plt.plot(x, clinski_harabaz_scores)
plt.xlabel('cluster numbers')
plt.xticks(range(2, 6))
plt.ylabel('clinski harabaz scores')
plt.title('Date, K-means, clinski harabz score')  # score with diff cluster numbers
max_clin = max(clinski_harabaz_scores)
max_clin_index = clinski_harabaz_scores.index(max(clinski_harabaz_scores)) + 2
plt.plot([max_clin_index], [max_clin], 'o')
plt.annotate('max score:[5,  9781.750835262113]', xy = (max_clin_index, max_clin), xytext = (20,20000),
            arrowprops = dict(facecolor = 'black', shrink = 0.05))
plt.show


# ### Cluster Numbers = 3

# In[86]:


kmeans = KMeans(n_clusters = 3, n_jobs = 4)
predict = kmeans.fit_predict(data_train)


# In[87]:


plot_date['cluster_kmeans'] = predict


# In[88]:


train_data[group_date['Id'] == 96]


# In[89]:


# id_list = group_date['Id'].drop_duplicates().tolist()
color_list = ['red','green','blue']
median_list = ['black', 'black', 'black']
fig, axes = plt.subplots(3, 1, figsize = (30, 60))
# fig.tight_layout()
# plt.subplots_adjust(wspace =0, hspace =0)

for i in range(len(id_list)):
    rows = plot_date[plot_date['Id'] == id_list[i]]
    cluster_index = int(rows['cluster_kmeans'].mode())  
    rows.plot('Date', 'VNC', kind = 'line',ax = axes[cluster_index], color= color_list[cluster_index], alpha = 0.8,legend = False, grid = True, fontsize = 30)
    axes[cluster_index].set_title('Date, K-means, cluster(pattern ' + str(cluster_index) + ")", fontsize = 30)
    axes[cluster_index].set_ylabel('Electric Work', fontsize = 30)
#     plt.show()
    
# plot median for 4 clusters
cluster_num = 3
for i in range(cluster_num):
    rows = plot_date[plot_date['cluster_kmeans'] == i]
    median = rows.groupby([ 'Date'])['VNC'].mean()
    median.plot(color = median_list[i],ax = axes[i], linewidth = 10, ls = ':')


# # train Hour, k-means

# In[90]:


clinski_harabaz_scores = []
cluster_num = range(2, 10)

for i in cluster_num:
    kmeans = KMeans(n_clusters = i, n_jobs = 4)
    predict = kmeans.fit_predict(plot_hour)
    clinski_harabaz_scores.append(metrics.calinski_harabaz_score(plot_hour, predict))


# In[91]:


clinski_harabaz_scores


# In[93]:


fig = plt.figure()
x = range(2,10)
plt.plot(x, clinski_harabaz_scores)
plt.xlabel('cluster numbers')
plt.xticks(range(2, 10))
plt.ylabel('clinski harabaz score')
plt.title('Hour, K-means, clinski harabaz score ') # scores with different cluster numbers
max_clin = max(clinski_harabaz_scores)
max_clin_index = clinski_harabaz_scores.index(max(clinski_harabaz_scores)) + 2
plt.plot([max_clin_index], [max_clin], 'o')
plt.annotate('max score:[9, 1767.93]', xy = (max_clin_index, max_clin), xytext = (20,1600),
            arrowprops = dict(facecolor = 'black', shrink = 0.05))
plt.show


# ### cluster numbers = 9 is best

# In[94]:


kmeans = KMeans(n_clusters = 3, n_jobs = 4)
predict = kmeans.fit_predict(plot_hour)


# In[95]:


plot_hour['cluster_kmeans'] = predict


# In[96]:


plot_hour.head()


# In[99]:


color_list = ['red','green','blue']
median_list = ['firebrick', 'darkgreen', 'darkblue']
cluster_num = 3

# fig, ax = plt.subplots(1, 1, figsize = (30, 12))
for i in range(cluster_num):
    rows = plot_hour[plot_hour['cluster_kmeans'] == i]
    rows.plot('Hour', 'VNC', kind = 'line', color= color_list[i],alpha = 0.8, legend = False, figsize = (30, 12), grid = True, xticks = range(0,24), fontsize = 30)
    median = rows.groupby([ 'Hour'])['VNC'].mean()
    median.plot(color = color_list[i], linewidth = 10, ls = ':')
    plt.title('Hour, K-means, cluster(pattern ' + str(i) + ")", fontsize = 30)
    plt.xlabel('Hour', fontsize = 30)
    plt.ylabel('VNC', fontsize = 30)
    plt.show()
    


# # train Date, mean shift

# In[100]:


data_hour = plot_hour.values.copy()


# In[101]:


from sklearn.cluster import MeanShift, estimate_bandwidth


# In[102]:


bandwidth = estimate_bandwidth(data_hour, quantile = 0.2, n_samples = 500)

meanShift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
predict = meanShift.fit_predict(data_train)


# In[104]:


plot_date['cluster_meanShift'] = predict


# In[105]:


plot_date['cluster_meanShift'].value_counts()


# ###### 可以看出， 只分了6类， 0， 1， 2，3， 4， 5对应类， 后面对应该类个数

# In[106]:


plot_date.head()


# In[116]:


# id_list = group_date['Id'].drop_duplicates().tolist()
color_list = ['red','green','blue','yellow','orange']
median_list = ['black', 'black', 'black','black', 'black']
fig, axes = plt.subplots(5, 1, figsize = (30, 60))
# fig.tight_layout()
# plt.subplots_adjust(wspace =0, hspace =0)

for i in range(len(id_list)):
    rows = plot_date[plot_date['Id'] == id_list[i]]
    cluster_index = int(rows['cluster_meanShift'].mode())  
    rows.plot('Date', 'VNC', kind = 'line',ax = axes[cluster_index], color= color_list[cluster_index], alpha = 0.8,legend = False, grid = True, fontsize = 30)
    axes[cluster_index].set_title('Date, mean shift, cluster(pattern ' + str(cluster_index) + ")", fontsize = 30)
#     axes[cluster_index].set_xlabel('date', fontsize = 30)
    axes[cluster_index].set_ylabel('VNC', fontsize = 30)
#     plt.show()
    
# plot median for 3 clusters
cluster_num = 5
for i in range(cluster_num):
    rows = plot_date[plot_date['cluster_meanShift'] == i]
    median = rows.groupby([ 'Date'])['VNC'].mean()
    median.plot(color = median_list[i],ax = axes[i], linewidth = 10, ls = ':')


# # train Hour, mean shift

# In[108]:


bandwidth = estimate_bandwidth(plot_hour, quantile = 0.2, n_samples = 500)

meanShift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
predict = meanShift.fit_predict(plot_hour)


# In[109]:


plot_hour['cluster_meanShift'] = predict


# In[110]:


plot_hour.head()


# In[111]:


plot_hour['cluster_meanShift'].value_counts()


# In[117]:


color_list = ['red','green','blue','yellow','orange']
median_list = ['black', 'black', 'black','black', 'black']
cluster_num = 5

# fig, ax = plt.subplots(1, 1, figsize = (30, 12))
for i in range(cluster_num):
    rows = plot_hour[plot_hour['cluster_meanShift'] == i]
    rows.plot('Hour', 'VNC', kind = 'line', color= color_list[i],alpha = 0.8, legend = False, figsize = (30, 12), grid = True, xticks = range(0,24), fontsize = 30)
    median = rows.groupby([ 'Hour'])['VNC'].mean()
    median.plot(color = color_list[i], linewidth = 10, ls = ':')
    plt.title('Hour, mean shift, cluster(pattern ' + str(i) + ")", fontsize = 30)
    plt.xlabel('Hour', fontsize = 30)
    plt.ylabel('VNC', fontsize = 30)
    plt.show()
    


# # train Date, DBSCAN

# In[ ]:


from sklearn.cluster import DBSCAN


# In[ ]:


predict = DBSCAN(eps = 0.01, min_samples = 100).fit_predict(data_train)


# In[ ]:


group_date['cluster_DBSCAN'] = predict


# In[ ]:


group_date['cluster_DBSCAN'].value_counts()


# ### 可以看出， 只分了一类， 760个data都是-1

# In[ ]:


group_date.head()


# In[ ]:


# id_list = group_date['Id'].drop_duplicates().tolist()
# color_list = ['red','green','blue']
# median_list = ['firebrick', 'darkgreen', 'darkblue']
fig, axes = plt.subplots(1, 1, figsize = (30, 12))
# fig.tight_layout()
# plt.subplots_adjust(wspace =0, hspace =0)

for i in range(len(id_list)):
    rows = group_date[group_date['Id'] == id_list[i]]
    cluster_index = int(rows['cluster_DBSCAN'].mode())  
    rows.plot('DateTime', 'Voltage', kind = 'line', color= 'blue', ax = axes, alpha = 0.8,legend = False, grid = True, fontsize = 30)
    
# plot median for 1 clusters
cluster_num = 1
for i in range(cluster_num):
    rows = group_date[group_date['cluster_DBSCAN'] == -1]
    median = rows.groupby([ 'DateTime'])['Voltage'].mean()
    median.plot(color = 'green',ax = axes, linewidth = 10, ls = ':')

plt.title('Date, DBSCAN, cluster_num = 1', fontsize = 30)
plt.xlabel('DateTime', fontsize = 30)
plt.ylabel('Voltage', fontsize = 30)
plt.show()


# # train Hour, DBSCAN

# In[ ]:


predict = DBSCAN(eps = 0.0000001, min_samples = 100).fit_predict(data_hour)


# In[ ]:


group_hour['cluster_DBSCAN'] = predict


# In[ ]:


group_hour['cluster_DBSCAN'].value_counts()


# In[ ]:


group_hour.head()


# In[ ]:


# color_list = ['red','green','blue']
# median_list = ['firebrick', 'darkgreen', 'darkblue']
cluster_num = 1

# fig, ax = plt.subplots(1, 1, figsize = (30, 12))
for i in range(cluster_num):
    rows = group_hour[group_hour['cluster_DBSCAN'] == -1]
    rows.plot('Hour', 'Voltage', kind = 'line', color= 'blue',alpha = 0.8, legend = False, figsize = (30, 12), grid = True, xticks = range(0,24), fontsize = 30)
    median = rows.groupby([ 'Hour'])['Voltage'].mean()
    median.plot(color = 'green', linewidth = 10, ls = ':')
    plt.title('Hour, DBSCAN, cluster_num = 1', fontsize = 30)
    plt.xlabel('Hour', fontsize = 30)
    plt.ylabel('Voltage', fontsize = 30)
    plt.show()
    


# In[ ]:




