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


data = pd.read_csv("C:/Users/Kaige Zhang/Desktop/t1.csv")


# In[3]:


data.head(5)


# In[4]:


data.shape


# # group  by date

# In[5]:


data = data.drop(['GenerationCurrent'], axis = 1)
data = data.drop(['GenerationImportEnergy'], axis = 1)
data = data.drop(['GenerationExportEnergy'], axis = 1)
data = data.drop(['Impedance'], axis = 1)
data = data.drop(['PowerFactor'], axis = 1)
data = data.drop(['GridExportEnergy'], axis = 1)
data = data.drop(['GridImportEnergy'], axis = 1)
data = data.drop(['GridCurrent'], axis = 1)


# In[6]:


data.head(5)


# In[7]:


weekday_data =  data[data['Date'] == '2019-02-25']
weekday_data = weekday_data.groupby(['Id','Date','Hour']).mean()
weekday_data = weekday_data.reset_index()
weekday_data.head(5)


# In[8]:


weekday_data = weekday_data.drop(['Minute'], axis = 1)
weekday_data = weekday_data.drop(['Sec'], axis = 1)
weekday_data.head(5)


# In[9]:


id_list = weekday_data['Id'].drop_duplicates().tolist()
fig, ax = plt.subplots(1, 1, figsize = (30, 12))
for i in range(len(id_list)):
    rows = weekday_data[weekday_data['Id'] == id_list[i]]
    rows.plot('Hour', 'GridVoltage', kind = 'line',color = 'blue', ax = ax, legend = False, figsize = (30, 12), grid = True, fontsize = 30)

plt.title('Weekday Voltage Curve Graph', fontsize = 30)
plt.xlabel('Hour', fontsize = 30)
plt.ylabel('Voltage', fontsize = 30)
x = range(0,24,1)
plt.xticks(x)
plt.show()


# In[10]:


len(id_list)


# # new dataFrame

# In[12]:


weekday_data_kmeans = []
id_list = weekday_data['Id'].drop_duplicates().tolist()
for i in range(len(id_list)):
    rows = weekday_data[weekday_data['Id'] == id_list[i]]
    power = rows['GridVoltage'].tolist()
    power.insert(0, id_list[i])
    weekday_data_kmeans.append(power)
len(weekday_data_kmeans)


# In[13]:


num = 0
while (num < len(weekday_data_kmeans)):
    if len(weekday_data_kmeans[num]) != 25:
        weekday_data_kmeans.pop(num)
        num = num - 1
    num = num + 1
print (len(weekday_data_kmeans))


# In[14]:


weekday_dataFrame = np.zeros(shape=(len(weekday_data_kmeans),len(weekday_data_kmeans[0])))


# In[15]:


for i in range (len(weekday_data_kmeans)):
    for x in range (len(weekday_data_kmeans[0])):
        weekday_dataFrame[i][x] = weekday_data_kmeans[i][x]
weekday_dataFrame = pd.DataFrame(weekday_dataFrame)
type(weekday_dataFrame)


# In[16]:


weekday_dataFrame.head(5)


# # plot figure Weekday PowerConsumption 

# # train Date, k-means

# In[17]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics


# In[18]:


clinski_harabaz_scores = []
cluster_num = range(2, 10)

for i in cluster_num:
    kmeans = KMeans(n_clusters = i, n_jobs = 4)
    predict = kmeans.fit_predict(weekday_dataFrame)
    clinski_harabaz_scores.append(metrics.calinski_harabasz_score(weekday_dataFrame, predict))


# In[19]:


clinski_harabaz_scores


# In[20]:


kmeans = KMeans(n_clusters = 3, n_jobs = 4)
predict = kmeans.fit_predict(weekday_dataFrame)


# In[21]:


weekday_dataFrame['cluster_kmeans'] = predict


# In[22]:


weekday_dataFrame.head(36)


# In[24]:


color_list = ['red','green','blue']
median_list = ['firebrick', 'darkgreen', 'darkblue']
cluster_num = 3

# fig, ax = plt.subplots(1, 1, figsize = (30, 12))
for i in range(cluster_num):
    rows = weekday_dataFrame[weekday_dataFrame['cluster_kmeans'] == i]
    rows.plot('Hour', 'PowerConsumption', kind = 'line', color= color_list[i],alpha = 0.8, legend = False, figsize = (30, 12), grid = True, xticks = range(0,24), fontsize = 30)
    median = rows.groupby([ 'Hour'])['PowerConsumption'].mean()
    median.plot(color = color_list[i], linewidth = 10, ls = ':')
    plt.title('Hour, K-means, cluster(pattern ' + str(i) + ")", fontsize = 30)
    plt.xlabel('Hour', fontsize = 30)
    plt.ylabel('PowerConsumption', fontsize = 30)
    plt.show()


# ### Cluster Numbers = 3

# In[ ]:


kmeans = KMeans(n_clusters = 3, n_jobs = 4)
predict = kmeans.fit_predict(data_train)


# In[ ]:


plot_date['cluster_kmeans'] = predict


# In[ ]:


train_data[group_date['Id'] == 96]


# In[ ]:


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

# In[ ]:


clinski_harabaz_scores = []
cluster_num = range(2, 10)

for i in cluster_num:
    kmeans = KMeans(n_clusters = i, n_jobs = 4)
    predict = kmeans.fit_predict(plot_hour)
    clinski_harabaz_scores.append(metrics.calinski_harabaz_score(plot_hour, predict))


# In[ ]:


clinski_harabaz_scores


# In[ ]:


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

# In[ ]:


kmeans = KMeans(n_clusters = 3, n_jobs = 4)
predict = kmeans.fit_predict(plot_hour)


# In[ ]:


plot_hour['cluster_kmeans'] = predict


# In[ ]:


plot_hour.head()


# In[ ]:


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

# In[ ]:


data_hour = plot_hour.values.copy()


# In[ ]:


from sklearn.cluster import MeanShift, estimate_bandwidth


# In[ ]:


bandwidth = estimate_bandwidth(data_hour, quantile = 0.2, n_samples = 500)

meanShift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
predict = meanShift.fit_predict(data_train)


# In[ ]:


plot_date['cluster_meanShift'] = predict


# In[ ]:


plot_date['cluster_meanShift'].value_counts()


# ###### 可以看出， 只分了6类， 0， 1， 2，3， 4， 5对应类， 后面对应该类个数

# In[ ]:


plot_date.head()


# In[ ]:


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

# In[ ]:


bandwidth = estimate_bandwidth(plot_hour, quantile = 0.2, n_samples = 500)

meanShift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
predict = meanShift.fit_predict(plot_hour)


# In[ ]:


plot_hour['cluster_meanShift'] = predict


# In[ ]:


plot_hour.head()


# In[ ]:


plot_hour['cluster_meanShift'].value_counts()


# In[ ]:


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




