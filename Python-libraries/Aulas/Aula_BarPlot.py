#!/usr/bin/env python
# coding: utf-8

# In this file, we demonstrate how to create a grouped bar chart and annotate bars with labels

# In[2]:


import matplotlib
import matplotlib.pyplot as plt
import numpy as np


labels = ['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10']
men_means = [20, 34, 30, 35, 27, 30, 31, 32, 23, 34]
women_means = [25, 32, 34, 20, 25, 29, 30, 35, 27, 30]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, men_means, width, label='S - 1')
rects2 = ax.bar(x + width/2, women_means, width, label='S - 2')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Lucro Anual (mi R$)')
ax.set_title('Lucro Anual por Empresa')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()

plt.show()


# This is an example of creating a stacked bar plot with error bars using bar. Note the parameters yerr used for error bars.

# In[3]:


labels = ['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10']
men_means = [20, 34, 30, 35, 27, 30, 31, 32, 23, 34]
women_means = [25, 32, 34, 20, 25, 29, 30, 35, 27, 30]
men_std = [2, 3, 4, 1, 2, 1, 4, 1, 2, 1]
women_std = [3, 5, 2, 3, 3, 2, 4, 5, 6, 7]
width = 0.35       # the width of the bars: can also be len(x) sequence

fig, ax = plt.subplots()

ax.bar(labels, men_means, width, yerr=men_std, label='S - 1')
ax.bar(labels, women_means, width, yerr=women_std, bottom=men_means,
       label='S - 2')

ax.set_ylabel('Lucro Anual (mi R$)')
ax.set_title('Lucro Anual por Empresa')
ax.legend()

plt.show()


# In[4]:


N = 10
menMeans = (20, 35, 30, 35, -27, -20, -17, -10, 5, 10)
womenMeans = (25, 32, 34, 20, -25, -15, -27, -11, 4, 11)
menStd = (2, 3, 4, 1, 2, 5, 4, 3, 2, 1)
womenStd = (3, 5, 2, 3, 3, 1, 1, 2, 2, 2)
ind = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence


# In[5]:


fig, ax = plt.subplots()

p1 = ax.bar(ind, menMeans, width, yerr=menStd, label='S - 1')
p2 = ax.bar(ind, womenMeans, width,
            bottom=menMeans, yerr=womenStd, label='S - 2')

ax.axhline(0, color='grey', linewidth=0.8)
ax.set_ylabel('Lucro Anual (mi R$)')
ax.set_title('Lucro Anual por Empresa')
ax.set_xticks(ind)
ax.set_xticklabels(('E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10'))
ax.legend()

plt.show()


# In[ ]:




