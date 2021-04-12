#!/usr/bin/env python
# coding: utf-8

# ## Simple contour plot

# This lesson illustrates a simple contour plot, contours in an image with a core bar for contours and labeled contours.

# In[43]:


import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt


delta = 0.25
x = np.arange(-2.0, 3.0, delta)
y = np.arange(-2.0, 3.0, delta)
X, Y = np.meshgrid(x, y)
Z1 = np.exp(-X**2 - Y**2)
Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
Z = (Z1 - Z2) * 2


# In[44]:


fig, ax = plt.subplots()
CS = ax.contour(X, Y, Z, 6, colors='k')
ax.clabel(CS, fontsize=9, inline=True)
ax.set_title('Single color - negative contours dashed')


# In[45]:


fig, ax = plt.subplots()
CS = ax.contour(X, Y, Z)
ax.clabel(CS, inline=True, fontsize=10)
ax.set_title('Simplest default with labels')


# In[46]:


delta = 0.25
x = np.arange(-1.0, 4.0, delta)
y = np.arange(-2.0, 3.0, delta)
X, Y = np.meshgrid(x, y)
Z1 = np.exp(-X**2 - Y**2 + X*3)
Z2 = np.exp(-(X-1)**2 - (Y - 1)**2)
Z = (Z1 - Z2) * 2


# In[47]:


fig, ax = plt.subplots()
CS = ax.contour(X, Y, Z, 6, colors='k')
ax.clabel(CS, fontsize=9, inline=True)
ax.set_title('Single color - negative contours dashed')


# In[48]:


fig, ax = plt.subplots()
CS = ax.contour(X, Y, Z)
ax.clabel(CS, inline=True, fontsize=10)
ax.set_title('Simplest default with labels')


# In[ ]:




