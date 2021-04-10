#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install faker')


# In[20]:


from faker import Faker


# In[33]:


import pandas as pd
fake = Faker('pt_BR')


# In[32]:


df = pd.DataFrame({"Name" : [fake.name() for x in range(0, 50)],
                   "Website" : [fake.url() for x in range(0, 50)],
                   "Endereço" : [fake.address() for x in range(0, 50)],
                   "Email" : [fake.email() for x in range(0, 50)],
                   "Nascimento" : [fake.date_of_birth() for x in range(0, 50)],
                   "Profissao" : [fake.job() for x in range(0, 50)]
                  })
df


# In[42]:


fake = Faker()
fake.sentence()


# In[43]:


fake = Faker()
fake.profile()


# In[46]:


fake = Faker(['en_US'])
profiles = [fake.profile() for i in range(50)]
pd.DataFrame(profiles)


# In[ ]:




