# Faker
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/Django)

Imagine that you need a dataframe with random information from people, such as email, name, website, city, date of birth, etc. You can create this dataframe the hard way:

```Python
import numpy as np
import pandas as pd
from random import randrange, uniform
import random

GeneroList = ['Masculino','Feminino']

df = pd.DataFrame()
COLUNAS = [
    'Genero',
    'Altura',
    'Idade',
    'Peso'
]
df = pd.DataFrame(columns=COLUNAS)

for x in range(100):
    genero = random.choice(GeneroList)
    altura = uniform(1.50,2.0)
    idade = randrange(15,60)
    peso = uniform(40.0,100.0)
    
    df.loc[-1] = [genero, altura, idade, peso]
    df.index = df.index + 1 
    df = df.sort_index()

df['Idade'] = df['Idade'].astype(int)
```
Or you can create this dataframe the easy way: Using Faker
> Faker is a Python package that generates fake data for you. Whether you need to bootstrap your database, create good-looking XML documents, fill-in your persistence to stress test it, or anonymize data taken from a production service, Faker is for you.

### Installation
```Python
pip install Faker
```
### Libraries
```Python
from faker import Faker
from faker.providers import internet
import pandas as pd
```
### Criando dataframe
```Python
fake = Faker('pt_BR')
df = pd.DataFrame({"Name" : [fake.name() for x in range(0, 50)],
                   "Website" : [fake.url() for x in range(0, 50)],
                   "Endereço" : [fake.address() for x in range(0, 50)],
                   "Email" : [fake.email() for x in range(0, 50)],
                   "Nascimento" : [fake.date_of_birth() for x in range(0, 50)],
                   "Profissao" : [fake.job() for x in range(0, 50)]
                  })
```
### Creating dataframe - mode 2
```Python
fake = Faker(['pt_BR'])
profiles = [fake.profile() for i in range(50)]
pd.DataFrame(profiles)
```
### 
