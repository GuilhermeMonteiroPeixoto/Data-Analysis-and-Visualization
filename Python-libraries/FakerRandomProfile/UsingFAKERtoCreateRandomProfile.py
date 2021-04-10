#!/usr/bin/env python
# coding: utf-8

get_ipython().system('pip install faker')
from faker import Faker
import pandas as pd
fake = Faker('pt_BR')

df = pd.DataFrame({"Name" : [fake.name() for x in range(0, 50)],
                   "Website" : [fake.url() for x in range(0, 50)],
                   "Endereço" : [fake.address() for x in range(0, 50)],
                   "Email" : [fake.email() for x in range(0, 50)],
                   "Nascimento" : [fake.date_of_birth() for x in range(0, 50)],
                   "Profissao" : [fake.job() for x in range(0, 50)]
                  })
df

fake = Faker()
fake.sentence()

fake = Faker()
fake.profile()

fake = Faker(['en_US'])
profiles = [fake.profile() for i in range(50)]
pd.DataFrame(profiles)
