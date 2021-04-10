# Faker
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/Django)

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

```Python
fake = Faker(['pt_BR'])
profiles = [fake.profile() for i in range(50)]
pd.DataFrame(profiles)
```