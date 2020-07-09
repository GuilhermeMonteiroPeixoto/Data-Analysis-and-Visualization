#Separar Email eh Valido

import numpy as np
import pandas as pd
import re

emails = pd.Series(['gomp.com', 'guilherme@gmail.com.br', 'guilherme@',
					'educacao@hotmail.com', 'aguamineral.com.br', 'usa.@uf',
					'email_valido@yahoo.com','escola@alpha@', 'gompeixoto.95@fis.ufpe.br'])

pattern ='[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,4}'

validos = pd.Series(emails.str.findall(pattern, flags=re.IGNORECASE))

for email in validos:
	if len(email) > 0:
		print(str(email))