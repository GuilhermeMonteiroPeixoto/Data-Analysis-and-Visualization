#AULA 3 PANDAS

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# link studentdata: https://www.kaggle.com/spscientist/students-performance-in-exams
studentdata = pd.read_csv('datasets_74977_169835_StudentsPerformance.csv')

#Conditional selection

print('Math Score menor que 30')
data2 = studentdata[['gender','parental level of education',
                              'writing score','reading score','math score']].loc[(studentdata['math score'] <= 30)]
print(data2)

data2 = studentdata[['gender','parental level of education',
                              'writing score','reading score','math score']].loc[(studentdata['writing score'] == 100) |
                                                                                 (studentdata['reading score'] == 100) |
                                                                                 (studentdata['math score'] == 100)]
print(data2)
#In Pandas, in order to use 'and' logical operation we have to use '&'
#'or' operation, use '|'

data2 = studentdata[['gender','parental level of education',
                              'writing score','reading score','math score']].loc[(studentdata['gender'] == 'female') &
                                                                                 (studentdata['math score'] >= 95)]
print(data2)

data2 = studentdata[['gender','parental level of education',
                              'writing score','reading score','math score']].loc[(studentdata['parental level of education'] == 'some college')]
print(data2)


