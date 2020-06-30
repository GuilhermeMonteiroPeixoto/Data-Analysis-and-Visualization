#AULA 4 PANDAS

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# link studentdata: https://www.kaggle.com/spscientist/students-performance-in-exams
studentdata = pd.read_csv('datasets_74977_169835_StudentsPerformance.csv')

grouped = studentdata[['math score','reading score','writing score']].groupby([studentdata['parental level of education'],studentdata['gender']])
print('\n\n',grouped.mean())

print('\n\n',grouped['math score'].count())

print('\n\n',grouped[['math score','reading score']].min())

print('\n\n',grouped[['math score','reading score']].max())

grouped = studentdata.groupby('parental level of education')
grouped = grouped['writing score'].agg(['min','max','count','mean'])

print('\n\n',grouped)

#Primeira nota de Matematica em cada agrupamento de parental level of education
grouped = studentdata.groupby('parental level of education').apply(lambda df: df['math score'].iloc[0])
print('\n\n',grouped)

grouped = studentdata.groupby(['parental level of education','gender']).apply(lambda df: df.loc[df['writing score'].idxmax()])
print('\n\n',grouped)
