# Missingno

We saw how to visualize the missing data in a graphical format and understand the relationship that exists among the different columns. Missingno helps in understanding the structure of the dataset with very few lines of code. With this information, it becomes easier and more efficient to use pandas either to impute the values or to drop them and can increase the overall accuracy of the model. 

The first step in implementing this is to install the library using the pip command

```Python
import missingno as msno

msno.matrix(dataset)
```
Another way to visualize the data for missing values is by using bar plots.
The white lines indicate the missing values in each column.
![matrix](https://user-images.githubusercontent.com/53547474/114289461-86863100-9a4e-11eb-9fc7-59cae2069dac.png)

```Python
msno.bar(dataset)
```
These bars show the values that are proportional to the non-missing data in the dataset. Along with that, the number of values missing is also shown.
![bar](https://user-images.githubusercontent.com/53547474/114289463-8be37b80-9a4e-11eb-82b4-476410be83d5.png)

```Python
msno.heatmap(dataset)
```
But to get a better idea about correlations we need to use heatmaps.
The heatmap shows a positive correlation with blue. The darker the shade of blue, the more the correlation.
![heatmap](https://user-images.githubusercontent.com/53547474/114289464-9140c600-9a4e-11eb-87f8-bdf8ccd6fcec.png)
