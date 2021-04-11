# Missingno

We saw how to visualize the missing data in a graphical format and understand the relationship that exists among the different columns. Missingno helps in understanding the structure of the dataset with very few lines of code. With this information, it becomes easier and more efficient to use pandas either to impute the values or to drop them and can increase the overall accuracy of the model. 

```Python
import missingno as msno

msno.matrix(dataset)
msno.heatmap(dataset)
msno.bar(dataset)
```