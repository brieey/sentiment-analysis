import pandas as pd #data manipulation
import numpy as np #data manipulation
import matplotlib.pyplot as plt  #visualization   
import seaborn as sns #visualization

plt.style.use('ggplot') #set the default style for plots

import nltk #the natural language tool 

#Read in data
df = pd.read_csv('Reviews.csv') #read the csv file into a pandas dataframe
print(df.head()) #display the first 5 rows of the dataframe

#See the review for that first product
print(df['Text'].values[0]) #display the review for the first product
