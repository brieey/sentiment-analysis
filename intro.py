import pandas as pd #data manipulation
import numpy as np #data manipulation
import matplotlib.pyplot as plt  #visualization   
import seaborn as sns #visualization

plt.style.use('ggplot') #set the default style for plots

import nltk #the natural language tool 
nltk.download('punkt_tab') # download the punkt tokenizer models, this is used for tokenizing text into words.
nltk.download('averaged_perceptron_tagger_eng') #used for part of speech tagging.
nltk.download('maxent_ne_chunker_tab') #used for named entity recognition
nltk.download('words') 
nltk.download('vader_lexicon')


#Read in data
df = pd.read_csv('Reviews.csv') #read the csv file into a pandas dataframe
print(df.head()) #display the first 5 rows of the dataframe
print(" ")

#See the review for that first product
#Going to be running my sentiment analysis on the text column of this dataframe
print(df['Text'].values[0]) #display the review for the first product
print(" ")

print(df.shape) #display the shape of the dataframe, which is the number of rows and columns
df = df.head(500) #this take the first 500 rows
print(" ")

print(df.shape) 

## QUICK EDA 
#value_counts -> how many times the score occurs, sort_index just sorts them

print(" ")
print("Quick EDA:") #quick exploratory data analysis
print("--------------------------------")
print(df['Score'].value_counts().sort_index())

print(" ")
print("Basic NTLK") #quick exploratory data analysis
print("--------------------------------")
print(" ")

example = df['Text'][50] #get the 50th review from the dataframe
print(example) #print the review
print(" ")

#Tokenize the text using nltk. Splits the text into individual words.
print(nltk.word_tokenize(example))

print(" ")
print(" ")

#Often to need to conver the text to some formart that the computer can interpret 
# and format. 
tokens = nltk.word_tokenize(example) 
print("First 10 Tokens:") #display the tokens from the tokenized text
print("--------------------------------")
print(tokens[:10]) #display the first 10 tokens from the tokenized text

#NLTK can also find the part of speech for each token. 
#This is useful for understanding the grammatical structure of the text.
print(" ")
print("Part of Speech for Each Tag")
print("--------------------------------")
print(nltk.pos_tag(tokens)) #display the part of speech for each token in the text. 
#Will return a list of tuples, where each tuple contains the token and its corresponding part of speech.
print(" ")
print("Parts of Speech Tag for the First 10 Tokens")
print("--------------------------------")
tagged = nltk.pos_tag(tokens)
print(tagged[:10])
print(" ")

#We can take the tags and put them into entities
#identifies named entities(specific pieces of information or objects) and chunk them together.
print("Entities")
entities = nltk.chunk.ne_chunk(tagged) #identifies named entities
entities.pprint()
print(" ")

#VADER takes all the words in our sentence and it has a value of positive negative or
#neutral. 
from nltk.sentiment import SentimentIntensityAnalyzer #import the SentimentIntensityAnalyzer from nltk.sentiment        
from tqdm import tqdm #import tqdm for progress bar, this will be used to show the progress of the sentiment analysis on the dataframe
#from tqdm.notebook import tqdm 

#We can run this on some text to tell us the sentiment
sia = SentimentIntensityAnalyzer() #create an instance of the SentimentIntensityAnalyzer
print("Sentiment Analysis Example:") #display a message to indicate sentiment analysis example
print("--------------------------------")
#The compound score is a normalized score between -1 (most negative) and +1 (most positive).
print("I am so happy!")
print(sia.polarity_scores("I am so happy!"))
print(" ")  
print(" ") 
print("This is the worst thing ever.")
print(sia.polarity_scores("This is the worst thing ever.")) #display the sentiment scores for a negative sentence
print(" ")  
print(" ") 
print(" ") 

print("Sentiment Analysis on Example ")
print("--------------------------------")
print(sia.polarity_scores(example))
print(" ")
print(" ") 

print("Running Sentiment Analysis on the Entire Dataset")
print("--------------------------------")
res = {} #initialize an empty dictionary to store the results

for i, row in tqdm(df.iterrows(), total=len(df)): 
   text = row['Text']
   myid = row['Id'] #get the Id of the current row, this will be used as the key in the dictionary
   res[myid] = sia.polarity_scores(text) #initialize the dictionary entry for this Id to None, in case of any errors in sentiment 
   break
print(" ") 
print(" ")     


print("Prints Dictionary")
print("--------------------------------")
print(res)

print(" ")
print(" ")

print("Prints Pandas Dataframe")
print("--------------------------------")
#Easier to convert this into a pandas dataframe so we can analyze it better.
print(pd.DataFrame(res).T)
vaders = pd.DataFrame(res).T
