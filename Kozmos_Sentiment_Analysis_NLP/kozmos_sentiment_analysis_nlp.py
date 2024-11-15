import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt
#from bs4         import BeautifulSoup
#from nltk.corpus import stopwords
#from textblob    import Word, TextBlob
import nltk
from internal_functions.textCleaner import textCleaner
#import testtesttest
#nltk.download('stopwords')
#nltk.download('wordnet')

# Text Preprocessing
amazon_kozmos_data = pd.read_excel("Dataset/amazon.xlsx")
df = amazon_kozmos_data.copy()
#df.head()
#df.value_counts().sum()
#df
#df["Review"].isnull().sum()
#df.columns
df = df.dropna(subset="Review")
#df["Review"].isnull().sum()
#df["Review"][1]

df["Review"] = textCleaner(df["Review"])
print(df["Review"][321])
#rarewords_df = pd.Series(' '.join(df["Review"]).split()).value_counts()
#drops = rarewords_df[rarewords_df<=2]
#drops