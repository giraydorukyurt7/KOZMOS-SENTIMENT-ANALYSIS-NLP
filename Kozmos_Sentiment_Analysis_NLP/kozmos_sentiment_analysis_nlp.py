import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

import nltk
nltk.download('stopwords')
# Text Preprocessing
amazon_kozmos_data = pd.read_excel("Dataset/amazon.xlsx")
df = amazon_kozmos_data.copy()
#df.head()
#df.value_counts().sum()
#df
df["Review"].isnull().sum()
#df.columns
df = df.dropna(subset="Review")
df["Review"].isnull().sum()
#df["Review"][1]
def textCleaner(df_text):
    # Remove html elements
    df_text = df_text.apply(lambda x: ' '.join(BeautifulSoup(str(x), "html.parser").get_text().split()) if pd.notnull(x) else x)
    # lowerCase transformation
    df_text = df_text.str.lower()
    # Remove punctions
    df_text = df_text.str.replace(r'[^\w\s]', ' ', regex=True)
    # Remove Numbers
    df_text = df_text.str.replace(r'\d+', " ", regex=True)
    # Remove Stopwords
    sw = stopwords.words('english')
    df_text = df_text.apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))

    # Remove Rarewords
    rarewords_df = pd.Series(' '.join(df_text).split()).value_counts()
    drops = rarewords_df[rarewords_df<=2]
    df_text = df_text.apply(lambda x: " ".join(x for x in str(x).split() if x not in drops))
    # Remove Extra Spaces
    df_text = df_text.str.replace(r'\s+', ' ', regex=True)
    return df_text
df["Review"] = textCleaner(df["Review"])
df["Review"][2]
#rarewords_df = pd.Series(' '.join(df["Review"]).split()).value_counts()
#drops = rarewords_df[rarewords_df<=2]
#drops