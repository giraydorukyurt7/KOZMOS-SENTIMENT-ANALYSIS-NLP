import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup

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
    df_text = df_text.apply(lambda x: ' '.join(BeautifulSoup(x, "html.parser").get_text().split()) if pd.notnull(x) else x) # Remove html elements
    df_text = df_text.str.lower()                             # lowerCase transformation
    df_text = df_text.str.replace(r'[^\w\s]', '', regex=True) # Remove punctions
    df_text = df_text.str.replace(r'\s+', ' ', regex=True) # Remove ekstra spaces
    return df_text
df["Review"] = textCleaner(df["Review"])
df["Review"][2]

