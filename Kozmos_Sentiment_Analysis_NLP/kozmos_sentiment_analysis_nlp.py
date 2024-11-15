import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt
#from bs4         import BeautifulSoup
#from nltk.corpus import stopwords
#from textblob    import Word, TextBlob
import nltk
from internal_functions.textCleaner import textCleaner
from warnings import filterwarnings
filterwarnings("ignore")

#import testtesttest
#nltk.download('stopwords')
#nltk.download('wordnet')

#### Text Preprocessing
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


#### Text Visualization
# Term Frequency
tf = df["Review"].apply(lambda x: pd.Series(x.split(" ")).value_counts()).sum(axis=0).reset_index()
tf.columns = ['Words', 'tf']
print(tf)

# Bar plot
ax = tf[tf["tf"]>200].plot.bar(x="Words", y="tf")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)

plt.tight_layout()
plt.savefig("Generated_photos/term_frequency.png")
plt.show()