import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt
#from bs4         import BeautifulSoup
#from nltk.corpus import stopwords
#from textblob    import Word, TextBlob
import nltk
from internal_functions.textCleaner     import textCleaner
from warnings                           import filterwarnings
from wordcloud                          import WordCloud
from PIL                                import Image
from nltk.sentiment                     import SentimentIntensityAnalyzer
from sklearn.preprocessing              import LabelEncoder
from sklearn.feature_extraction.text    import TfidfVectorizer
from sklearn.model_selection            import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model               import LogisticRegression
from sklearn.ensemble                   import RandomForestClassifier

filterwarnings("ignore")

#Download packages
#nltk.download('stopwords')    # --> Text Preprocessing
#nltk.download('wordnet')      # --> Text Preprocessing
#nltk.download('vader_lexicon') # --> Sentiment Analysis

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
#plt.savefig("Generated_photos/term_frequency.png") #save the bar plot
plt.show()
# Word Cloud
text = " ".join(i for i in df.Review) # transform Review column into single string
amazon_mask = np.array(Image.open("Dataset/amazon.png"))
wordcloud = WordCloud(background_color= "lightgray",
                      mask=amazon_mask,
                      max_words=1000,
                      contour_width=3,
                      contour_color="firebrick",
                      colormap="gist_rainbow",
                      width=8000,
                      height=4000).generate(text)
plt.figure(figsize=(20, 10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
#plt.savefig("Generated_photos/wordcloud__.png", dpi=600, bbox_inches='tight') #save wordcloud
plt.show()


#### Sentiment Analysis
print(df["Review"].head())
sia = SentimentIntensityAnalyzer()
df["polarity_score"] = df["Review"].apply(lambda x: sia.polarity_scores(x)["compound"])
print(df[["Review","polarity_score"]])

#### Feature Engineering
df["sentiment_label"] = df["Review"].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")
# How many negative/positive comments
print("Quantities:")
print(df["sentiment_label"].value_counts())
# The overall rating for comment polarity.
print("Average Overall Ratings:")
print(df.groupby("sentiment_label")["Star"].mean())
# The helpfulness votes for comment polarity.
print("Number of Helpfulness Votes:")
print(df.groupby("sentiment_label")["HelpFul"].sum())

df_analyzed = pd.concat([df["sentiment_label"].value_counts(), 
                         df.groupby("sentiment_label")["Star"].mean(), 
                         df.groupby("sentiment_label")["HelpFul"].sum()],
                        axis=1)
print(df_analyzed)
#df_analyzed.to_csv("Generated_files/df_analyzed.csv") # Save df_analyzed

# turn pos/neg to 1-0 for machine learning
df["sentiment_label"] = LabelEncoder().fit_transform(df["sentiment_label"])
# create dependent variable & independent variable
y = df["sentiment_label"]
X = df["Review"]
# Word vectorizing with TF-IDF Vectors
tf_idf_word_vectorizer = TfidfVectorizer()
X_tf_idf_word = tf_idf_word_vectorizer.fit_transform(X)
# Train-Test split
X_train, X_test, Y_train, Y_test = train_test_split(X_tf_idf_word,
                                                    y,
                                                    test_size=0.30,
                                                    random_state=20)
print("size of X_train  : "   + str(X_train.size),
      "\nsize of X_test   : " + str(X_test.size), 
      "\nsize of Y_train  : " + str(Y_train.size), 
      "\nsize of Y_test   : " + str(Y_test.size))

#### Sentiment Modeling
### Logistic Regressing Model

log_model = LogisticRegression(random_state=20)

log_params_l1 = {"penalty": ["l1"],
                 "solver": ["liblinear", "saga"],  # solvers for L1
                 "max_iter": [200, 500, 1000, 2000]
}
log_params_l2 = {"penalty": ["l2"],
                 "solver": ["liblinear", "lbfgs", "newton-cg","newton-cholesky", "sag", "saga"],  # solvers for L2
                 "max_iter": [200, 500, 1000, 2000]
}
log_params_elasticnet = {"penalty": ["elasticnet"],
                         "solver": ["saga"],             # Solver for ElasticNet
                         "l1_ratio": [0.25, 0.5, 0.75],  # effects of l1 and l2
                         "max_iter": [200, 500, 1000, 2000]
}


log_best_grid_l1 = GridSearchCV(log_model,
                                log_params_l1,
                                cv=10,
                                n_jobs=-1,
                                verbose=True).fit(X_tf_idf_word, y)
log_best_grid_l2 = GridSearchCV(log_model,
                                log_params_l2,
                                cv=10,
                                n_jobs=-1,
                                verbose=True).fit(X_tf_idf_word, y)
log_best_grid_elasticnet = GridSearchCV(log_model,
                                        log_params_elasticnet,
                                        cv=10,
                                        n_jobs=-1,
                                        verbose=True).fit(X_tf_idf_word, y)

print("l1 score        : %f" % log_best_grid_l1.best_score_)
print(log_best_grid_l1.best_params_)
print("l2 score        : %f" % log_best_grid_l2.best_score_)
print(log_best_grid_l2.best_params_)
print("ElasticNet score: %f" % log_best_grid_elasticnet.best_score_)
print(log_best_grid_elasticnet.best_params_)

###########################Obtained Data###########################
####l1 score        : 0.953695
####{'max_iter': 200, 'penalty': 'l1', 'solver': 'saga'}
####l2 score        : 0.911855
####{'max_iter': 200, 'penalty': 'l2', 'solver': 'saga'}
####ElasticNet score: 0.938495
####{'l1_ratio': 0.75, 'max_iter': 200, 'penalty': 'elasticnet', 'solver': 'saga'}

log_final = log_model.set_params(**log_best_grid_l1.best_params_,
                                 random_state=20).fit(X_train,y)
cross_val_score_log_model = cross_val_score(log_final,
                                            X_test,
                                            y,
                                            cv=10,
                                            n_jobs=-1).mean()
print("Cross validation score for Logistic Regression %f" %cross_val_score_log_model)