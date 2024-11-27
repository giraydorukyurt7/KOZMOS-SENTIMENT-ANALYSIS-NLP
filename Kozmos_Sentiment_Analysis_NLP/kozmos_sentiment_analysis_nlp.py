import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt
#from bs4         import BeautifulSoup
#from nltk.corpus import stopwords
#from textblob    import Word, TextBlob
import nltk
from internal_functions.textCleaner     import textCleaner
from internal_functions.dfToXml         import dfToXml
from warnings                           import filterwarnings
from wordcloud                          import WordCloud
from PIL                                import Image
from nltk.sentiment                     import SentimentIntensityAnalyzer
from sklearn.preprocessing              import LabelEncoder
from sklearn.feature_extraction.text    import TfidfVectorizer
from sklearn.model_selection            import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model               import LogisticRegression
from sklearn.ensemble                   import RandomForestClassifier
from sklearn.metrics                    import accuracy_score, classification_report

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
#df_analyzed.to_xml("Generated_files/df_analyzed.xml") # Save df_analyzed.xml

dfToXml(df            = df_analyzed,
        filename      = "df_analyzed.xml",
        filedirectory = "Frontend/DatasetAnalyzedPage",
        xsl_href      = "datasetanalyzedpage.xsl",
        index_        = True,
        encode        = "utf-8")

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

#log_params_l1 = {"penalty": ["l1"],
#                 "solver": ["liblinear", "saga"],  # solvers for L1
#                 "max_iter": [200, 500, 1000, 2000]
#}
#log_params_l2 = {"penalty": ["l2"],
#                 "solver": ["liblinear", "lbfgs", "newton-cg","newton-cholesky", "sag", "saga"],  # solvers for L2
#                 "max_iter": [200, 500, 1000, 2000]
#}
#log_params_elasticnet = {"penalty": ["elasticnet"],
#                         "solver": ["saga"],             # Solver for ElasticNet
#                         "l1_ratio": [0.25, 0.5, 0.75],  # effects of l1 and l2
#                         "max_iter": [200, 500, 1000, 2000]
#}
#
#
#log_best_grid_l1 = GridSearchCV(log_model,
#                                log_params_l1,
#                                cv=10,
#                                n_jobs=-1,
#                                verbose=True).fit(X_tf_idf_word, y)
#log_best_grid_l2 = GridSearchCV(log_model,
#                                log_params_l2,
#                                cv=10,
#                                n_jobs=-1,
#                                verbose=True).fit(X_tf_idf_word, y)
#log_best_grid_elasticnet = GridSearchCV(log_model,
#                                        log_params_elasticnet,
#                                        cv=10,
#                                        n_jobs=-1,
#                                        verbose=True).fit(X_tf_idf_word, y)
#
#print("l1 score        : %f" % log_best_grid_l1.best_score_)
#print(log_best_grid_l1.best_params_)
#print("l2 score        : %f" % log_best_grid_l2.best_score_)
#print(log_best_grid_l2.best_params_)
#print("ElasticNet score: %f" % log_best_grid_elasticnet.best_score_)
#print(log_best_grid_elasticnet.best_params_)

###########################Obtained Data###########################
####l1 score        : 0.953695
####{'max_iter': 200, 'penalty': 'l1', 'solver': 'saga'}
####l2 score        : 0.911855
####{'max_iter': 200, 'penalty': 'l2', 'solver': 'saga'}
####ElasticNet score: 0.938495
####{'l1_ratio': 0.75, 'max_iter': 200, 'penalty': 'elasticnet', 'solver': 'saga'}
# 
# !!! Model has been tuned according to these hyper parameters.
#
log_final = log_model.set_params(max_iter= 200, 
                                 penalty = 'l1', 
                                 solver  = 'saga',
                                 random_state=20).fit(X_train,Y_train)

# Error and Accuracy Metrics

cross_val_score_log_model = cross_val_score(log_final,
                                            X_test,
                                            Y_test,
                                            cv=10,
                                            n_jobs=-1).mean()
y_pred = log_final.predict(X_test)
accuracy_score_log_model = accuracy_score(Y_test, y_pred)
classification_report_log_model = classification_report(Y_test, y_pred, output_dict=True)
print("Cross validation score (Mean Cross-Validation Accuracy) for Logistic Regression %f" % cross_val_score_log_model)
print("Accuracy score (Test Set Accuracy) for Logistic Regression %f" % accuracy_score_log_model)
print("Classification Report for Logistic Regression: ")  
print(classification_report_log_model)

Log_model_scores_df = pd.DataFrame(classification_report_log_model).transpose()
Log_model_scores_df.loc["Cross Validation Score"] = [cross_val_score_log_model, None, None, None]
Log_model_scores_df.loc["Accuracy Score"] = [accuracy_score_log_model, None, None, None]
Log_model_scores_df.reset_index(inplace=True)
Log_model_scores_df.rename(columns={"index": "Metric"}, inplace=True)

# Save analyzed Log_model_scores_df.xml
#dfToXml(df            = Log_model_scores_df,
#        filename      = "Log_model_scores_df.xml",
#        filedirectory = "Frontend/FeaturesOfModelsPage",
#        xsl_href      = "featuresofmodelspage.xsl",
#        index_        = False,
#        encode        = "utf-8")

#log_model_file_name = "Generated_files/Log_model_scores_df.xml"
#Log_model_scores_df.to_xml(log_model_file_name,
#                           index=False, 
#                           root_name="data") # Save analyzed Log_model_scores_df.xml
#with open(log_model_file_name,"r", encoding="utf-8") as file:
#    xml_content = file.read()
#xslt_reference = '<?xml-stylesheet type="text/xsl" href="../Frontend/FeaturesOfModelsPage/featuresofmodelspage.xsl"?>\n'
#xml_with_xslt = xml_content.replace("<?xml version='1.0' encoding='utf-8'?>",
#                                    '<?xml version="1.0" encoding="utf-8"?>\n' + xslt_reference)
#with open(log_model_file_name, "w", encoding="utf-8") as file:
#    file.write(xml_with_xslt)
#### Testing the model
#examples
# sentence_to_df function
def sentence_to_df(sentence, vectorizer_method):
    #---Normal Sentence---
    #print(sentence)
    # Create Df
    df_sentence = pd.DataFrame([sentence], columns=['sentence'])
    # Clean sentence
    df_sentence['sentence'] = textCleaner(df_sentence['sentence'], rare_words=False)
    #---Cleaned Sentence---
    cleaned_sentence = df_sentence['sentence'].iloc[0]
    #print("\nCleaned:\n", cleaned_sentence)
    # tf-idf
    new_sentence_tf_idf = vectorizer_method.transform(df_sentence['sentence'])
    #print(new_sentence_tf_idf) #remove '#' if you want to see the values of the attributes
    return new_sentence_tf_idf, cleaned_sentence

## Test sentence
#sentence1 = "The curtains look great and set a dramatic tone to the room. They are thin enough to allow in sunlight so the room isnt completely dark. curtain look great set dramatic room thin enough allow sunlight room completely dark"
#a = sentence_to_df(sentence1, tf_idf_word_vectorizer)
#
#
#sentence1 = "The movie was great."
#sentence2 = "This is the worst holiday trip I have ever been."
#sentence3 = "I really liked this product's features."
#
#sentences = [sentence1, sentence2, sentence3]
#
#for sentence in sentences:
#      sentence_to_df(sentence, tf_idf_word_vectorizer)

# Test on Random Sentences
randomSentencesFromDataset_df = amazon_kozmos_data["Review"]
def findRandomSentencesFromDataset(df):
    index = np.random.randint(0, (df.size))
    return df[index]
randomSentence = findRandomSentencesFromDataset(randomSentencesFromDataset_df)
randomSentence_tf_idf, cleaned_sentence =sentence_to_df(randomSentence, tf_idf_word_vectorizer)
prediction = log_final.predict(randomSentence_tf_idf)
print("\nRandom Sentence:\n", randomSentence)
print("\nCleaned:\n", cleaned_sentence)
print("\nPredicted Label:\n", prediction)

#### Random Forests
rf_model = RandomForestClassifier(random_state=20)
#
#
#
#rf_params = {'n_estimators'     : [200, 500, 750, 1000],
#             'max_depth'        : [10, 25, 50],
#             'min_samples_split': [30,50,75,100],
#             'min_samples_leaf' : [5,25,50],
#             'max_features'     : [100, 250, 500, 1000],
#             'bootstrap'        : [True],
#             'criterion'        : ['gini'] # gini works better for big datasets, entropy works better for small datasets.
#             }
#
#rf_tuned = GridSearchCV(rf_model,
#                        rf_params,
#                        cv=10,
#                        n_jobs=-1,
#                        verbose=3).fit(X_train, Y_train)
#
#  !!! Model has been tuned according to these hyper parameters. The training duration took 51 minutes.
#
rf_final = rf_model.set_params(bootstrap = True,
                               criterion = 'gini',
                               max_depth= 50,
                               max_features= 1000,
                               min_samples_leaf= 5,
                               min_samples_split= 30,
                               n_estimators = 750,
                               random_state=20).fit(X_train,Y_train)

cross_val_score_rf_model = cross_val_score(rf_final,
                                           X_test,
                                           Y_test,
                                           cv=10,
                                           n_jobs=-1).mean()

y_pred = rf_final.predict(X_test)
accuracy_score_rf_model = accuracy_score(Y_test, y_pred)
classification_report_rf_model = classification_report(Y_test, y_pred, output_dict=True)
print("Cross validation score (Mean Cross-Validation Accuracy) for Random Forests %f" % cross_val_score_rf_model)
print("Accuracy score (Test Set Accuracy) for Random Forests %f" % accuracy_score_rf_model)
print("Classification Report for Random Forests: ")  
print(classification_report_rf_model)

rf_model_scores_df = pd.DataFrame(classification_report_rf_model).transpose()
rf_model_scores_df.loc["Cross Validation Score"] = [cross_val_score_rf_model, None, None, None]
rf_model_scores_df.loc["Accuracy Score"] = [accuracy_score_rf_model, None, None, None]
rf_model_scores_df.reset_index(inplace=True)
rf_model_scores_df.rename(columns={"index": "Metric"}, inplace=True)

# Save analyzed Rf_model_scores_df.xml
#dfToXml(df            = rf_model_scores_df,
#        filename      = "Rf_model_scores_df.xml",
#        filedirectory = "Frontend/FeaturesOfModelsPage",
#        xsl_href      = "featuresofmodelspage.xsl",
#        index_        = False,
#        encode        = "utf-8")


randomSentencesFromDataset_df = amazon_kozmos_data["Review"]
def findRandomSentencesFromDataset(df):
    index = np.random.randint(0, (df.size))
    return df[index]
randomSentence = findRandomSentencesFromDataset(randomSentencesFromDataset_df)
randomSentence_tf_idf, cleaned_sentence =sentence_to_df(randomSentence, tf_idf_word_vectorizer)
prediction = rf_final.predict(randomSentence_tf_idf)
print("\nRandom Sentence:\n", randomSentence)
print("\nCleaned:\n", cleaned_sentence)
print("\nPredicted Label:\n", prediction)