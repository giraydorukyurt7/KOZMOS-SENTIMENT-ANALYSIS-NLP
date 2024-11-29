import pandas as pd
from internal_functions.textCleaner import textCleaner

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