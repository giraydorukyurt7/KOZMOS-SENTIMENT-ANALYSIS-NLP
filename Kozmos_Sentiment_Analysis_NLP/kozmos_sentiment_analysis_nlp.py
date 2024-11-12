import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Text Preprocessing
df = pd.read_excel("Dataset/amazon.xlsx")
df.head()
df.value_counts().sum()
df
df.isnull().sum()
df.columns

