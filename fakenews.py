import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
fake = pd.read_csv(r"C:\Users\HP\Desktop\Fake.csv")
true = pd.read_csv(r"C:\Users\HP\Desktop\True.csv")
fake.shape
true.shape
# Add flag to track fake and real
fake['target'] = 'fake'
true['target'] = 'true'
# Concatenate dataframes
data = pd.concat([fake, true]).reset_index(drop = True)
data.shape
# Shuffle the data
from sklearn.utils import shuffle
data = shuffle(data)
data = data.reset_index(drop=True)
# Check the data
data.head()