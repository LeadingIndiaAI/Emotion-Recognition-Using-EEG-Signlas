import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import seaborn as sns
#get_ipython().magic(u'matplotlib inline')

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import LSTM
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
#import xgboost as xgb

import warnings
warnings.filterwarnings('ignore')

brainwave_df = pd.read_csv('Neutral.csv')
brainwave_df = brainwave_df.mask(brainwave_df.eq(0)).dropna(how='all', axis=1)
brainwave_df=brainwave_df.dropna()

label_df=brainwave_df.drop('label',axis=1)


