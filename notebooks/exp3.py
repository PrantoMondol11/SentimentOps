import os
import re
import string
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub

from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import warnings
warnings.simplefilter("ignore",UserWarning)
warnings.filterwarnings("ignore")

mlflow.set_tracking_uri("https://dagshub.com/mondolpranto83/SentimentOps.mlflow")
dagshub.init(repo_name="SentimentOps", repo_owner="mondolpranto83",mlflow=True)
mlflow.set_experiment("Hyperparameter Tuning")


