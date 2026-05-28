import numpy as np
import pandas as pd
import os
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from src.logger import logging
nltk.download('stopwords')
nltk.download('wordnet')


def preprocess_dataframe(df,col='text'):
    
    lemmatizer=WordNetLemmatizer()
    stop_words=set(stopwords.words('english'))
    
    def preprocess_text(text):
        text = re.sub(r'http?://\s+|www\.\S+', '', text)
        text=''.join([word for word in text.split() if word not in stop_words])
        text=''.join([char for char in text if not char.isdigit])
        text=text.lower()
        text=re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text=text.replace(':',"")
        text=re.sub('\s+', ' ', text).strip()
        text=' '.join([lemmatizer.lemmatize(word) for word in text.split()])
        return text
    
    df[col]=df[col].apply(preprocess_text)
    df=df.dropna(subset=[col])
    
    logging.info
    return df

def main():
    try:
        train_data=pd.read_csv("./data/raw/tarin.csv")
        test_data=pd.read_csv("./data/raw/test.csv")
        logging.info("Data loaded successfully.")
        logging.info("Starting data preprocessing...")
    
        train_preprocessed_data=preprocess_dataframe(train_data,'review')
        test_preprocessed_data=preprocess_dataframe(test_data,'review')
        logging.info("Data preprocessing completed successfully.")
        
        data_path=os.path.join("./data","interim")
        os.makedirs(data_path, exist_ok=True)
        
        train_preprocessed_data.to_csv(os.path.join(data_path,"train_preprocessed.csv"),index=False)
        test_preprocessed_data.to_csv(os.path.join(data_path,"test_preprocessed.csv"),index=False)
        logging.info("Preprocessed data saved successfully. %s",data_path)
    except FileNotFoundError as e:
        logging.error(f"Data file not found: {e}")
        
if __name__=="__main__":
    main()
        
    