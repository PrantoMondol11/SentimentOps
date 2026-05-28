import boto3
import pandas as pd
import logging
from src.logger import logging
from io import StringIO


class s3_connection:
    def __init__(self,bucket_name:str,aws_access_key:str,aws_secret_key:str,region_name="us-east-1"):
        self.bucket_name=bucket_name
        self.s3_client=boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=region_name
        )
        logging.info("S3 connection established successfully")
        
        
    def get_object(self,key:str) -> pd.DataFrame:
        """Get object from S3 bucket and return as pandas DataFrame"""
        try:
            obj=self.s3_client.get_object(Bucket=self.bucket_name,Key=key)
            data=obj['Body'].read().decode('utf-8')
            df=pd.read_csv(StringIO(data))
            logging.info(f"Object '{key}' retrieved successfully from bucket '{self.bucket_name}'")
            return df
        except self.s3_client.exceptions.NoSuchKey:
            logging.error(f"Object '{key}' not found in bucket '{self.bucket_name}'")
            raise
        except Exception as e:
            logging.error(f"Error retrieving object '{key}' from bucket '{self.bucket_name}': {e}")
            raise