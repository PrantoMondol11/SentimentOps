import boto3
import pandas as pd
import logging
from src.logger import logging
from io import StringIO
import os  
region = os.getenv("AWS_DEFAULT_REGION")

if not region:
    logging.info("AWS_DEFAULT_REGION is not set")

print("Using region:", region)

aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
if not aws_access_key_id:
    logging.error("AWS_ACCESS_KEY_ID is not set")
    
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
if not aws_secret_access_key:
    logging.error("AWS_SECRET_ACCESS_KEY is not set")
   

print("AWS credentials loaded successfully",aws_access_key_id[:4]+"****",aws_secret_access_key[:4]+"****")
print("aws_secret_access_key_id:",aws_access_key_id)
class s3_connection:
    def __init__(self):
        self.s3_client = boto3.client("s3",
                                      aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                                      aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                                      region_name=os.getenv("AWS_DEFAULT_REGION"))
        logging.info("S3 connection established successfully")
        
        
    def get_object(self,Bucket:str,Key:str) -> pd.DataFrame:
        """Get object from S3 bucket and return as pandas DataFrame"""
        try:
            obj=self.s3_client.get_object(Bucket=Bucket,Key=Key)
            data=obj['Body'].read().decode('utf-8')
            df=pd.read_csv(StringIO(data))
            logging.info(f"Object '{Key}' retrieved successfully from bucket '{Bucket}'")
            return df
        except self.s3_client.exceptions.NoSuchKey:
            logging.error(f"Object '{Key}' not found in bucket '{Bucket}'")
            raise
        except Exception as e:
            logging.error(f"Error retrieving object '{Key}' from bucket '{Bucket}': {e}")
            raise