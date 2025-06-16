import os
from networksecurity.logger.logger import logger

class S3Sync:
    def sync_folder_to_s3(self,folder,aws_bucket_url):
        command = f"aws s3 sync {folder} {aws_bucket_url}"
        logger.info(f"Syncing local folder '{folder}' to S3 bucket '{aws_bucket_url}'...")
        os.system(command)

    def sync_folder_from_s3(self,folder,aws_bucket_url):
        command = f"aws s3 sync {aws_bucket_url} {folder}"
        logger.info(f"Syncing S3 bucket '{aws_bucket_url}' to local folder '{folder}'...")
        os.system(command)




