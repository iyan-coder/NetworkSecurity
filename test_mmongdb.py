
from pymongo.mongo_client import MongoClient

uri = "mongodb+srv://iyanuoluwaadebayo04:<@password>@cluster0.l0fwss3.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Create a new client and connect to the server
client = MongoClient(uri)

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)


# Setup github secrets:
# AWS_ACCESS_KEY_ID=

# AWS_SECRET_ACCESS_KEY=

# AWS_REGION = us-east-1

# AWS_ECR_LOGIN_URI = 788614365622.dkr.ecr.us-east-1.amazonaws.com/networkssecurity
# ECR_REPOSITORY_NAME = networkssecurity


# Docker Setup In EC2 commands to be Executed
# #optinal

# sudo apt-get update -y

# sudo apt-get upgrade

# #required

# curl -fsSL https://get.docker.com -o get-docker.sh

# sudo sh get-docker.sh

# sudo usermod -aG docker ubuntu

# newgrp docker