
import ibm_boto3
from ibm_botocore.client import Config, ClientError

class CloudObjectStore:
    '''
    Interface to IBM Cloud Object Store, typical values:
        bucket_name:  name of your storage bucket
        endpoint: for external access, "https://s3.us-east.cloud-object-storage.appdomain.cloud"
        endpoint: for internal access, "https://s3.private.us-east.cloud-object-storage.appdomain.cloud"
        api_key:  your API key, "Gok5I2azHDLqKtjdLvbVDB64JS-4IY9Mm7PJY6aMnBcO"
        auth_endpoint: "https://iam.cloud.ibm.com/identity/token"
        resource_crn:  your bucket crn,  "crn:v1:bluemix:public:cloud-object-storage:global:a/0b5a00334eaf9eb9339d2ab48f2d4767:e6f2b348-2eef-48af-9abe-338bea538358:bucket:dictations"

    '''
    def __init__(self, bucket_name, 
                 api_key, 
                 resource_crn,
                 endpoint="https://s3.us-east.cloud-object-storage.appdomain.cloud", 
                 auth_endpoint="https://iam.cloud.ibm.com/identity/token", 
                 ):
        self.bucket_name = bucket_name
        self.COS_API_KEY_ID = api_key
        self.COS_RESOURCE_CRN = resource_crn
        self.COS_ENDPOINT = endpoint
        self.COS_AUTH_ENDPOINT = auth_endpoint

        self.cos = ibm_boto3.resource("s3",
                                      ibm_api_key_id=self.COS_API_KEY_ID,
                                      ibm_service_instance_id=self.COS_RESOURCE_CRN,
                                      ibm_auth_endpoint=self.COS_AUTH_ENDPOINT,
                                      config=Config(signature_version="oauth"),
                                      endpoint_url=self.COS_ENDPOINT)
        
    def get_bucket_contents(self):
        print("Retrieving bucket contents from: {0}".format(self.bucket_name))
        try:
            files = self.cos.Bucket(self.bucket_name).objects.all()
            for file in files:
                print("Item: {0} ({1} bytes).".format(file.key, file.size))
        except ClientError as be:
            print("CLIENT ERROR: {0}\n".format(be))
        except Exception as e:
            print("Unable to retrieve bucket contents: {0}".format(e))
        
    def get_item(self, item_name):
        print("Retrieving item from bucket: {0}, key: {1}".format(self.bucket_name, item_name))
        try:
            file = self.cos.Object(self.bucket_name, item_name).get()
            print("File Contents: {0}".format(file["Body"].read()))
        except ClientError as be:
            print("CLIENT ERROR: {0}\n".format(be))
        except Exception as e:
            print("Unable to retrieve file contents: {0}".format(e))
        
    def create_text_file(self, item_name, file_text):
        print("Creating new item: {0}".format(item_name))
        try:
            self.cos.Object(self.bucket_name, item_name).put(
                Body=file_text
                )
            print("Item: {0} created!".format(item_name))
        except ClientError as be:
            print("CLIENT ERROR: {0}\n".format(be))
        except Exception as e:
            print("Unable to create text file: {0}".format(e))
