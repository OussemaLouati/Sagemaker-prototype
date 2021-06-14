import json
import urllib.parse
import boto3
import csv
import os

print('Loading function')
    
s3 = boto3.client('s3')

out_bucket = os.environ['INFERENCE_OUTPUT_BUCKET']
endpoint_name = os.environ['MODEL_ENDPOINT']

def lambda_handler(event, context):
    #print("Received event: " + json.dumps(event, indent=2))

    # Get the object from the event and show its content type
    bucket = event['Records'][0]['s3']['bucket']['name']
    datetime = event['Records'][0]['eventTime']
    key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'], encoding='utf-8')
    
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        
        input = response['Body'].read()
        
        print("Data: " + input.decode('utf-8'))
        
        results = run_inference(endpoint_name, input.decode('utf-8'))
        
        save_results(
            output=results, 
            dest_bucket=out_bucket, 
            eventime=datetime
        )
        
        return response['Body'].read().decode('utf-8')
    
    except Exception as e:
        print(e)
        print('Error getting object {} from bucket {}. Make sure they exist and your bucket is in the same region as this function.'.format(key, bucket))
        raise e


def run_inference(endpoint, input):
    """ Invoke SageMaker endpoint

    Args:
        endpoint (string): endpoint name
        input (string): input csv rows to predict

    Raises:
        ex: SageMaker Endpoint Execptions

    Returns:
        [json]: results of batch inference
    """
    try:
        print('inference code here')

        client = boto3.client('sagemaker-runtime')

        # prepare the input
        # sstream = StringIO(input.split("\n"))
        csv_reader = csv.reader(input.split("\n"), delimiter=',')
        payload = json.dumps(list(csv_reader))
        print("Paylod: " + payload)
        # payload = list(csv_reader)

        # endpoint config    
        endpoint_name = "custom-sm-inference"                  # Your endpoint name.
        content_type = "application/json"                                   # The MIME type of the input data in the request body.
        accept = "application/json"                                         # The desired MIME type of the inference in the response.
        response = client.invoke_endpoint(
            EndpointName=endpoint_name, 
            ContentType=content_type,
            Accept=accept,
            Body=payload
            )

        print("RESULTS")

        output = response["Body"].read()
        print('Predicted labels are: {}'.format(output))
        
        return output


    except Exception as ex:
        print(ex)
        print('Error while invoking SageMaker endpoint {}. Make sure the endpoint is operational'.format(endpoint))
        raise ex
        

def save_results(output, dest_bucket, eventime):
    """ Saving inference results into destination bucket 

    Args:
        output ([list]): inference results
        dest_bucket (string): destination bucket
        eventime (string): eventTime of input data insertion 

    Raises:
        e: S3 IO Exceptions 
    """
    try:
        print("Saving results into: " + dest_bucket)
        out='output/inference-{}.csv'.format(eventime)
        s3.put_object(Body=output, Bucket=dest_bucket, Key=out)       

    except Exception as e:
        print(e)
        print('Error getting object {} from bucket {}. Make sure they exist and your bucket is in the same region as this function.'.format(key, bucket))
        raise e
