## SageMaker Custom Training and Inference:

This prototype demonstrates how to build and deploy containerized model training jobs and inference servers to AWS SageMaker as well as consuming the served model using AWS Lambda.

### I. Setup:

To setup the AWS resources required for deploying the training and inference components: 

0. Start by adding these variables
    ```bash
    export AWS_ACCESS_KEY_ID="xxxxx"
    export AWS_SECRET_ACCESS_KEY="xxxxxx"
    export ACCOUNT_ID="xxxxxxx"
    export REGION="xxxxx"
    export AWS_USER="Administrator"
    ```

0. Prepare environment variables:
    ```bash
    export TRAIN_IMAGE="custom-training-test"
    export INFERENCE_IMAGE="custom-inference-test"
    export DATA_BUCKET="custom-sm-test-data"
    export MODELS_BUCKET="custom-sm-test-models"
    export ROLE_PREFIX="1234"
    ```

1. Create S3 buckets to store dataset and models:
    ```bash
    aws s3api create-bucket --bucket $DATA_BUCKET --region $REGION --create-bucket-configuration LocationConstraint=$REGION
    ```
    ```bash
    aws s3api create-bucket --bucket $MODELS_BUCKET --region $REGION --create-bucket-configuration LocationConstraint=$REGION
    ```

2. Create an ECR repository to host the training and inference image:
    ```bash
    aws ecr create-repository --repository-name $TRAIN_IMAGE --region $REGION
    ```

    ```bash
    aws ecr create-repository --repository-name $INFERENCE_IMAGE --region $REGION
    ```
    > The command outputs a repository URL: `xxxxxxxxxx.dkr.ecr.eu-central-1.amazonaws.com/custom-inference-test`

3. Create an IAM role with access to S3, ECR, EC2, and SageMaker:
    ```bash
    aws iam create-role \
        --role-name AmazonSageMaker-ExecutionRole-$ROLE_PREFIX \
        --assume-role-policy-document "{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Principal\":{\"AWS\":\"arn:aws:iam::$ACCOUNT_ID:user\/$AWS_USER\",\"Service\":\"sagemaker.amazonaws.com\"},\"Action\":\"sts:AssumeRole\"}]}" \
        --description="Created from CLI"
    
    aws iam attach-role-policy \
        --role-name AmazonSageMaker-ExecutionRole-$ROLE_PREFIX \
        --policy-arn "arn:aws:iam::aws:policy/AmazonS3FullAccess"
    
    aws iam attach-role-policy \
        --role-name AmazonSageMaker-ExecutionRole-$ROLE_PREFIX \
        --policy-arn "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
    
    aws iam attach-role-policy \
        --role-name AmazonSageMaker-ExecutionRole-$ROLE_PREFIX \
        --policy-arn "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryFullAccess"
         
    aws iam attach-role-policy \
        --role-name AmazonSageMaker-ExecutionRole-$ROLE_PREFIX \
        --policy-arn "arn:aws:iam::aws:policy/AmazonElasticContainerRegistryPublicFullAccess"
    ```


### II. Server Deployment:

#### Training:

To deploy training jobs: 

0. Prepare environment variables:
    ```bash
    export TRAINING_JOB="custom-sm-training"
    export TRAINING_DATA_BUCKET_URI=s3://$DATA_BUCKET/training
    export VALIDATION_DATA_BUCKET_URI=s3://$DATA_BUCKET/validation
    export INSTANCE_TYPE="ml.m4.xlarge"
    export INSTANCE_VOLUME="5"
    export MAX_WAIT_TIME="3600"
    export MAX_RUN_TIME="1200"

    export MODEL_DOCKER_IMAGE=$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$TRAIN_IMAGE
    export SERVER_DOCKER_IMAGE=$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$INFERENCE_IMAGE
    ```

1. Configure docker to push to the ECR repository:
    ```bash
    aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/
    ```

2. Build, tag and push the docker image to ECR:
    ```bash
    docker build -t $TRAIN_IMAGE -f Dockerfile.training src/server
    docker tag $TRAIN_IMAGE $MODEL_DOCKER_IMAGE
    docker push $MODEL_DOCKER_IMAGE
    ```
    ```bash
    docker build -t $INFERENCE_IMAGE -f Dockerfile.inference src/server
    docker tag $INFERENCE_IMAGE $SERVER_DOCKER_IMAGE
    docker push $SERVER_DOCKER_IMAGE
    ```

4. Create a training job:
    ```bash    
    aws sagemaker create-training-job \
        --training-job-name "$TRAINING_JOB" \
        --hyper-parameters "{}" \
        --algorithm-specification "{\"TrainingImage\":\"$MODEL_DOCKER_IMAGE\",\"TrainingInputMode\":\"File\",\"MetricDefinitions\":[{\"Name\":\"Validation Accuracy\",\"Regex\":\"val_accuracy: ([0-9\\.]+)\"}],\"EnableSageMakerMetricsTimeSeries\":true}" \
        --role-arn "arn:aws:iam::$ACCOUNT_ID:role/AmazonSageMaker-ExecutionRole-$ROLE_PREFIX" \
        --input-data-config "[{\"ChannelName\":\"training\",\"DataSource\":{\"S3DataSource\":{\"S3DataType\":\"S3Prefix\",\"S3Uri\":\"$TRAINING_DATA_BUCKET_URI\",\"S3DataDistributionType\":\"FullyReplicated\"}},\"CompressionType\":\"None\",\"RecordWrapperType\":\"None\",\"InputMode\":\"File\"},{\"ChannelName\":\"validation\",\"DataSource\":{\"S3DataSource\":{\"S3DataType\":\"S3Prefix\",\"S3Uri\":\"$VALIDATION_DATA_BUCKET_URI\",\"S3DataDistributionType\":\"FullyReplicated\"}},\"CompressionType\":\"None\",\"RecordWrapperType\":\"None\",\"InputMode\":\"File\"}]"  \
        --output-data-config "{\"S3OutputPath\":\"$MODELS_BUCKET\"}" \
        --resource-config "{\"InstanceType\":\"$INSTANCE_TYPE\",\"InstanceCount\":1,\"VolumeSizeInGB\":$INSTANCE_VOLUME}" \
        --stopping-condition "{\"MaxRuntimeInSeconds\":$MAX_RUN_TIME,\"MaxWaitTimeInSeconds\":$MAX_WAIT_TIME}"
    ```


#### Inference:

0. Prepare environment variables:
    ```bash
    export MODEL="custom-sm-model-1"
    export MODEL_CONFIG="custom-sm-model-config-1"
    export MODEL_ENDPOINT="custom-sm-model-endpoint-1"
    export INSTANCE_TYPE="m4.xlarge"
    export ACCELERATOR_TYPE="ml.eia1.large"
    ```
1. Create a Sagemaker model:
    ```bash
    aws sagemaker create-model \
        --model-name "$MODEL" \
        --primary-container "{\"Image\":\"$IMAGE\",\"ImageConfig\":{\"RepositoryAccessMode\":\"Platform\"},\"Mode\":\"SingleModel\",\"ModelDataUrl\":\"$MODELS_BUCKET\/$TRAINING_JOB\/output\/model.tar.gz\"}" \
        --execution-role-arn "arn:aws:iam::$ACCOUNT_ID:role/AmazonSageMaker-ExecutionRole-$ROLE_PREFIX" 
    ```

2. Create a Sagemaker model endpoint configuration:
    ```bash
    aws sagemaker create-endpoint-config \
        --endpoint-config-name "$MODEL_CONFIG" \
        --production-variants "[{\"VariantName\":\"$MODEL\",\"ModelName\":\"custom-sm-model-$ROLE_PREFIX\",\"InitialInstanceCount\":1,\"InstanceType\":\"$INSTANCE_TYPE\",\"InitialVariantWeight\":1,\"AcceleratorType\":\"$ACCELERATOR_TYPE\"}]" \
        --data-capture-config "{\"EnableCapture\":false}"
    ```

3. Create a Sagemaker model endpoint:
    ```bash
    aws sagemaker create-endpoint \
        --endpoint-name "$MODEL_ENDPOINT" \
        --endpoint-config-name "$MODEL_CONFIG"
    ```
    

### III. Client Deployment:

In this section, we'll use a Lambda function and configure a trigger for Amazon Simple Storage Service (Amazon S3). The trigger invokes the inference function on SM endpoint every time that you add an object (inference data) to your Amazon S3 bucket.
https://docs.aws.amazon.com/lambda/latest/dg/with-s3-example.html

0. Configure environment variables 
    ```bash 
    export INFERENCE_INPUT_BUCKET="input-storage-for-inference"
    export INFERENCE_OUTPUT_BUCKET="output-storage-for-inference"
    export LAMBDA_FN_NAME="inferrer-from-s3-event" 
    export LAMBDA_FUNCTION_CONFIGURATION_ID="1234"
    ```
1. Configure the previously created IAM role with access to Lambda service:
     ```bash
    aws iam attach-role-policy \
        --role-name AmazonSageMaker-ExecutionRole-$ROLE_PREFIX \
        --policy-arn "arn:aws:iam::aws:policy/AWSLambda_FullAccess"
    ```

2. Create S3 buckets to store input data and inference results:
    ```bash
    aws s3api create-bucket --bucket $INFERENCE_INPUT_BUCKET --region $REGION --create-bucket-configuration LocationConstraint=$REGION
    ```
    ```bash
    aws s3api create-bucket --bucket $INFERENCE_OUTPUT_BUCKET --region $REGION --create-bucket-configuration LocationConstraint=$REGION
    ```

3. Create Lambda function to execute the inference (from zip package) :
    ```bash
    aws lambda create-function \ 
        --function-name $LAMBDA_FN_NAME \
        --zip-file fileb://absolute-path-to-inferrer-lambda-package.zip \ 
        --handler lambda_function.lambda_handler \
        --runtime python3.8 \
        --environment "Variables={INFERENCE_OUTPUT_BUCKET=$INFERENCE_OUTPUT_BUCKET,MODEL_ENDPOINT=$MODEL_ENDPOINT}" \
        --role-arn "arn:aws:iam::$ACCOUNT_ID:role/AmazonSageMaker-ExecutionRole-$ROLE_PREFIX"

    ```
5. Grant Lambda function Invoke perissions from S3 eventTrigger
   ```bash
    aws lambda add-permission  \
        --function-name $LAMBDA_FN_NAME \ 
        --action lambda:InvokeFunction \   
        --statement-id s3-account \
        --principal s3.amazonaws.com \
        --source-arn arn:aws:s3:::$INFERENCE_INPUT_BUCKET \
        --source-account $ACCOUNT_ID
        
4. Enable S3 eventTrigger on the created Lambda Function

   ```bash
    aws s3api put-bucket-notification-configuration  \
        --bucket $INFERENCE_INPUT_BUCKET \
        --notification-configuration "{\"LambdaFunctionConfigurations\":[{\"Id\":\"$LAMBDA_FUNCTION_CONFIGURATION_ID\",\"LambdaFunctionArn\":\"arn:aws:lambda:$REGION:$ACCOUNT_ID:function:$LAMBDA_FN_NAME\",\"Events\":[\"s3:ReducedRedundancyLostObject\",\"s3:ObjectCreated:*\",\"s3:ObjectCreated:Put\",\"s3:ObjectCreated:Post\",\"s3:ObjectCreated:Copy\",\"s3:ObjectCreated:CompleteMultipartUpload\"]}]}"




