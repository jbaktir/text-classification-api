# AWS Lambda Function with API Gateway

This project demonstrates how to create and deploy an AWS Lambda function using Docker. The Lambda function is served via API Gateway. The function is designed for document classification, utilizing a pre-trained machine learning model.

## Prerequisites

- AWS CLI
- Docker
- AWS Account

## Project Structure

```
.
├── Dockerfile
├── lambda_function.py
├── document_classification_model.pkl
├── label_mappings.pkl
├── requirements.txt
├── README.md
├── update_lambda_function.py
├── test_lambda_function.py
├── test_api_gateway.py
├── example_events.py
├── helper.py
├── train.py
└── with_sentence_transformers
    └── train.py
```

## Dockerfile

The Dockerfile sets up the environment, installs dependencies, and specifies the handler for the Lambda function.

```Dockerfile
FROM public.ecr.aws/lambda/python:3.9

# Install system dependencies
RUN yum update -y && \
    yum install -y gcc gcc-c++ make

# Copy function code and model files
COPY lambda_function.py ${LAMBDA_TASK_ROOT}
COPY document_classification_model.pkl ${LAMBDA_TASK_ROOT}
COPY label_mappings.pkl ${LAMBDA_TASK_ROOT}

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Set the CMD to your handler
CMD [ "lambda_function.lambda_handler" ]
```

## Steps to Build and Deploy

### 1. Build the Docker Image

Navigate to the project directory and build the Docker image:

```sh
docker build -t my-lambda-image .
```

### 2. Test the Docker Image Locally (Optional)

Run the Docker container locally to test the Lambda function:

```sh
docker run -p 9000:8080 my-lambda-image
```

Invoke the function locally:

```sh
curl -XPOST "http://localhost:9000/2015-03-31/functions/function/invocations" -d '{}'
```

### 3. Push the Docker Image to Amazon ECR

Create a repository in Amazon ECR and push the Docker image:

```sh
# Authenticate Docker to your ECR registry
aws ecr get-login-password --region <your-region> | docker login --username AWS --password-stdin <your-account-id>.dkr.ecr.<your-region>.amazonaws.com

# Tag the image
docker tag my-lambda-image:latest <your-account-id>.dkr.ecr.<your-region>.amazonaws.com/my-lambda-repo:latest

# Push the image
docker push <your-account-id>.dkr.ecr.<your-region>.amazonaws.com/my-lambda-repo:latest
```

### 4. Create the Lambda Function

Create the Lambda function using the pushed Docker image:

```sh
aws lambda create-function \
    --function-name my-document-classifier \
    --package-type Image \
    --code ImageUri=<your-account-id>.dkr.ecr.<your-region>.amazonaws.com/my-lambda-repo:latest \
    --role arn:aws:iam::<your-account-id>:role/<your-lambda-execution-role>
```

### 5. Set Up API Gateway

Create an API Gateway and configure it to invoke your Lambda function. Detailed steps can be found in the [AWS documentation](https://docs.aws.amazon.com/apigateway/latest/developerguide/http-api-develop-integrations-lambda.html).

## Lambda Function Handler

The Lambda function is defined in `lambda_function.py` and is set to use the `lambda_handler` function as the entry point.

```python
import json
import pickle

def lambda_handler(event, context):
    # Load the model and mappings
    with open('document_classification_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    
    with open('label_mappings.pkl', 'rb') as mappings_file:
        label_mappings = pickle.load(mappings_file)

    # Process the input event
    document = event['document']
    prediction = model.predict([document])
    label = label_mappings[prediction[0]]
    
    return {
        'statusCode': 200,
        'body': json.dumps({'label': label})
    }
```

## Explanation of Additional Files

### update_lambda_function.py

This script automates the process of updating the Lambda function code. It packages the necessary files into a zip file and updates the Lambda function.

### test_lambda_function.py

This script is used for testing the Lambda function locally by invoking it with predefined events.

### test_api_gateway.py

This script tests the API Gateway by sending a request to the endpoint and printing the response.

### example_events.py

This file contains example events that can be used to test the Lambda function.

### helper.py

This script includes helper functions to process documents, generate embeddings, and save the processed data.

### train.py

This script trains the machine learning model using the processed data. It utilizes LightGBM and Optuna for hyperparameter optimization.

### with_sentence_transformers/train.py

This script provides an alternative training approach using Sentence Transformers to generate embeddings for the documents. It also trains a LightGBM model with the generated embeddings.

---
training logs:
```angular2html
[I 2024-08-06 00:37:08,015] A new study created in memory with name: no-name-75b1da14-9772-4708-b8fb-d976034a0aab
[I 2024-08-06 00:37:13,673] Trial 0 finished with value: 0.95 and parameters: {'boosting_type': 'dart', 'num_leaves': 90, 'learning_rate': 0.022261802625329435, 'feature_fraction': 0.8924798358143056, 'bagging_fraction': 0.3583439859123016, 'bagging_freq': 1, 'min_child_samples': 21, 'lambda_l1': 0.18975394940288765, 'lambda_l2': 0.0009006147586226727, 'min_split_gain': 0.16086178059932363, 'max_depth': 10, 'num_round': 297}. Best is trial 0 with value: 0.95.
[I 2024-08-06 00:37:16,996] Trial 1 finished with value: 0.89 and parameters: {'boosting_type': 'dart', 'num_leaves': 18, 'learning_rate': 0.015256841693984155, 'feature_fraction': 0.6648832090326127, 'bagging_fraction': 0.4216401879326943, 'bagging_freq': 7, 'min_child_samples': 11, 'lambda_l1': 0.2950150524421981, 'lambda_l2': 0.7084932477895242, 'min_split_gain': 1.3063829300810152e-05, 'max_depth': 10, 'num_round': 158}. Best is trial 0 with value: 0.95.
[I 2024-08-06 00:37:18,635] Trial 2 finished with value: 0.815 and parameters: {'boosting_type': 'dart', 'num_leaves': 28, 'learning_rate': 0.004825523135816073, 'feature_fraction': 0.49564463582153157, 'bagging_fraction': 0.6672697574014519, 'bagging_freq': 1, 'min_child_samples': 22, 'lambda_l1': 0.015287511974056443, 'lambda_l2': 8.542856291869436, 'min_split_gain': 0.025515297196292776, 'max_depth': 4, 'num_round': 117}. Best is trial 0 with value: 0.95.
[I 2024-08-06 00:37:23,015] Trial 3 finished with value: 0.85 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 184, 'learning_rate': 0.003258295879181298, 'feature_fraction': 0.5653160332412555, 'bagging_fraction': 0.541788589118877, 'bagging_freq': 4, 'min_child_samples': 45, 'lambda_l1': 0.015252611168754473, 'lambda_l2': 6.917563504628225e-08, 'min_split_gain': 0.00011352481611905354, 'max_depth': 6, 'num_round': 300}. Best is trial 0 with value: 0.95.
[I 2024-08-06 00:37:27,240] Trial 4 finished with value: 0.8 and parameters: {'boosting_type': 'dart', 'num_leaves': 105, 'learning_rate': 0.008177768900967013, 'feature_fraction': 0.6635219980528978, 'bagging_fraction': 0.728625026021048, 'bagging_freq': 2, 'min_child_samples': 79, 'lambda_l1': 0.0037741325046994506, 'lambda_l2': 3.5010946466887954e-07, 'min_split_gain': 0.17718821477805782, 'max_depth': 6, 'num_round': 274}. Best is trial 0 with value: 0.95.
[I 2024-08-06 00:37:29,174] Trial 5 finished with value: 0.96 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 16, 'learning_rate': 0.056395261310963625, 'feature_fraction': 0.6580301151237983, 'bagging_fraction': 0.8052190754851812, 'bagging_freq': 5, 'min_child_samples': 73, 'lambda_l1': 0.000765299819734838, 'lambda_l2': 0.006237606493777741, 'min_split_gain': 0.34190736231831476, 'max_depth': 5, 'num_round': 297}. Best is trial 5 with value: 0.96.
[I 2024-08-06 00:37:29,278] Trial 6 finished with value: 0.065 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 62, 'learning_rate': 0.008345917001604006, 'feature_fraction': 0.7002919142462868, 'bagging_fraction': 0.13947490316033592, 'bagging_freq': 5, 'min_child_samples': 95, 'lambda_l1': 0.0007832752206004494, 'lambda_l2': 1.0673526567440548e-08, 'min_split_gain': 0.0004097472172123862, 'max_depth': 9, 'num_round': 221}. Best is trial 5 with value: 0.96.
[I 2024-08-06 00:37:29,996] Trial 7 finished with value: 0.925 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 118, 'learning_rate': 0.06493006818824563, 'feature_fraction': 0.2971524669145053, 'bagging_fraction': 0.496869728885202, 'bagging_freq': 3, 'min_child_samples': 15, 'lambda_l1': 0.20330087754039425, 'lambda_l2': 0.16159738073495863, 'min_split_gain': 0.00042068464210472215, 'max_depth': 5, 'num_round': 55}. Best is trial 5 with value: 0.96.
[I 2024-08-06 00:37:35,124] Trial 8 finished with value: 0.715 and parameters: {'boosting_type': 'dart', 'num_leaves': 141, 'learning_rate': 0.0014275338969386988, 'feature_fraction': 0.6077714849742816, 'bagging_fraction': 0.8915509287799176, 'bagging_freq': 4, 'min_child_samples': 46, 'lambda_l1': 1.7405926755043245e-07, 'lambda_l2': 2.189370709021638e-08, 'min_split_gain': 3.2541669147046757e-06, 'max_depth': 10, 'num_round': 176}. Best is trial 5 with value: 0.96.
[I 2024-08-06 00:37:44,918] Trial 9 finished with value: 0.94 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 80, 'learning_rate': 0.025079615319304593, 'feature_fraction': 0.9541440161517842, 'bagging_fraction': 0.9932967294939983, 'bagging_freq': 1, 'min_child_samples': 83, 'lambda_l1': 6.519163395635203e-05, 'lambda_l2': 7.939420812991562e-05, 'min_split_gain': 0.0017125113283977648, 'max_depth': 7, 'num_round': 298}. Best is trial 5 with value: 0.96.
[I 2024-08-06 00:37:45,952] Trial 10 finished with value: 0.985 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 49, 'learning_rate': 0.0876526001668165, 'feature_fraction': 0.10362638157198034, 'bagging_fraction': 0.7841097840513762, 'bagging_freq': 6, 'min_child_samples': 64, 'lambda_l1': 1.146887410013211e-05, 'lambda_l2': 0.0016393633149801889, 'min_split_gain': 1.562830728959768e-08, 'max_depth': 3, 'num_round': 243}. Best is trial 10 with value: 0.985.
[I 2024-08-06 00:37:47,471] Trial 11 finished with value: 0.975 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 47, 'learning_rate': 0.0887164495984591, 'feature_fraction': 0.20660718687118063, 'bagging_fraction': 0.794094211051867, 'bagging_freq': 6, 'min_child_samples': 67, 'lambda_l1': 4.393013711364684e-06, 'lambda_l2': 0.001436339897697084, 'min_split_gain': 2.8372887386542346e-08, 'max_depth': 3, 'num_round': 249}. Best is trial 10 with value: 0.985.
[I 2024-08-06 00:37:48,381] Trial 12 finished with value: 0.975 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 53, 'learning_rate': 0.09320331312575127, 'feature_fraction': 0.10946387113629946, 'bagging_fraction': 0.6707290610879353, 'bagging_freq': 7, 'min_child_samples': 62, 'lambda_l1': 2.4580830444234052e-06, 'lambda_l2': 2.924663617260209e-05, 'min_split_gain': 1.4743192854157676e-08, 'max_depth': 3, 'num_round': 228}. Best is trial 10 with value: 0.985.
[I 2024-08-06 00:37:49,484] Trial 13 finished with value: 0.975 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 54, 'learning_rate': 0.0525809830433975, 'feature_fraction': 0.11182784305363254, 'bagging_fraction': 0.8407115397930984, 'bagging_freq': 6, 'min_child_samples': 59, 'lambda_l1': 1.1692700734427504e-05, 'lambda_l2': 4.141483166045951e-06, 'min_split_gain': 1.3548518519029248e-08, 'max_depth': 3, 'num_round': 235}. Best is trial 10 with value: 0.985.
[I 2024-08-06 00:37:53,030] Trial 14 finished with value: 0.955 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 44, 'learning_rate': 0.029284066460518236, 'feature_fraction': 0.31613036856045323, 'bagging_fraction': 0.9843262220740608, 'bagging_freq': 6, 'min_child_samples': 37, 'lambda_l1': 1.0118196427759569e-08, 'lambda_l2': 0.006798706926054802, 'min_split_gain': 2.1006140899432826e-07, 'max_depth': 12, 'num_round': 193}. Best is trial 10 with value: 0.985.
[I 2024-08-06 00:37:54,628] Trial 15 finished with value: 0.99 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 138, 'learning_rate': 0.08739729646931178, 'feature_fraction': 0.25985969693554833, 'bagging_fraction': 0.721684882057917, 'bagging_freq': 6, 'min_child_samples': 68, 'lambda_l1': 9.644840590681516e-07, 'lambda_l2': 0.0023541551369483248, 'min_split_gain': 2.88954751460976e-07, 'max_depth': 3, 'num_round': 253}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:37:58,197] Trial 16 finished with value: 0.975 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 149, 'learning_rate': 0.04034380503250955, 'feature_fraction': 0.41938530365759297, 'bagging_fraction': 0.6134758102650758, 'bagging_freq': 5, 'min_child_samples': 32, 'lambda_l1': 1.720530511355029e-07, 'lambda_l2': 0.14008023316605467, 'min_split_gain': 3.671341458364894e-07, 'max_depth': 4, 'num_round': 255}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:37:58,807] Trial 17 finished with value: 0.875 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 198, 'learning_rate': 0.015438752242788352, 'feature_fraction': 0.25184736488817944, 'bagging_fraction': 0.2879143850933273, 'bagging_freq': 7, 'min_child_samples': 98, 'lambda_l1': 3.014093784969566e-07, 'lambda_l2': 0.01894020885136828, 'min_split_gain': 3.4981031135973124e-07, 'max_depth': 8, 'num_round': 211}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:38:00,976] Trial 18 finished with value: 0.96 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 159, 'learning_rate': 0.037142949773201754, 'feature_fraction': 0.40643579198683033, 'bagging_fraction': 0.7332243934409909, 'bagging_freq': 6, 'min_child_samples': 54, 'lambda_l1': 6.99246079548934e-05, 'lambda_l2': 8.363959200494318e-06, 'min_split_gain': 6.130990262747941e-06, 'max_depth': 4, 'num_round': 146}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:38:02,957] Trial 19 finished with value: 0.775 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 127, 'learning_rate': 0.0010762985150620972, 'feature_fraction': 0.18969647637134573, 'bagging_fraction': 0.8975781909251729, 'bagging_freq': 3, 'min_child_samples': 86, 'lambda_l1': 1.1247565572405157e-08, 'lambda_l2': 0.00016640794253409795, 'min_split_gain': 1.4254559335963755e-07, 'max_depth': 5, 'num_round': 261}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:38:03,756] Trial 20 finished with value: 0.94 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 83, 'learning_rate': 0.09865430282495326, 'feature_fraction': 0.3640685412435578, 'bagging_fraction': 0.6214692750339498, 'bagging_freq': 5, 'min_child_samples': 70, 'lambda_l1': 4.205355754198212, 'lambda_l2': 7.393553096452127e-07, 'min_split_gain': 1.769832934058804e-06, 'max_depth': 7, 'num_round': 210}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:38:05,427] Trial 21 finished with value: 0.97 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 32, 'learning_rate': 0.06771613921568241, 'feature_fraction': 0.2184147786567568, 'bagging_fraction': 0.7749663626980979, 'bagging_freq': 6, 'min_child_samples': 66, 'lambda_l1': 1.874491392435931e-06, 'lambda_l2': 0.0010093523063932526, 'min_split_gain': 5.354222733228511e-08, 'max_depth': 3, 'num_round': 249}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:38:06,595] Trial 22 finished with value: 0.975 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 171, 'learning_rate': 0.09691952322438226, 'feature_fraction': 0.17393978956465928, 'bagging_fraction': 0.8818276875565858, 'bagging_freq': 6, 'min_child_samples': 75, 'lambda_l1': 8.877114216110592e-06, 'lambda_l2': 0.0011305321597709754, 'min_split_gain': 1.0056326576029617e-08, 'max_depth': 3, 'num_round': 245}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:38:07,907] Trial 23 finished with value: 0.98 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 72, 'learning_rate': 0.044160632418556584, 'feature_fraction': 0.1096814509791578, 'bagging_fraction': 0.7635222007181732, 'bagging_freq': 7, 'min_child_samples': 60, 'lambda_l1': 6.0702425966658555e-05, 'lambda_l2': 0.028813050098820306, 'min_split_gain': 6.936150415234702e-08, 'max_depth': 4, 'num_round': 272}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:38:09,177] Trial 24 finished with value: 0.98 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 67, 'learning_rate': 0.043756674147245316, 'feature_fraction': 0.10487411659016858, 'bagging_fraction': 0.6957913352258971, 'bagging_freq': 7, 'min_child_samples': 56, 'lambda_l1': 6.276653834397139e-05, 'lambda_l2': 0.037680813267603586, 'min_split_gain': 1.0739628772873999e-06, 'max_depth': 4, 'num_round': 273}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:38:11,499] Trial 25 finished with value: 0.955 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 104, 'learning_rate': 0.01649904225787031, 'feature_fraction': 0.2781536785861569, 'bagging_fraction': 0.6123043358769937, 'bagging_freq': 7, 'min_child_samples': 44, 'lambda_l1': 3.2057616827263047e-07, 'lambda_l2': 1.521656103666069, 'min_split_gain': 2.8673581203135557e-05, 'max_depth': 6, 'num_round': 281}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:38:14,055] Trial 26 finished with value: 0.885 and parameters: {'boosting_type': 'dart', 'num_leaves': 72, 'learning_rate': 0.032287144977269235, 'feature_fraction': 0.8063253539557955, 'bagging_fraction': 0.49966849290267334, 'bagging_freq': 4, 'min_child_samples': 89, 'lambda_l1': 0.00020235578564517935, 'lambda_l2': 0.010084367492725933, 'min_split_gain': 6.028456470840883e-08, 'max_depth': 5, 'num_round': 190}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:38:14,867] Trial 27 finished with value: 0.97 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 126, 'learning_rate': 0.06597641611822866, 'feature_fraction': 0.1718778088381393, 'bagging_fraction': 0.9397475026623028, 'bagging_freq': 7, 'min_child_samples': 51, 'lambda_l1': 2.0717493015593823e-05, 'lambda_l2': 0.0657240507425014, 'min_split_gain': 8.039498868601881e-07, 'max_depth': 4, 'num_round': 98}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:38:17,578] Trial 28 finished with value: 0.97 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 94, 'learning_rate': 0.055349591743101605, 'feature_fraction': 0.47655358376678647, 'bagging_fraction': 0.7585405206304445, 'bagging_freq': 5, 'min_child_samples': 63, 'lambda_l1': 9.4146548185273e-07, 'lambda_l2': 0.5160320647342078, 'min_split_gain': 7.850117983030772e-08, 'max_depth': 3, 'num_round': 272}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:38:21,168] Trial 29 finished with value: 0.945 and parameters: {'boosting_type': 'dart', 'num_leaves': 94, 'learning_rate': 0.02086625927586767, 'feature_fraction': 0.3376486937115972, 'bagging_fraction': 0.8352764336447598, 'bagging_freq': 6, 'min_child_samples': 36, 'lambda_l1': 7.104196163044919e-08, 'lambda_l2': 0.00033972294444240214, 'min_split_gain': 0.009637920241116073, 'max_depth': 12, 'num_round': 202}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:38:21,803] Trial 30 finished with value: 0.915 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 117, 'learning_rate': 0.010921143546723435, 'feature_fraction': 0.15803248760109284, 'bagging_fraction': 0.387848201140003, 'bagging_freq': 7, 'min_child_samples': 78, 'lambda_l1': 0.00027688091042299927, 'lambda_l2': 0.004537932969653517, 'min_split_gain': 7.40977111348031e-07, 'max_depth': 4, 'num_round': 233}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:38:23,108] Trial 31 finished with value: 0.975 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 68, 'learning_rate': 0.04687899257602219, 'feature_fraction': 0.11813673343777743, 'bagging_fraction': 0.7017485655135979, 'bagging_freq': 7, 'min_child_samples': 56, 'lambda_l1': 4.4224924483921466e-05, 'lambda_l2': 0.061557424881310885, 'min_split_gain': 1.4867123941825435e-07, 'max_depth': 4, 'num_round': 277}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:38:24,970] Trial 32 finished with value: 0.975 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 39, 'learning_rate': 0.07308311095030724, 'feature_fraction': 0.24599180552146657, 'bagging_fraction': 0.6676802450122364, 'bagging_freq': 7, 'min_child_samples': 52, 'lambda_l1': 9.941233994952807e-07, 'lambda_l2': 0.04287578261176574, 'min_split_gain': 1.6488931084041277e-06, 'max_depth': 5, 'num_round': 265}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:38:26,101] Trial 33 finished with value: 0.975 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 63, 'learning_rate': 0.04177996808355707, 'feature_fraction': 0.10293903934451956, 'bagging_fraction': 0.5886427614708616, 'bagging_freq': 6, 'min_child_samples': 59, 'lambda_l1': 0.0017116080322305493, 'lambda_l2': 0.0024062933641852114, 'min_split_gain': 2.0764209082230204e-05, 'max_depth': 4, 'num_round': 286}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:38:27,469] Trial 34 finished with value: 0.925 and parameters: {'boosting_type': 'dart', 'num_leaves': 80, 'learning_rate': 0.026563672944191184, 'feature_fraction': 0.16499359452642046, 'bagging_fraction': 0.721141855443989, 'bagging_freq': 7, 'min_child_samples': 69, 'lambda_l1': 0.00015043051220889437, 'lambda_l2': 4.4953506123066065, 'min_split_gain': 4.2333783406739544e-08, 'max_depth': 3, 'num_round': 239}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:38:30,067] Trial 35 finished with value: 0.92 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 24, 'learning_rate': 0.0031641829813318545, 'feature_fraction': 0.23385913589812674, 'bagging_fraction': 0.8217172254012082, 'bagging_freq': 7, 'min_child_samples': 47, 'lambda_l1': 0.025834435222060952, 'lambda_l2': 0.02517535645075558, 'min_split_gain': 7.611412887654848e-06, 'max_depth': 6, 'num_round': 261}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:38:31,448] Trial 36 finished with value: 0.955 and parameters: {'boosting_type': 'dart', 'num_leaves': 105, 'learning_rate': 0.04578516817142092, 'feature_fraction': 0.16267471137288952, 'bagging_fraction': 0.5468683650951838, 'bagging_freq': 6, 'min_child_samples': 25, 'lambda_l1': 1.5144325824994138e-05, 'lambda_l2': 0.000233764316044516, 'min_split_gain': 4.6317370805247813e-07, 'max_depth': 4, 'num_round': 150}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:38:33,528] Trial 37 finished with value: 0.98 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 55, 'learning_rate': 0.07445032926466205, 'feature_fraction': 0.3717421144094991, 'bagging_fraction': 0.6800900938565069, 'bagging_freq': 5, 'min_child_samples': 73, 'lambda_l1': 0.0023648569512140388, 'lambda_l2': 0.6649634961669916, 'min_split_gain': 9.656984454350695e-05, 'max_depth': 5, 'num_round': 287}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:38:35,230] Trial 38 finished with value: 0.975 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 14, 'learning_rate': 0.020349350093573096, 'feature_fraction': 0.12712961927335423, 'bagging_fraction': 0.7716112684964893, 'bagging_freq': 2, 'min_child_samples': 41, 'lambda_l1': 4.667607906509569e-06, 'lambda_l2': 0.000453100485708151, 'min_split_gain': 1.0525644148938277e-07, 'max_depth': 6, 'num_round': 222}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:38:36,706] Trial 39 finished with value: 0.9 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 31, 'learning_rate': 0.005704303322019908, 'feature_fraction': 0.2811439119348535, 'bagging_fraction': 0.46833252166036915, 'bagging_freq': 5, 'min_child_samples': 63, 'lambda_l1': 2.0948774958968155e-05, 'lambda_l2': 0.2252545737099369, 'min_split_gain': 3.0208566561870375e-08, 'max_depth': 3, 'num_round': 270}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:38:38,475] Trial 40 finished with value: 0.775 and parameters: {'boosting_type': 'dart', 'num_leaves': 88, 'learning_rate': 0.011050464229379206, 'feature_fraction': 0.5211926560952952, 'bagging_fraction': 0.2983064756082347, 'bagging_freq': 4, 'min_child_samples': 78, 'lambda_l1': 0.00681124064723356, 'lambda_l2': 5.8410527007478e-05, 'min_split_gain': 1.6723625732303025e-06, 'max_depth': 11, 'num_round': 293}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:38:40,436] Trial 41 finished with value: 0.975 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 55, 'learning_rate': 0.07741486818727553, 'feature_fraction': 0.3666556305657695, 'bagging_fraction': 0.6742198111099212, 'bagging_freq': 5, 'min_child_samples': 72, 'lambda_l1': 0.000911863540347208, 'lambda_l2': 1.717265266770882, 'min_split_gain': 0.00010097383364169548, 'max_depth': 5, 'num_round': 285}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:38:42,438] Trial 42 finished with value: 0.97 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 68, 'learning_rate': 0.05924323686485729, 'feature_fraction': 0.45321679569395795, 'bagging_fraction': 0.7057445263606723, 'bagging_freq': 6, 'min_child_samples': 82, 'lambda_l1': 0.03599437196857214, 'lambda_l2': 0.39252823006219173, 'min_split_gain': 0.04975679458640662, 'max_depth': 5, 'num_round': 300}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:38:44,565] Trial 43 finished with value: 0.97 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 39, 'learning_rate': 0.08002280109513611, 'feature_fraction': 0.21402826743462836, 'bagging_fraction': 0.5767378682501105, 'bagging_freq': 6, 'min_child_samples': 5, 'lambda_l1': 9.718614335470239e-05, 'lambda_l2': 0.012697567767287924, 'min_split_gain': 0.0003361688397727592, 'max_depth': 4, 'num_round': 255}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:38:46,599] Trial 44 finished with value: 0.965 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 72, 'learning_rate': 0.035045667037346115, 'feature_fraction': 0.2596612438385911, 'bagging_fraction': 0.8495621228040693, 'bagging_freq': 7, 'min_child_samples': 75, 'lambda_l1': 0.0022932721373596933, 'lambda_l2': 0.0026972814046656067, 'min_split_gain': 4.224404161942561e-05, 'max_depth': 3, 'num_round': 282}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:38:49,460] Trial 45 finished with value: 0.98 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 58, 'learning_rate': 0.051274634184281055, 'feature_fraction': 0.6059377839520499, 'bagging_fraction': 0.6447735670583958, 'bagging_freq': 5, 'min_child_samples': 58, 'lambda_l1': 0.0006135175403555318, 'lambda_l2': 1.4334525171207995, 'min_split_gain': 0.003636281737235767, 'max_depth': 4, 'num_round': 269}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:38:51,322] Trial 46 finished with value: 0.97 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 48, 'learning_rate': 0.07826459113707535, 'feature_fraction': 0.31577524205694674, 'bagging_fraction': 0.7437461203767084, 'bagging_freq': 5, 'min_child_samples': 50, 'lambda_l1': 0.006579657392611472, 'lambda_l2': 8.285966545892476, 'min_split_gain': 4.251236322767424e-06, 'max_depth': 6, 'num_round': 242}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:38:52,305] Trial 47 finished with value: 0.97 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 144, 'learning_rate': 0.058607618944813464, 'feature_fraction': 0.14933847114946464, 'bagging_fraction': 0.8000496537642222, 'bagging_freq': 6, 'min_child_samples': 65, 'lambda_l1': 0.0004350379674958499, 'lambda_l2': 0.09779678070037706, 'min_split_gain': 2.33330948841317e-08, 'max_depth': 5, 'num_round': 171}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:38:53,155] Trial 48 finished with value: 0.96 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 24, 'learning_rate': 0.04263229978298984, 'feature_fraction': 0.19922620360907708, 'bagging_fraction': 0.7017568881081802, 'bagging_freq': 7, 'min_child_samples': 68, 'lambda_l1': 3.6570564146836604e-05, 'lambda_l2': 0.03211728496426304, 'min_split_gain': 2.3141308782115657e-07, 'max_depth': 8, 'num_round': 126}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:38:55,852] Trial 49 finished with value: 0.86 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 99, 'learning_rate': 0.0027465950002355387, 'feature_fraction': 0.3925009863470253, 'bagging_fraction': 0.9451799206414008, 'bagging_freq': 4, 'min_child_samples': 59, 'lambda_l1': 0.08177299181927467, 'lambda_l2': 0.26374202556702325, 'min_split_gain': 0.0006168464644053488, 'max_depth': 3, 'num_round': 288}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:38:56,552] Trial 50 finished with value: 0.95 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 77, 'learning_rate': 0.03237869643688715, 'feature_fraction': 0.10000709695961911, 'bagging_fraction': 0.6499248529393241, 'bagging_freq': 6, 'min_child_samples': 74, 'lambda_l1': 0.6475319201491052, 'lambda_l2': 0.005304274333941016, 'min_split_gain': 9.133320261973242e-07, 'max_depth': 4, 'num_round': 229}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:38:59,448] Trial 51 finished with value: 0.975 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 58, 'learning_rate': 0.05046668918616435, 'feature_fraction': 0.586393755302456, 'bagging_fraction': 0.6421944848569585, 'bagging_freq': 5, 'min_child_samples': 56, 'lambda_l1': 0.0006435186765411657, 'lambda_l2': 1.3668902445724076, 'min_split_gain': 0.009088236083925299, 'max_depth': 4, 'num_round': 270}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:39:02,207] Trial 52 finished with value: 0.98 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 47, 'learning_rate': 0.08596181394738148, 'feature_fraction': 0.620285202405807, 'bagging_fraction': 0.6827409095865686, 'bagging_freq': 5, 'min_child_samples': 60, 'lambda_l1': 5.944877533436768e-06, 'lambda_l2': 4.1482549079773765, 'min_split_gain': 0.0020948116428091434, 'max_depth': 4, 'num_round': 257}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:39:02,745] Trial 53 finished with value: 0.945 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 59, 'learning_rate': 0.06589052672171951, 'feature_fraction': 0.8137444041293164, 'bagging_fraction': 0.14053015983430367, 'bagging_freq': 4, 'min_child_samples': 55, 'lambda_l1': 0.0012813239803914974, 'lambda_l2': 0.9749256551020302, 'min_split_gain': 0.0024429454679396737, 'max_depth': 3, 'num_round': 250}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:39:03,422] Trial 54 finished with value: 0.91 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 40, 'learning_rate': 0.05202912068981058, 'feature_fraction': 0.55548987419148, 'bagging_fraction': 0.7560630495862867, 'bagging_freq': 3, 'min_child_samples': 91, 'lambda_l1': 2.410175828740192e-06, 'lambda_l2': 0.11362908899137103, 'min_split_gain': 0.00023966434894812541, 'max_depth': 3, 'num_round': 61}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:39:04,928] Trial 55 finished with value: 0.93 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 64, 'learning_rate': 0.0999899876916559, 'feature_fraction': 0.7283712415697288, 'bagging_fraction': 0.7845527205781275, 'bagging_freq': 6, 'min_child_samples': 71, 'lambda_l1': 0.00012014189303613063, 'lambda_l2': 0.0005907588021666864, 'min_split_gain': 0.7903841044569804, 'max_depth': 5, 'num_round': 274}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:39:09,103] Trial 56 finished with value: 0.95 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 160, 'learning_rate': 0.023240686928176317, 'feature_fraction': 0.5167892179342604, 'bagging_fraction': 0.8726783721672148, 'bagging_freq': 5, 'min_child_samples': 65, 'lambda_l1': 0.004772804617316178, 'lambda_l2': 0.017661825070917292, 'min_split_gain': 0.058759433611971404, 'max_depth': 7, 'num_round': 294}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:39:14,435] Trial 57 finished with value: 0.97 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 53, 'learning_rate': 0.03787234538204207, 'feature_fraction': 0.6745389155665342, 'bagging_fraction': 0.632351837487879, 'bagging_freq': 7, 'min_child_samples': 49, 'lambda_l1': 4.160014821700407e-05, 'lambda_l2': 0.0024368619773413136, 'min_split_gain': 1.4514593865738916e-08, 'max_depth': 9, 'num_round': 264}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:39:16,645] Trial 58 finished with value: 0.96 and parameters: {'boosting_type': 'dart', 'num_leaves': 116, 'learning_rate': 0.060869668869253726, 'feature_fraction': 0.4370056471804432, 'bagging_fraction': 0.5120473587801795, 'bagging_freq': 6, 'min_child_samples': 83, 'lambda_l1': 0.00043037132507931777, 'lambda_l2': 0.5614323780228953, 'min_split_gain': 1.0519399923965804e-05, 'max_depth': 3, 'num_round': 278}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:39:17,550] Trial 59 finished with value: 0.965 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 87, 'learning_rate': 0.08409820170022013, 'feature_fraction': 0.12948577845863415, 'bagging_fraction': 0.5770125982146005, 'bagging_freq': 5, 'min_child_samples': 62, 'lambda_l1': 7.087602578171362e-07, 'lambda_l2': 3.79396541846856, 'min_split_gain': 4.751289915279218e-05, 'max_depth': 5, 'num_round': 248}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:39:22,929] Trial 60 finished with value: 0.97 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 76, 'learning_rate': 0.07241965042592803, 'feature_fraction': 0.9877886382816607, 'bagging_fraction': 0.7353968495786033, 'bagging_freq': 7, 'min_child_samples': 42, 'lambda_l1': 7.831817441544397e-08, 'lambda_l2': 0.00010071135582653588, 'min_split_gain': 2.4945279133831053e-07, 'max_depth': 4, 'num_round': 210}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:39:25,822] Trial 61 finished with value: 0.985 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 47, 'learning_rate': 0.08915949400443796, 'feature_fraction': 0.6338082032611992, 'bagging_fraction': 0.6843996587624389, 'bagging_freq': 5, 'min_child_samples': 59, 'lambda_l1': 5.924196762291866e-06, 'lambda_l2': 3.612058002953299, 'min_split_gain': 0.002577858425906836, 'max_depth': 4, 'num_round': 257}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:39:28,197] Trial 62 finished with value: 0.97 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 46, 'learning_rate': 0.08990526982507784, 'feature_fraction': 0.702660134848329, 'bagging_fraction': 0.5971451976105118, 'bagging_freq': 4, 'min_child_samples': 58, 'lambda_l1': 6.856290967576635e-06, 'lambda_l2': 0.2601832471807364, 'min_split_gain': 0.005863423470654173, 'max_depth': 4, 'num_round': 222}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:39:31,116] Trial 63 finished with value: 0.98 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 51, 'learning_rate': 0.06903201823539111, 'feature_fraction': 0.6247703904509238, 'bagging_fraction': 0.6509853607349355, 'bagging_freq': 5, 'min_child_samples': 54, 'lambda_l1': 1.8797163411296228e-06, 'lambda_l2': 2.2874668833122858, 'min_split_gain': 0.0008136494337576379, 'max_depth': 3, 'num_round': 268}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:39:33,958] Trial 64 finished with value: 0.97 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 37, 'learning_rate': 0.04908467050676854, 'feature_fraction': 0.6542209810992623, 'bagging_fraction': 0.7033440734861417, 'bagging_freq': 6, 'min_child_samples': 67, 'lambda_l1': 1.8523380611087842e-05, 'lambda_l2': 0.7990345131814326, 'min_split_gain': 0.005389595276520186, 'max_depth': 4, 'num_round': 236}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:39:37,697] Trial 65 finished with value: 0.95 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 135, 'learning_rate': 0.027435676531693428, 'feature_fraction': 0.7906677257318528, 'bagging_fraction': 0.803002230563254, 'bagging_freq': 5, 'min_child_samples': 61, 'lambda_l1': 6.768532374341295e-05, 'lambda_l2': 7.127827666491408, 'min_split_gain': 0.0009705874498979649, 'max_depth': 4, 'num_round': 256}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:39:41,339] Trial 66 finished with value: 0.98 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 68, 'learning_rate': 0.056947692623805814, 'feature_fraction': 0.5929649224416218, 'bagging_fraction': 0.7297451176481069, 'bagging_freq': 6, 'min_child_samples': 65, 'lambda_l1': 0.0001836656023895939, 'lambda_l2': 0.008731850699726006, 'min_split_gain': 8.607376003097158e-08, 'max_depth': 3, 'num_round': 292}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:39:42,714] Trial 67 finished with value: 0.975 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 61, 'learning_rate': 0.09987433306379424, 'feature_fraction': 0.1919019599571065, 'bagging_fraction': 0.7642924354533016, 'bagging_freq': 4, 'min_child_samples': 70, 'lambda_l1': 4.6729819265079634e-07, 'lambda_l2': 0.07741724945564872, 'min_split_gain': 0.00022550862896133676, 'max_depth': 5, 'num_round': 279}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:39:47,824] Trial 68 finished with value: 0.945 and parameters: {'boosting_type': 'dart', 'num_leaves': 23, 'learning_rate': 0.042471168751470514, 'feature_fraction': 0.7412191985288219, 'bagging_fraction': 0.6765122542822909, 'bagging_freq': 7, 'min_child_samples': 47, 'lambda_l1': 3.3509407338080794e-06, 'lambda_l2': 0.04174391784673247, 'min_split_gain': 0.01873871342282581, 'max_depth': 4, 'num_round': 243}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:39:49,121] Trial 69 finished with value: 0.98 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 33, 'learning_rate': 0.06854965720053427, 'feature_fraction': 0.13542687398249173, 'bagging_fraction': 0.8223997502380394, 'bagging_freq': 5, 'min_child_samples': 52, 'lambda_l1': 9.563741142306453e-06, 'lambda_l2': 2.6161889236842097, 'min_split_gain': 3.088159226665661e-08, 'max_depth': 3, 'num_round': 265}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:39:50,601] Trial 70 finished with value: 0.945 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 177, 'learning_rate': 0.030368846954877342, 'feature_fraction': 0.22838462989721703, 'bagging_fraction': 0.6227479247694286, 'bagging_freq': 3, 'min_child_samples': 77, 'lambda_l1': 3.279095331327241e-05, 'lambda_l2': 0.0006565967664459663, 'min_split_gain': 3.2439877049958046e-06, 'max_depth': 4, 'num_round': 252}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:39:53,235] Trial 71 finished with value: 0.975 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 48, 'learning_rate': 0.08708118799950647, 'feature_fraction': 0.625437966997644, 'bagging_fraction': 0.6850702631518909, 'bagging_freq': 5, 'min_child_samples': 59, 'lambda_l1': 8.870745770512287e-06, 'lambda_l2': 8.331159222699588, 'min_split_gain': 0.002194095154430423, 'max_depth': 4, 'num_round': 259}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:39:55,904] Trial 72 finished with value: 0.98 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 198, 'learning_rate': 0.08508237124427027, 'feature_fraction': 0.5764877519861971, 'bagging_fraction': 0.7144941836692412, 'bagging_freq': 5, 'min_child_samples': 62, 'lambda_l1': 1.2188897460555899e-06, 'lambda_l2': 3.163957802999903, 'min_split_gain': 0.001800874354834291, 'max_depth': 5, 'num_round': 275}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:39:57,918] Trial 73 finished with value: 0.975 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 43, 'learning_rate': 0.07678183195086534, 'feature_fraction': 0.5425012667566231, 'bagging_fraction': 0.6842964512135521, 'bagging_freq': 5, 'min_child_samples': 56, 'lambda_l1': 4.85036617447086e-06, 'lambda_l2': 0.4090065690802888, 'min_split_gain': 0.01611988478289291, 'max_depth': 3, 'num_round': 260}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:40:00,074] Trial 74 finished with value: 0.975 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 53, 'learning_rate': 0.06162405637636677, 'feature_fraction': 0.6086923964818208, 'bagging_fraction': 0.52589844576994, 'bagging_freq': 6, 'min_child_samples': 68, 'lambda_l1': 0.0026440170940207897, 'lambda_l2': 0.9437607449266264, 'min_split_gain': 0.044653828448876505, 'max_depth': 4, 'num_round': 300}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:40:03,089] Trial 75 finished with value: 0.965 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 71, 'learning_rate': 0.05225977642122913, 'feature_fraction': 0.47883449526697963, 'bagging_fraction': 0.6502612460256999, 'bagging_freq': 5, 'min_child_samples': 60, 'lambda_l1': 0.0003178189784299662, 'lambda_l2': 0.0013377571272674752, 'min_split_gain': 0.004063316987843841, 'max_depth': 6, 'num_round': 231}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:40:06,337] Trial 76 finished with value: 0.975 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 56, 'learning_rate': 0.08923372293480264, 'feature_fraction': 0.6663686822929296, 'bagging_fraction': 0.5567581028174613, 'bagging_freq': 4, 'min_child_samples': 73, 'lambda_l1': 1.8878477317311072e-07, 'lambda_l2': 1.3400034453838286e-07, 'min_split_gain': 1.2195706697921687e-07, 'max_depth': 5, 'num_round': 286}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:40:07,321] Trial 77 finished with value: 0.96 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 63, 'learning_rate': 0.046305824486985066, 'feature_fraction': 0.14451648631183406, 'bagging_fraction': 0.8609033291371806, 'bagging_freq': 7, 'min_child_samples': 64, 'lambda_l1': 1.498903818840053e-05, 'lambda_l2': 0.17661802447521902, 'min_split_gain': 0.11700306525896348, 'max_depth': 4, 'num_round': 242}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:40:09,252] Trial 78 finished with value: 0.95 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 29, 'learning_rate': 0.03661476445408839, 'feature_fraction': 0.3435340911774256, 'bagging_fraction': 0.9024789281451154, 'bagging_freq': 1, 'min_child_samples': 81, 'lambda_l1': 7.707714327982386e-05, 'lambda_l2': 4.414740921778839, 'min_split_gain': 0.0013112237005147027, 'max_depth': 3, 'num_round': 268}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:40:14,676] Trial 79 finished with value: 0.96 and parameters: {'boosting_type': 'dart', 'num_leaves': 84, 'learning_rate': 0.06666093358664117, 'feature_fraction': 0.641397030483269, 'bagging_fraction': 0.7455333110942681, 'bagging_freq': 6, 'min_child_samples': 57, 'lambda_l1': 5.95699340315131e-06, 'lambda_l2': 0.0036370639198415496, 'min_split_gain': 5.8609343504038786e-08, 'max_depth': 4, 'num_round': 278}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:40:16,100] Trial 80 finished with value: 0.89 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 43, 'learning_rate': 0.005752451624503679, 'feature_fraction': 0.27131078259656916, 'bagging_fraction': 0.6069481806245345, 'bagging_freq': 6, 'min_child_samples': 54, 'lambda_l1': 0.013367700210618404, 'lambda_l2': 1.7687523057686585, 'min_split_gain': 5.010808277810788e-07, 'max_depth': 5, 'num_round': 217}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:40:18,975] Trial 81 finished with value: 0.98 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 51, 'learning_rate': 0.07374831800369613, 'feature_fraction': 0.6236935563488577, 'bagging_fraction': 0.657703714148499, 'bagging_freq': 5, 'min_child_samples': 53, 'lambda_l1': 1.644790034541554e-06, 'lambda_l2': 2.515720330807897, 'min_split_gain': 0.0007454124829968834, 'max_depth': 3, 'num_round': 270}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:40:22,082] Trial 82 finished with value: 0.965 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 35, 'learning_rate': 0.07092137221784348, 'feature_fraction': 0.6895763712231223, 'bagging_fraction': 0.6841743006840507, 'bagging_freq': 5, 'min_child_samples': 48, 'lambda_l1': 5.434219228725153e-07, 'lambda_l2': 4.936654018051468, 'min_split_gain': 0.0005012129809248569, 'max_depth': 3, 'num_round': 255}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:40:25,272] Trial 83 finished with value: 0.945 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 50, 'learning_rate': 0.05783429817361911, 'feature_fraction': 0.5415293428768426, 'bagging_fraction': 0.780052760558118, 'bagging_freq': 5, 'min_child_samples': 44, 'lambda_l1': 3.5409919055909247e-06, 'lambda_l2': 1.3192155455457026, 'min_split_gain': 0.0028837910099776417, 'max_depth': 3, 'num_round': 289}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:40:28,404] Trial 84 finished with value: 0.975 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 64, 'learning_rate': 0.08061317040304769, 'feature_fraction': 0.6072871658060125, 'bagging_fraction': 0.716382051452406, 'bagging_freq': 4, 'min_child_samples': 52, 'lambda_l1': 1.7931671623733289e-06, 'lambda_l2': 0.660186513345772, 'min_split_gain': 2.0295365194418554e-08, 'max_depth': 3, 'num_round': 265}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:40:29,784] Trial 85 finished with value: 0.98 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 76, 'learning_rate': 0.09094417173334149, 'feature_fraction': 0.183422167996948, 'bagging_fraction': 0.620723977547467, 'bagging_freq': 5, 'min_child_samples': 61, 'lambda_l1': 2.2512830185319616e-05, 'lambda_l2': 6.1862869951251355e-06, 'min_split_gain': 1.0282226320381651e-08, 'max_depth': 4, 'num_round': 246}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:40:31,816] Trial 86 finished with value: 0.965 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 19, 'learning_rate': 0.06345288096298202, 'feature_fraction': 0.5613219954245683, 'bagging_fraction': 0.45156526014332143, 'bagging_freq': 6, 'min_child_samples': 67, 'lambda_l1': 2.748554549981566e-06, 'lambda_l2': 2.0966173744726633, 'min_split_gain': 0.00018894560733530378, 'max_depth': 3, 'num_round': 283}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:40:33,990] Trial 87 finished with value: 0.98 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 43, 'learning_rate': 0.053212185064333326, 'feature_fraction': 0.2919308465869005, 'bagging_fraction': 0.8231586555845841, 'bagging_freq': 7, 'min_child_samples': 57, 'lambda_l1': 3.0835239963512215e-05, 'lambda_l2': 0.015022169507173595, 'min_split_gain': 0.0010597778938767639, 'max_depth': 4, 'num_round': 237}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:40:37,759] Trial 88 finished with value: 0.775 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 10, 'learning_rate': 0.001827539881276515, 'feature_fraction': 0.5139724413291434, 'bagging_fraction': 0.6493158339604438, 'bagging_freq': 5, 'min_child_samples': 71, 'lambda_l1': 0.0007025599812866143, 'lambda_l2': 2.882848716087372e-05, 'min_split_gain': 0.008224874883711092, 'max_depth': 4, 'num_round': 260}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:40:42,133] Trial 89 finished with value: 0.975 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 57, 'learning_rate': 0.07937248651485133, 'feature_fraction': 0.757622464050909, 'bagging_fraction': 0.7517075087658378, 'bagging_freq': 7, 'min_child_samples': 64, 'lambda_l1': 0.0001449340763021617, 'lambda_l2': 0.28920280210703064, 'min_split_gain': 1.7586355389946208e-07, 'max_depth': 3, 'num_round': 272}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:40:44,916] Trial 90 finished with value: 0.95 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 113, 'learning_rate': 0.04569098016050479, 'feature_fraction': 0.7031126964586036, 'bagging_fraction': 0.5663447639276411, 'bagging_freq': 6, 'min_child_samples': 76, 'lambda_l1': 0.0012843964802850892, 'lambda_l2': 6.213620834811804, 'min_split_gain': 4.2850997445056515e-08, 'max_depth': 5, 'num_round': 250}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:40:49,141] Trial 91 finished with value: 0.98 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 66, 'learning_rate': 0.056262098452195854, 'feature_fraction': 0.5848120400757163, 'bagging_fraction': 0.7200082704925146, 'bagging_freq': 6, 'min_child_samples': 65, 'lambda_l1': 6.251959457205693e-05, 'lambda_l2': 0.006848550467929245, 'min_split_gain': 2.9307903985220634e-07, 'max_depth': 3, 'num_round': 288}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:40:53,979] Trial 92 finished with value: 0.955 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 68, 'learning_rate': 0.013472606450754565, 'feature_fraction': 0.6103981983018552, 'bagging_fraction': 0.7352738594765126, 'bagging_freq': 6, 'min_child_samples': 60, 'lambda_l1': 0.00022238982387993205, 'lambda_l2': 0.010750644506336862, 'min_split_gain': 1.0580778453055177e-07, 'max_depth': 3, 'num_round': 295}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:40:55,350] Trial 93 finished with value: 0.97 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 94, 'learning_rate': 0.07180410425212695, 'feature_fraction': 0.11691679709258165, 'bagging_fraction': 0.694639375847869, 'bagging_freq': 6, 'min_child_samples': 69, 'lambda_l1': 0.0003432609713381999, 'lambda_l2': 0.0092440300861628, 'min_split_gain': 7.128137517425535e-08, 'max_depth': 4, 'num_round': 291}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:40:58,799] Trial 94 finished with value: 0.975 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 72, 'learning_rate': 0.09490448223121235, 'feature_fraction': 0.6398352849979383, 'bagging_fraction': 0.6655992267331592, 'bagging_freq': 5, 'min_child_samples': 55, 'lambda_l1': 0.0001782623984407397, 'lambda_l2': 0.001981930474949216, 'min_split_gain': 6.162627813653717e-05, 'max_depth': 3, 'num_round': 275}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:41:02,725] Trial 95 finished with value: 0.985 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 61, 'learning_rate': 0.06242143613838316, 'feature_fraction': 0.5982001618075298, 'bagging_fraction': 0.7324796715638533, 'bagging_freq': 2, 'min_child_samples': 63, 'lambda_l1': 5.0233051729363704e-05, 'lambda_l2': 0.029414201139989226, 'min_split_gain': 7.193503090180128e-07, 'max_depth': 3, 'num_round': 282}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:41:05,308] Trial 96 finished with value: 0.985 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 60, 'learning_rate': 0.06412479749536334, 'feature_fraction': 0.16241477124508694, 'bagging_fraction': 0.6386791644060781, 'bagging_freq': 2, 'min_child_samples': 63, 'lambda_l1': 5.3291526216407784e-05, 'lambda_l2': 0.024348908492749927, 'min_split_gain': 1.0686542616572931e-06, 'max_depth': 9, 'num_round': 267}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:41:08,322] Trial 97 finished with value: 0.945 and parameters: {'boosting_type': 'dart', 'num_leaves': 59, 'learning_rate': 0.038495858422285686, 'feature_fraction': 0.1577503066864474, 'bagging_fraction': 0.7934738410752278, 'bagging_freq': 2, 'min_child_samples': 63, 'lambda_l1': 4.6778848634689456e-05, 'lambda_l2': 0.02504246645155722, 'min_split_gain': 2.063335512745765e-06, 'max_depth': 8, 'num_round': 281}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:41:10,209] Trial 98 finished with value: 0.98 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 81, 'learning_rate': 0.06375318213624646, 'feature_fraction': 0.10134254660725006, 'bagging_fraction': 0.7694241548390148, 'bagging_freq': 3, 'min_child_samples': 50, 'lambda_l1': 1.362505259311782e-05, 'lambda_l2': 0.04995348782771506, 'min_split_gain': 1.0847486990135194e-06, 'max_depth': 7, 'num_round': 255}. Best is trial 15 with value: 0.99.
[I 2024-08-06 00:41:12,481] Trial 99 finished with value: 0.985 and parameters: {'boosting_type': 'gbdt', 'num_leaves': 157, 'learning_rate': 0.04799283233433215, 'feature_fraction': 0.17437639840635769, 'bagging_fraction': 0.6330973216539877, 'bagging_freq': 2, 'min_child_samples': 73, 'lambda_l1': 2.6505400675814316e-05, 'lambda_l2': 0.02694807261354032, 'min_split_gain': 5.516577380430795e-07, 'max_depth': 10, 'num_round': 265}. Best is trial 15 with value: 0.99.
Best hyperparameters:  {'boosting_type': 'gbdt', 'num_leaves': 138, 'learning_rate': 0.08739729646931178, 'feature_fraction': 0.25985969693554833, 'bagging_fraction': 0.721684882057917, 'bagging_freq': 6, 'min_child_samples': 68, 'lambda_l1': 9.644840590681516e-07, 'lambda_l2': 0.0023541551369483248, 'min_split_gain': 2.88954751460976e-07, 'max_depth': 3, 'num_round': 253}
Best accuracy:  0.99
Accuracy: 0.99
True Label: historical, Predicted Label: historical
True Label: food, Predicted Label: food
True Label: food, Predicted Label: food
True Label: medical, Predicted Label: medical
True Label: politics, Predicted Label: politics
True Label: medical, Predicted Label: medical
True Label: medical, Predicted Label: medical
True Label: historical, Predicted Label: historical
True Label: space, Predicted Label: space
True Label: entertainment, Predicted Label: entertainment
Top 10 important features:
feature_841: 87
feature_797: 78
feature_824: 64
feature_787: 60
feature_161: 58
feature_819: 54
feature_974: 52
feature_811: 49
feature_796: 45
feature_501: 43
```
