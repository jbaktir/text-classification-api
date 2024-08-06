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