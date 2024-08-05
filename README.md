conda create -n=text-classification python=3.8
conda activate text-classification  

/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
(echo; echo 'eval "$(/opt/homebrew/bin/brew shellenv)"') >> /Users/nurgulbaktir/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
source /Users/<username>/.zprofile
brew install libomp 



# Create a directory for your lightgbm layer
mkdir -p lightgbm_layer/python

# Install the lightgbm package into the layer directory without dependencies
pip install lightgbm==3.3.3 -t lightgbm_layer/python 

# Create a ZIP file of the layer contents
cd lightgbm_layer
zip -r ../lightgbm_layer.zip .
cd ..

# Upload the ZIP file to S3
aws s3 cp lightgbm_layer.zip s3://joebaktir-lambda-layers/lightgbm_layer.zip

# Publish the layer
aws lambda publish-layer-version \
    --layer-name "lightgbm" \
    --description "Lambda layer for lightgbm" \
    --license-info "MIT" \
    --content S3Bucket=joebaktir-lambda-layers,S3Key=lightgbm_layer.zip \
    --compatible-runtimes python3.8 python3.9 \
    --compatible-architectures "x86_64"
