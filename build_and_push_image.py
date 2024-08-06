import subprocess
import boto3

def run_command(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    output, error = process.communicate()
    if error:
        print(f"Error: {error}")
    return output.decode('utf-8')

def build_and_push_image(repository_name, tag='latest'):
    account_id = boto3.client('sts').get_caller_identity().get('Account')
    region = boto3.session.Session().region_name

    print("Building Docker image...")
    run_command(f"docker build -t {repository_name}:{tag} .")

    print("Authenticating Docker to ECR...")
    auth_command = f"aws ecr get-login-password --region {region} | docker login --username AWS --password-stdin {account_id}.dkr.ecr.{region}.amazonaws.com"
    run_command(auth_command)

    ecr_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/{repository_name}:{tag}"
    print(f"Tagging image as: {ecr_uri}")
    run_command(f"docker tag {repository_name}:{tag} {ecr_uri}")

    print("Pushing image to ECR...")
    run_command(f"docker push {ecr_uri}")

    print("Image successfully built and pushed to ECR.")

if __name__ == "__main__":
    repository_name = 'document-classifier'
    build_and_push_image(repository_name)