import os

def print_py_files_contents(folder_path):
    # Iterate through all files and directories in the given folder
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # Check if the file has a .py extension
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                print(f"File: {file_path}")
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        contents = f.read()
                        print(contents)
                        print('-' * 80)  # Print a separator for better readability
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")

# Specify the folder path you want to iterate through
folder_path = '/Users/nurgulbaktir/Dropbox/trellis/text-classification-api'
print_py_files_contents(folder_path)
