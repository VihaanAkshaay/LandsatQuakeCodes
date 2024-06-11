import subprocess

# Define the different tags you want to run
tags = ['3band', '4band', '5band', '6band']

# Iterate over each tag and run the script with the corresponding argument
for tag in tags:
    print(f"Running script with tag: {tag}")
    result = subprocess.run(['python', 'RCNN_seven_band_train.py', '--tag', tag], capture_output=True, text=True)
    print(f"Output for tag {tag}:\n{result.stdout}")
    if result.stderr:
        print(f"Errors for tag {tag}:\n{result.stderr}")