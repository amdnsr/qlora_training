
#!/bin/bash

# Specify the target directory

original_directory=$(pwd)

target_directory="/home/kokil/ahmad/benchmarking_llms/projects/qlora_training/"

# Change to the specified directory
cd "$target_directory" || exit 1  # Exit the script if the directory doesn't exist

# Get the base name of the shell script without the extension
script_basename=$(basename "$0" .sh)

# Run the Python script in the background with nohup
nohup /home/kokil/anaconda3/envs/qlora_conda_3912_env/bin/python /home/kokil/ahmad/benchmarking_llms/projects/qlora_training/train_individual_model_for_individual_dataset.py --dataset_name=restricted --dataset_home=/home/kokil/ahmad/benchmarking_llms/projects/qlora_training/create_dataset_jsons_aadish --model_name=llama --model_adapter_output_home=/home/kokil/ahmad/benchmarking_llms/projects/qlora_training/model_adapter_output_home > "$original_directory/${script_basename}.log" 2>&1 &

# Store the process ID (PID) in a separate file
echo $! > "$original_directory/${script_basename}.pid"
