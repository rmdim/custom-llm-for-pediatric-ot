import sys
import os
import subprocess
import json
import sys
import logging
from argparse import ArgumentParser
import tarfile

# From https://github.com/philschmid/deepspeed-sagemaker-example/blob/main/ds_launcher.py

logger = logging.getLogger(__name__)


def parse_args():
    parser = ArgumentParser(
        description=("SageMaker DeepSpeed Launch helper utility that will spawn deepspeed training scripts")
    )
    # positional
    parser.add_argument(
        "--training_script",
        type=str,
        help="Path to the training program/script to be run in parallel, can be either absolute or relative",
    )
    parser.add_argument('--input_model_dir', type=str, default=os.environ['SM_CHANNEL_MODEL'])

    # rest from the training program
    parsed, nargs = parser.parse_known_args()

    return parsed.training_script, parsed.input_model_dir, nargs


def main():
    # https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/launcher/launch.py
    num_gpus = int(os.environ.get("SM_NUM_GPUS", 0))
    hosts = json.loads(os.environ.get("SM_HOSTS", "{}"))
    num_nodes = len(hosts)
    current_host = os.environ.get("SM_CURRENT_HOST", 0)
    rank = hosts.index(current_host)
    print(f"num_gpus = {num_gpus}, num_nodes = {num_nodes}, current_host = {current_host}, rank = {rank}")


    # os.environ['NCCL_DEBUG'] = 'INFO'

    # get number of GPU
    # if num_gpus == 0:
    #     raise ValueError("No GPUs found.")

    train_script, input_model_dir, args = parse_args()

    print(f"Extracting files from {input_model_dir}")
    with tarfile.open(input_model_dir + "/model.tar.gz") as tar:
        tar.extractall(input_model_dir)
        print("EXTRACTED")
    
    # print(f"{os.listdir(args.input_model_dir)}")

    command = f"deepspeed --num_gpus={num_gpus} {train_script} {' '.join(args)}"
    print(f"command = {command}")
    # launch deepspeed training
    deepspeed_launch(command)


def deepspeed_launch(command):
    # try:
    try:
        subprocess.run(command, shell=True)
    except Exception as e:
        logger.info(e)


if __name__ == "__main__":
    main()
