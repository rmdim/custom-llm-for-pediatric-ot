import torch
import transformers
from transformers import  TrainingArguments, HfArgumentParser, AutoConfig, AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizer
from trl import SFTTrainer
from datasets import Dataset, DatasetDict
import evaluate
import numpy as np
import time
import argparse
import os
import sys
import logging
import tarfile
import json
import boto3
import tempfile

from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig, AutoPeftModelForCausalLM

def download_and_extract_model(model_uri, destination_dir):
    # Create a temporary directory to store the downloaded file
    with tempfile.TemporaryDirectory() as tmp_dir:
        model_file_path = model_uri.replace('s3://eko-ekoka-ai-project/', '')
        tmp_file = os.path.join(tmp_dir, 'model.tar.gz')
        # Download the model artifact to the temporary directory
        s3_client = boto3.client('s3')
        s3_client.download_file('eko-ekoka-ai-project', model_file_path, tmp_file)
        # Extract the model artifact to the destination directory
        with tarfile.open(tmp_file, 'r:gz') as tar:
            tar.extractall(destination_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_base_model_uri', type=str)
    parser.add_argument('--lora_adaptor_artifact_uri', type=str)
    parser.add_argument('--output_model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    args, _ = parser.parse_known_args()

        # Set up logging
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


    # Download and extract the pretrained base model
    download_and_extract_model(args.pretrained_base_model_uri, '/opt/ml/input/data/pretrained_base_model')

    # Download and extract the LoRa adaptor artifact
    download_and_extract_model(args.lora_adaptor_artifact_uri, '/opt/ml/input/data/lora_adaptor_artifact')

    logger.info(f"BASE MODEL: {os.listdir('/opt/ml/input/data/pretrained_base_model')}")
    logger.info(f"LORA MODEL: {os.listdir('/opt/ml/input/data/lora_adaptor_artifact')}")

    
    # base_model = AutoModelForCausalLM.from_pretrained('/opt/ml/input/data/pretrained_base_model/2023_12_15-221854_/output')
    # base_with_adapters_model = PeftModel.from_pretrained(base_model, '/opt/ml/input/data/lora_adaptor_artifact')

    
    # merged_model = base_with_adapters_model.merge_and_unload()
    
    # merged_model.save_pretrained(args.output_model_dir)
    base_model = '/opt/ml/input/data/pretrained_base_model'
    peft_model = '/opt/ml/input/data/lora_adaptor_artifact'



    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map="auto",
        offload_folder="tmp", 
    )
    
    tokenizer = LlamaTokenizer.from_pretrained(
        base_model
    )
        
    model = PeftModel.from_pretrained(
        model, 
        peft_model, 
        torch_dtype=torch.float16,
        device_map="auto",
        offload_folder="tmp",
    )

    model = model.merge_and_unload()
    model.save_pretrained(args.output_model_dir)
    tokenizer.save_pretrained(args.output_model_dir)

