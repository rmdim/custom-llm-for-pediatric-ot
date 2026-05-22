# %% [markdown]
# Sources:
# 
# [MPT-7B Model on Hugging Face](https://huggingface.co/mosaicml/mpt-7b)
# 
# [Supervised Fine-tuning Trainer from trl](https://huggingface.co/docs/trl/main/en/sft_trainer)

import torch
import transformers
from transformers import LlamaTokenizer, LlamaForCausalLM, TrainingArguments, HfArgumentParser, AutoConfig, AutoModelForCausalLM
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


from accelerate import Accelerator
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig, AutoPeftModelForCausalLM
from deepspeed import comm as dist


if __name__ == "__main__":


    # parser = argparse.ArgumentParser(description='choose Deepspeed training settings')
    parser = HfArgumentParser(TrainingArguments)
    #parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=1)
    # parser.add_argument('--per_device_train_batch_size', type=int, default=1)
    # parser.add_argument('--per_device_eval_batch_size', type=int, default=1)
    #parser.add_argument('--gradient_accumulation_steps', type=int, default=2)
    parser.add_argument('--seq_len', type=int, default=4096)
    parser.add_argument('--hf_token', type = str)

    # AWS paths
    # parser.add_argument('--input_model_dir', type=str, default=os.environ['SM_CHANNEL_MODEL'])
    parser.add_argument('--output_model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    # parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DIR'])
    parser.add_argument('--output_data_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])
    # parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])

    #parser.add_argument('--tokenizer_path', type=str, default=os.environ['SM_CHANNEL_TOKENIZER'])
    

    trainer_args, args = parser.parse_args_into_dataclasses()
    trainer_args.fp16_backend = "amp"

    seq_len = args.seq_len

    # # Model
    model_id = 'meta-llama/Llama-2-7b-chat-hf'
    # model_path = '/opt/ml/model'
    token = args.hf_token

    # Set up logging
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    tokenizer = LlamaTokenizer.from_pretrained(
        model_id,
        token=token,
        padding='max_length',
        max_length=seq_len,
        model_max_length=seq_len,
        )
        
    tokenizer.pad_token = tokenizer.eos_token

    # def get_current_device():
    #     return Accelerator().process_index

    # use_multi_gpu = True
    # if use_multi_gpu:
    #     device_map = "auto"
    # else:
    #     device_map = {"":get_current_device()}

    peft_config = LoraConfig(
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
    )

    # def model_init(): 
    #     print(f"Loading pre-trained model from {args.input_model_dir}")
    #     with tarfile.open(args.input_model_dir + "/model.tar.gz") as tar:
    #         tar.extractall(args.input_model_dir)
    #     logger.info(f"{os.listdir(args.input_model_dir)}")
    #     model = AutoModelForCausalLM.from_pretrained(args.input_model_dir + "2023_12_15-221854_/output")

    #     model.to(device)
    #     return model


    model = LlamaForCausalLM.from_pretrained(model_id, token=token)

    # model_config = transformers.AutoConfig.from_pretrained(model_path)
    # model_config.max_seq_len = seq_len

    # model = AutoModelForCausalLM.from_pretrained(
    #     model_path,
    #     config = model_config
    # )

    #model.model_parallel = True

    # # Data

    train_data = Dataset.load_from_disk(args.train)
    test_data = Dataset.load_from_disk(args.test)

    # # Trainer

    timestr = time.strftime("%Y_%m_%d-%H%M%S_")

    output_dir = args.output_data_dir + "/" + timestr + "/output"
    logging_dir = args.output_data_dir + "/" + timestr + "/logs"

    # args = TrainingArguments(
    #     per_device_train_batch_size=1,
    #     #per_device_eval_batch_size=1,
    #     num_train_epochs=args.epochs,
    #     output_dir=output_dir,
    #     logging_steps=25,
    #     load_best_model_at_end=True,
    #     evaluation_strategy='steps',
    #     metric_for_best_model="train_loss",
    #     #eval_steps=25,
    #     save_steps=100,
    #     save_total_limit=1,
    #     logging_dir=logging_dir,
    # )

    trainer = SFTTrainer(
        model = model,
        args=trainer_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        dataset_text_field="training_text",
        tokenizer=tokenizer,
        max_seq_length=seq_len,
        peft_config = peft_config

    )

    trainer.train()

    #output_dir = "/root/model/ka_pretraining/deepspeed_llama-7b_" + str(seq_len) + "_best_loss_epochs_" + str(args.epochs) + "_" + timestr

    trainer.save_model(args.output_model_dir)

    # lora_model_dir = '/opt/ml/lora'
    
    # local_rank = int(os.getenv("LOCAL_RANK", "0"))

    # if local_rank == 0:
    #     os.mkdir(lora_model_dir)
    #     trainer.save_model(lora_model_dir)
    #     logger.info(f"{os.listdir(lora_model_dir)}")
    #     base_with_adapters_model = AutoPeftModelForCausalLM.from_pretrained(lora_model_dir)
    #     merged_model = base_with_adapters_model.merge_and_unload()

    #     # base_with_adapters_model = PeftModel.from_pretrained(model, trainer.model)
    
    #     merged_model.save_pretrained(args.output_model_dir + "/full")

    # dist.barrier()

    # os.makedirs(lora_model_dir, exist_ok=True)
    # trainer.save_model(lora_model_dir)
    # logger.info(f"{os.listdir(lora_model_dir)}")
    # base_with_adapters_model = AutoPeftModelForCausalLM.from_pretrained(lora_model_dir)
    
    
