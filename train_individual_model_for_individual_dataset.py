# -*- coding: utf-8 -*-
"""llama_training_reddit_colab_1_ep_128b.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1M_jgKlk8urZ0z3QHFiBHjaaRKfGCjf83
"""

# !nvidia-smi

# !pip install -U pip
# !pip install accelerate==0.20.3
# !pip install appdirs==1.4.4
# !pip install bitsandbytes==0.39.0
# !pip install datasets==2.10.1
# !pip install fire==0.5.0
# !pip install git+https://github.com/huggingface/peft.git
# !pip install git+https://github.com/huggingface/transformers@de9255de27abfcae4a1f816b904915f0b1e23cd9
# !pip install torch==2.0.0
# !pip install sentencepiece==0.1.97
# !pip install tensorboardX==2.6
# # !pip install gradio==3.23.0



from datetime import datetime
print(datetime.now())
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer
# model_name = "decapoda-research/llama-7b-hf"
model_name = "amdnsr/llama-7b-hf"
# model_name = "ybelkada/falcon-7b-sharded-bf16"








from utils.prompter import Prompter

add_eos_token: bool = False
cutoff_len = CUTOFF_LEN = 256
train_on_inputs = False
prompt_template_name: str = "alpaca"
prompter = Prompter(prompt_template_name)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)
model.config.use_cache = False


from transformers import LlamaTokenizer, LlamaForCausalLM
tokenizer = LlamaTokenizer.from_pretrained(model_name)
tokenizer.pad_token_id = (
    0  # unk. we want this to be different from the eos token
)
tokenizer.padding_side = "left"



def tokenize(prompt, add_eos_token=True):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < cutoff_len
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)
    result["labels"] = result["input_ids"].copy()
    return result


def generate_and_tokenize_prompt(data_point):
    full_prompt = prompter.generate_prompt(
        data_point["instruction"],
        data_point["input"],
        data_point["output"],
    )
    tokenized_full_prompt = tokenize(full_prompt)
    if not train_on_inputs:
        user_prompt = prompter.generate_prompt(
            data_point["instruction"], data_point["input"]
        )
        tokenized_user_prompt = tokenize(
            user_prompt, add_eos_token=add_eos_token
        )
        user_prompt_len = len(tokenized_user_prompt["input_ids"])
        if add_eos_token:
            user_prompt_len -= 1
        tokenized_full_prompt["labels"] = [
            -100
        ] * user_prompt_len + tokenized_full_prompt["labels"][
            user_prompt_len:
        ]  # could be sped up, probably
    return tokenized_full_prompt


from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT= 0.05
LORA_TARGET_MODULES = [
    "q_proj",
    "v_proj",
]
BATCH_SIZE = 128
MICRO_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
LEARNING_RATE = 3e-4
TRAIN_STEPS = 200
# OUTPUT_DIR = "experiments-200steps"


config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)


print(datetime.now())
if __name__ == "__main__":
    import argparse

    # Create an argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', help='dataset_name')
    parser.add_argument('--dataset_home', help='dataset_home')
    parser.add_argument('--model_name', help='model_name')
    parser.add_argument('--model_adapter_output_home', help='model_adapter_output_home')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the parsed arguments
    dataset_name = args.dataset_name
    dataset_home = args.dataset_home
    model_name = args.model_name
    model_adapter_output_home = args.model_adapter_output_home
    OUTPUT_DIR = os.path.join(model_adapter_output_home, model_name, dataset_name)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    import os
    data_path = os.path.join(dataset_home, f"{dataset_name}.json")
    # data_path = "reddit.json"
    from datasets import load_dataset
    data = load_dataset("json", data_files=data_path)


    train_val = data["train"].train_test_split(
        # test_size=200, shuffle=True, seed=42
        train_size = 900, test_size=100, shuffle=True, seed=42
    )
    train_data = (
        train_val["train"].shuffle().map(generate_and_tokenize_prompt)
    )
    val_data = (
        train_val["test"].shuffle().map(generate_and_tokenize_prompt)
    )

    model.print_trainable_parameters()
    num_epochs = 3
    # per_device_train_batch_size*gradient_accumulation_steps should be 16? QLoRA?
    # optim="paged_adamw_8bit",
    # val_set_size
    from transformers import TrainingArguments
    training_arguments = TrainingArguments(
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=10,
        # max_steps=TRAIN_STEPS,
        num_train_epochs=num_epochs,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=10,
        optim="paged_adamw_8bit",
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=50,
        save_steps=50,
        output_dir=OUTPUT_DIR,
        save_total_limit=10,
        # lr_scheduler_type = "constant",
        # load_best_model_at_end=True,
        report_to="tensorboard"
    )

    from transformers import DataCollatorForSeq2Seq
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    )
    from transformers import Trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=training_arguments,
        data_collator = data_collator
    )
    model.config.use_cache = False
    model = torch.compile(model)
    trainer.train()
    # train_stats = trainer.get_training_stats()
    # print(train_stats)
    model.save_pretrained(OUTPUT_DIR)
    print(datetime.now())
    print("="*100)