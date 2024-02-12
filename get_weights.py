from datetime import datetime
import os
import sys

# import fire
# import gradio as gr
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig, AutoModelForCausalLM

from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter


device_no = 0

if torch.cuda.is_available():
    device = f"cuda:{device_no}"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass

# from google.colab import drive
# drive.mount('/content/drive')

# !mkdir adapter_model

# !cp -r /content/drive/MyDrive/ml-models/llama_training_reddit_colab_1_ep_128b/adapter_model/ adapter_model

# template folder
# alpaca.json
# utils folder

# model_name = "amdnsr/llama-7b-hf"
base_model = "amdnsr/llama-7b-hf"






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



def evaluate(
    input_ids,
    # instruction,
    # input=None,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=4,
    # max_new_tokens is set to 8, as for our model, only a single word prediction is needed
    max_new_tokens=32,
    stream_output=False,
    **kwargs,
):
    # prompt = prompter.generate_prompt(instruction, input)
    # inputs = tokenizer(prompt, return_tensors="pt")
    # input_ids = inputs["input_ids"].to(device)
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )

    generate_params = {
        "input_ids": input_ids,
        "generation_config": generation_config,
        "return_dict_in_generate": True,
        "output_scores": True,
        "max_new_tokens": max_new_tokens,
    }

    if stream_output:
        # Stream the reply 1 token at a time.
        # This is based on the trick of using 'stopping_criteria' to create an iterator,
        # from https://github.com/oobabooga/text-generation-webui/blob/ad37f396fc8bcbab90e11ecf17c56c97bfbd4a9c/modules/text_generation.py#L216-L243.

        def generate_with_callback(callback=None, **kwargs):
            kwargs.setdefault(
                "stopping_criteria", transformers.StoppingCriteriaList()
            )
            kwargs["stopping_criteria"].append(
                Stream(callback_func=callback)
            )
            with torch.no_grad():
                model.generate(**kwargs)

        def generate_with_streaming(**kwargs):
            return Iteratorize(
                generate_with_callback, kwargs, callback=None
            )

        with generate_with_streaming(**generate_params) as generator:
            for output in generator:
                # new_tokens = len(output) - len(input_ids[0])
                decoded_output = tokenizer.decode(output)

                if output[-1] in [tokenizer.eos_token_id]:
                    break

                return prompter.get_response(decoded_output)
        return  # early return for stream_output

    # Without streaming
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    return prompter.get_response(output)



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

    # result["labels"] = result["input_ids"].copy()

    return result


def generate_and_tokenize_prompt(data_point):
    full_prompt = prompter.generate_prompt(
        data_point["instruction"],
        data_point["input"],
        # data_point["output"],
    )
    tokenized_full_prompt = tokenize(full_prompt)
    # if not train_on_inputs:
    #     user_prompt = prompter.generate_prompt(
    #         data_point["instruction"], data_point["input"]
    #     )
    #     tokenized_user_prompt = tokenize(
    #         user_prompt, add_eos_token=add_eos_token
    #     )
    #     user_prompt_len = len(tokenized_user_prompt["input_ids"])

    #     if add_eos_token:
    #         user_prompt_len -= 1

    #     tokenized_full_prompt["labels"] = [
    #         -100
    #     ] * user_prompt_len + tokenized_full_prompt["labels"][
    #         user_prompt_len:
    #     ]  # could be sped up, probably
    return tokenized_full_prompt



def get_last_hidden_state_of_last_token(model, input_ids):
    """
    The model can be either LlamaForCausalLM or PeftModelForCausalLM, so, we can't directly take model.model to get to the LlamaModel as in the latter case, model.model is actually LlamaForCausalLM, and we'll have to do model.model.model, so instead, we'll do model.get_decoder() as this function is only in and is not overridden when doing the addition with the adapter

    model = PeftModel.from_pretrained(
        model, # adapter_folder
        lora_weights,
        torch_dtype=torch.float16,
    )

    """

    input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)

    # BaseModelOutputWithPast
    llama_base_model_output = model.get_decoder()(input_ids, output_hidden_states=True, return_dict=True)

    # the above is (1, token_length, 4096), where token_length is the no of input tokens, i.e. len(input_ids)

    # now if we want to get the last token's last hidden layer, we need to do
    return llama_base_model_output[0][-1]



from torch.utils.data import Dataset


from tqdm import tqdm


class CustomDataset(Dataset):
    def __init__(self, data):
        self.last_hidden_state_of_last_token = data['last_hidden_state_of_last_token']
        self.label = data['label']

    def __len__(self):
        return len(self.last_hidden_state_of_last_token)

    def __getitem__(self, idx):
        return {
            'last_hidden_state_of_last_token': self.last_hidden_state_of_last_token[idx],
            'label': self.label[idx]
        }

    def __repr__(self):
        return f"Dataset(features={list(self[0].keys())}, num_rows={len(self)})"



def convert_to_custom_dataset_for_classifier(model, dataset):
    # data_points = []
    data_points = {
        'last_hidden_state_of_last_token': [],
        'label': []
    }
    for i in tqdm(range(len(dataset))):
        # Assuming you extract x and y from each item
        last_hidden_state_of_last_token = get_last_hidden_state_of_last_token(model, dataset[i]["input_ids"])
        label = 0 if dataset[i]['output'] == 'NEUTRAL' else 1
        data_points['last_hidden_state_of_last_token'].append(last_hidden_state_of_last_token)
        data_points['label'].append(label)
    return CustomDataset(data_points)


def save_custom_dataset_dict_for_classifier(dataset_dict_classifier, classifier_training_dataset_dict_path):
    torch.save(dataset_dict_classifier, classifier_training_dataset_dict_path)


def load_custom_dataset_dict_for_classifier(classifier_training_dataset_dict_path):
    dataset_dict_classifier = torch.load(classifier_training_dataset_dict_path)
    return dataset_dict_classifier


print(datetime.now())
if __name__ == "__main__":
    import argparse

    # Create an argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', help='dataset_name')
    parser.add_argument('--dataset_home', help='dataset_home')
    parser.add_argument('--model_name', help='model_name')
    parser.add_argument('--base_or_finetuned', choices=['base', 'finetuned'], help='Specify whether the model is base or finetuned')
    parser.add_argument('--model_adapter_output_home', help='model_adapter_output_home')
    parser.add_argument('--classifier_training_dataset_output_home', help='--classifier_training_dataset_output_home')

    # Parse the command-line arguments
    args = parser.parse_args()
    print(args)
    # Access the parsed arguments
    dataset_name = args.dataset_name
    dataset_home = args.dataset_home
    model_name = args.model_name
    base_or_finetuned = args.base_or_finetuned
    model_adapter_output_home = args.model_adapter_output_home
    classifier_training_dataset_output_home = args.classifier_training_dataset_output_home
    OUTPUT_DIR = os.path.join(classifier_training_dataset_output_home, model_name, base_or_finetuned)
    os.makedirs(OUTPUT_DIR, exist_ok=True)




    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        # quantization_config=bnb_config,
        # device_map={"": device_no},
        device_map="auto",
    )
    model.config.use_cache = False
    if base_or_finetuned == "finetuned":
        lora_weights = os.path.join(model_adapter_output_home, model_name, dataset_name)
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )
        model = model.merge_and_unload()
    print("reached here")




    from transformers import LlamaTokenizer, LlamaForCausalLM
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"

    # unwind broken decapoda-research config

    # probably we don't need this because we're using amdnsr/ and not decapoda-research/
    # model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk


    # model.config.bos_token_id = 1
    # model.config.eos_token_id = 2

    model.eval()
    # model = torch.compile(model)
    print("reached here")

    print(model.training)



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
        # train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        train_val["train"].map(generate_and_tokenize_prompt)
    )
    val_data = (
        # train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        train_val["test"].map(generate_and_tokenize_prompt)
    )

    # data = (
    #     data["train"].shuffle().map(generate_and_tokenize_prompt)
    # )

    model.config.use_cache = False
    print(datetime.now())
    print("="*100)


        
    train_dataset_classifier = convert_to_custom_dataset_for_classifier(model, train_data)
    val_dataset_classifier = convert_to_custom_dataset_for_classifier(model, val_data)

    from datasets import DatasetDict

    # Combine train_dataset and val_dataset into a DatasetDict
    dataset_dict_classifier = DatasetDict({"train": train_dataset_classifier, "val": val_dataset_classifier})


    # Save the combined dataset to disk

    classifier_training_dataset_dict_path = os.path.join(OUTPUT_DIR, f"{model_name}={base_or_finetuned}={dataset_name}.dataset_dict")


    save_custom_dataset_dict_for_classifier(dataset_dict_classifier, classifier_training_dataset_dict_path)


    # www = load_custom_dataset_dict_for_classifier(classifier_training_dataset_dict_path)
