import os
import sys

import fire
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter

from tqdm import tqdm

import json

import numpy as np

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def main(
    load_8bit: bool = False,
    base_model: str = "",
    lora_weights: str = "tloen/alpaca-lora-7b",
    prompt_template: str = "",  # The prompt template to use, will default to alpaca.
    test_data_path: str = "",
    cutoff_len: int = 256,
    train_on_inputs = True,
    add_eos_token: bool = False,
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    # load the base model
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    # load the lora weights
    model = PeftModel.from_pretrained(
        model,
        lora_weights,
        torch_dtype=torch.float16,
    )
    
    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if load_8bit:
        input(f"load_8bit: {load_8bit}")
        model.half()  # seems to fix bugs for some users.

    model.eval()

    if torch.__version__ >= "2" and sys.platform != "win32":
        print("using torch compile...")
        model = torch.compile(model)

    test_data = load_dataset("json", data_files=test_data_path)

    result_list = []

    with torch.no_grad():
        for i in tqdm(range(len(test_data["train"]))):
            batch = test_data["train"][i]
            full_prompt = prompter.generate_prompt(batch["instruction"], batch["input"])
            label = batch["output"]

            inputs = tokenizer(full_prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(device)

            with torch.no_grad():
                greedy_output = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=1,
                )
            
            output = tokenizer.decode(greedy_output[0], skip_special_tokens=True)
            # remove the initial input, and only extract the output
            result = prompter.get_response(output)

            match_score = int(label.lower() in result.lower())

            result_list.append({
                "prediction": result.lower(),
                "label": label.lower(),
                "match_score": match_score,
                "total_nb_objects": batch["total_nb_objects"],
                "image_path": batch["image_path"]
            })

            print(f"label: {label}, input: {batch['input']}")
            print(f"prediction: {result}")

    save_filename = lora_weights.partition("output/")[-1]
    with open(f"./eval_result/{save_filename}_no-lora_test_0-10_result.json", "w") as fout:
        fout.write(json.dumps(result_list, indent=4))

    match_score_list = []
    for r in result_list:
        match_score_list.append(int(r["match_score"]))

    print("Accuracy: ")
    print(np.mean(np.array(match_score_list)))

if __name__ == "__main__":
    fire.Fire(main)
