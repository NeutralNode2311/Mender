import torch
import transformers
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig, 
    HfArgumentParser, 
    TrainingArguments, 
    pipeline, 
    logging
)
from datasets import load_dataset, Dataset
import json
import textwrap
import csv
import os
import pandas as pd
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
import time
import re
from tqdm import tqdm


##device set up

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Set model name and access token

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
access_token = "..." ### huggingface access token

## Bnb configuration
bnb_4bit_compute_dtype = "float16"
use_4bit = True
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,                      # Activate 4-bit precision base model loading
        bnb_4bit_quant_type="nf4",              # Quantization type (fp4 or nf4)
        bnb_4bit_compute_dtype=compute_dtype,   # Compute dtype for 4-bit base models
        bnb_4bit_use_double_quant=False,        # Activate nested quantization for 4-bit base models (double quantization)
    )


## Model loading

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    trust_remote_code=True, 
    device_map="auto",       
    use_auth_token=access_token,
)

## Tokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=access_token)


system_prompt = '''
Generate rationales for generating the target utterance("Target:"). The rational consists of 3-hop subquestion-subanswer pairs.
Each question should contain commonsense relation in [xNeed, xWant, Causes].
These rationales should be the crucial cue for generating the target utterance but you should not include the target utterance and also pretend you don't know the target utterance.
Subquestion 3 and Subanswer 3 should be about guessing the target utterance. So the Subanswer 3 should be closely related to the target utterance but don't mention it directly.
If you think generating target utterance doesn't need commonsense, then just generate None for the rationale.
'''
user_1 = '''
[Example 1]
A: Hi, I need help planning my trip to the airport tomorrow.
B: Sure! What time is your flight?
A: It’s at 10 AM. I’m not sure when I should leave home, though.
B: It’s good to arrive at least 2 hours early. How far is your home from the airport?
A: It’s about a 45-minute drive without traffic.
B: Morning traffic can be heavy. I suggest you leave by 6:30 AM to be safe.
A: Got it. Thanks for the suggestion. Do I need to consider anything else?
Target:
B: Make sure to check in online today, pack your ID and ticket, and set an alarm for tomorrow morning.
'''
assistant_1='''
Rationale:
Subquestion 1: What does Person A need to do today to ensure a smooth trip to the airport? (xNeed)
Subanswer 1: Person A needs to check in online and pack their ID and ticket.
Subquestion 2: Why does Person B suggest leaving at 6:30 AM? (Causes)
Subanswer 2: Person B suggests leaving at 6:30 AM to account for morning traffic and ensure Person A arrives at the airport 2 hours before the flight.
Subquestion 3: What is Person A’s primary concern? (xWant)
Subanswer 3: Person A wants to plan their trip to the airport to ensure they arrive on time for their flight.

'''

def execute_chat(text):

  chat = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_1},
    {"role": "assistant", "content": assistant_1},
    {'role': 'user','content': text},
  ]

  return chat

"""## json file creation"""

def input_text(text):

    final_prompt = execute_chat(text)

    input_ids = tokenizer.apply_chat_template(
        final_prompt,
        add_generation_prompt=True,
        return_tensors="pt"
        ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=1024,
        eos_token_id=terminators,
        do_sample=True,
        temperature=1e-12,
        top_k=1,
        pad_token_id=tokenizer.eos_token_id,
    )

    response = outputs[0][input_ids.shape[-1]:]
    content_value = tokenizer.decode(response, skip_special_tokens=True)
    return content_value


df = pd.read_csv('/DATA/priyanshu_2021cs26/Abid/Knowledge Using RAG/Knowledge_using_RAG/llama3.1/commonsense_new/commonsense0-1000/dialouge0-1000.csv')

role_to_speaker = {
    'bot': 'bot',
    'user': 'user'
}

df['speaker'] = df['authorRole'].map(role_to_speaker)

data = []

for dialogue_id, group in tqdm(df.groupby('dialogueId'), desc="Processing Dialogues"):
    group = group.sort_values('utteranceNo').reset_index(drop=True)
    num_utterances = len(group)
    context_lengths = list(range(4, num_utterances, 2))

    for context_len in context_lengths:
        context = []
        for idx in range(context_len):
            speaker = group.loc[idx, 'speaker']
            utterance = group.loc[idx, 'utterances']
            context.append(f"{speaker}: {utterance}")

        target_idx = context_len
        if target_idx >= num_utterances:
            break
        target_speaker = group.loc[target_idx, 'speaker']
        target_utterance = group.loc[target_idx, 'utterances']
        target = f"{target_speaker}: {target_utterance}"

        context_text = '\n'.join(context)

        prediction = input_text(context_text)

        data_entry = {
            'd_id': str(dialogue_id),
            'context': context,
            'target': target,
            'commonsense knowledge': prediction
        }
        data.append(data_entry)

with open('/DATA/priyanshu_2021cs26/Abid/Knowledge Using RAG/Knowledge_using_RAG/llama3.1/commonsense_new/commonsense0-1000/dialouge0-1000.json', 'w') as f:
    json.dump(data, f, indent=4)

