import torch
import transformers
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig, 
    pipeline, 
    HfArgumentParser, 
    TrainingArguments, 
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
Given a conversation context, generate rationales that logically lead to the target utterance (“Target:”). Your task is to create three subquestion-subanswer pairs, each utilizing one of the following commonsense relations:

	•	oEffect: Effects on others.
	•	oReact: Others’ reactions.
	•	oWant: What others want.
	•	xAttr: Person’s attributes.
	•	xIntent: Person’s intentions.
	•	xNeed: What the person needs.
	•	xReact: Person’s reactions.
	•	xWant: What the person wants.
	•	isAfter: Events that happen after.
	•	isBefore: Events that happen before.
	•	Causes: What causes an event.

Instructions for Each Subquestion-Subanswer Pair:

	1.	Subquestion [number]:
	•	Begin with “Subquestion [number]: [Question] ([Commonsense Relation])”.
	•	The question should be specific, relevant to the context, and incorporate the indicated commonsense relation.
	2.	Subanswer [number]:
	•	Provide a direct and concise answer to the subquestion, offering insight based on the conversation context.

Important Guidelines:

	•	Do not include or directly mention the target utterance in your rationales.
	•	Assume you do not know the exact wording of the target utterance.
	•	For Subquestion 3 and Subanswer 3:
	•	Focus on inferring what the assistant might logically say next, using commonsense reasoning.
	•	The subanswer should be closely related to the target utterance without stating it directly.
	•	Ensure clarity and coherence:
	•	Make sure your rationales are clear, concise, and logically connected to the conversation context.
	•	Follow the specified format precisely for consistency.
'''


user_1 = '''
bot: Greetings , this is rakshak to help you in secure environment . How may i help you ?
user: The mosquitos are biting me! I can’t stop scratching . They are everywhere ! Sneaky little jerks .
bot: Do you have any bug spray ?
user: No , I forgot to buy some .
bot: Then we ’ ll have to put up with it .
user: We can cover ourselves with beer ! That way if they bite us , they ’ ll get drunk and fall asleep .
bot: That’s without a doubt , the best idea you’ve had ! Let’s do it !
Target:
user: Run! They are thirsty for more!
'''

assistant_1 = '''

Rationale:
Subquestion 1: What is the intent of user when suggesting the use of beer to ward off mosquitos? (xIntent)
Subanswer 1:  user's intention is to make the mosquitos 'drunk' and cause them to fall asleep, reducing the amount of bites.
Subquestion 2: What is  bot's reaction to  user's unique idea to use beer? (xReact)
Subanswer 2:  user finds the idea amusing and agreeable, and shows enthusiasm in trying it out.
Subquestion 3: What might be the effect on the mosquitos after  user and bot use beer to ward them off? (oEffect)
Subanswer 3: Unexpectedly, the mosquitos might be attracted to the beer, causing them to swarm more intensively, creating the need for  bot to warn user about the increased mosquito activity.

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


# Step 1: Read the CSV file
df = pd.read_csv('context.csv')

# Step 2: Map Author Roles to Speaker Labels
role_to_speaker = {
    'bot': 'bot',
    'user': 'user'
}

df['speaker'] = df['authorRole'].map(role_to_speaker)

# Step 3: Process the data
data = []

# Use tqdm to wrap over dialogue IDs
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

        # Join the context into a single string
        context_text = '\n'.join(context)

        # Get the prediction
        prediction = input_text(context_text)

        data_entry = {
            'd_id': str(dialogue_id),
            'context': context,
            'target': target,
            'commonsense knowledge': prediction
        }
        data.append(data_entry)

# Step 4: Save the data to a JSON file
with open('cmm_knn.json', 'w') as f:
    json.dump(data, f, indent=4)

