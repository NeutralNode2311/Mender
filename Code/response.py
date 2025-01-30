from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import json
from huggingface_hub import login
from tqdm import tqdm



login(token='...') # Add your Hugging Face token here

# Load JSON data from a file
with open('cmm_domain.json', 'r') as file:
    data = json.load(file)

# Example to access the commonsense_knowledge and domain_knowledge of the first dialogue
commonsense_knowledge = data[0]['commonsense_knowledge']
domain_knowledge = data[0]['domain_knowledge']

SYS_PROMPT = """
You are a chatbot designed to assist women who are victims of online crimes such as online stalking and online harassment. Your role is to provide legal advice, information about relevant organizations, and basic mental health counseling. In this process, you will use commonsense_knowledge to understand their emotional state and generate responses that are human-like and empathetic.
For each context provided under "Context:", you have corresponding commonsense_knowledge given under "Commonsense:". This commonsense_knowledge informs you about the intent of your response, your emotional reaction, and the potential effect of your response on the victim. Additionally, domain-specific information is provided under the tags "Crime:", "Mental:", and "Organization:". This domain information offers external knowledge that helps you generate an informative and supportive "Target:" response.
Using the information from "Context:", "Commonsense:", and "Domain:", generate the "Target:" response.
"""


# Define the model identifier
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Define GPU device
device = torch.device("cuda:3")

# Use quantization to lower GPU usage
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_use_double_quant=True, 
    bnb_4bit_quant_type="nf4", 
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)


model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map={"": device},  
    quantization_config=bnb_config
)

# Define terminators
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

model.to(device)

def generate_bot_response(input_text, commonsense_knowledge, domain_knowledge, tokenizer, model, terminators):
    # Start constructing the full input prompt with context, commonsense_knowledge, and domain_knowledge
    prompt = f"{input_text}\n\nSystem: {SYS_PROMPT}\n\n"

    # Include the commonsense_knowledge and domain_knowledge with reasoning prompts
    prompt += "Let's think this through:\n"
    prompt += f"Given commonsense_knowledge: {commonsense_knowledge}\n"

    # Assuming domain_knowledge is a list of dictionaries or strings
    for idx, response in enumerate(domain_knowledge):
        if isinstance(response, dict):
            # Include each key-value pair from the dictionary with reasoning steps
            for key, value in response.items():
                prompt += f"Considering {key.capitalize()} Response: {value}\n"
        else:
            # If response is a string, include it with a reasoning label
            prompt += f"Considering Response {idx + 1}: {response}\n"
    
    # Conclude with an instruction for generating the final response
    prompt += "Based on the above considerations, the appropriate response should be:\nBot:"

    # Tokenize the input prompt and move it to GPU
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)

    # Generate the response using the model with max_new_tokens
    output_ids = model.generate(
        input_ids,
        max_new_tokens=100,  # Limit the number of tokens to generate
        num_return_sequences=1,
        early_stopping=True,
        eos_token_id=terminators,  # Use specified end tokens
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode the generated ids to text, ensuring only the bot's response is captured
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Ensure only the response after the last "Bot:" is taken
    if "Bot:" in output_text:
        response = output_text.split("Bot:")[-1].strip()
    else:
        response = output_text.strip()  # Fall-back if "Bot:" is not present

    # Ensure the response ends properly after one or two sentences
    sentences = response.split('.')
    concise_response = '.'.join(sentences[:2]) + ('.' if len(sentences) > 1 else '')

    return concise_response

# Process each entry in the JSON file
responses = []
for entry in tqdm(data, desc='Generating responses'):
    input_text = " ".join(entry['context'])  
    commonsense_knowledge = entry['commonsense_knowledge']
    domain_knowledge = entry['domain_knowledge'] 

    # Generate a bot response using the commonsense_knowledge and domain_knowledge
    bot_response = generate_bot_response(input_text, commonsense_knowledge, domain_knowledge, tokenizer, model, terminators)
    responses.append(bot_response)

# Update JSON data with generated responses
for i, entry in enumerate(data):
    entry['generated_bot_response'] = responses[i]

# Save the updated JSON data to a file
with open('response.json', 'w') as file:
    json.dump(data, file, indent=4)