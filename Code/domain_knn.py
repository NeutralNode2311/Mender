import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from huggingface_hub import login
import spacy
from sentence_transformers import util
import json
from tqdm import tqdm 


device = torch.device('cuda:3')
torch.cuda.set_device(device) 

# Login to Hugging Face
login(token='...')  # Add your Hugging Face token here

# Load Sentence Transformer Model on GPU
ST = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1").to(device)

# Function to process dataset and add FAISS index
def process_dataset(dataset):
    def embed(batch):
        embeddings = ST.encode(batch['text'], convert_to_tensor=True, device=device)
        return {'embeddings': embeddings.cpu().numpy()}

    # Embedding the text and mapping to dataset
    dataset = dataset.map(embed, batched=True, batch_size=8)
    
    # Adding a FAISS index to the dataset
    dataset.add_faiss_index(column="embeddings")
    return dataset

from concurrent.futures import ThreadPoolExecutor

def load_and_process_parallel(folder_paths, split='train'):
    def load_process(path):
        dataset = load_dataset(path, split=split)
        print(f"Processing {dataset.num_rows} entries from {path}")
        return process_dataset(dataset)

    with ThreadPoolExecutor(max_workers=len(folder_paths)) as executor:
        results = list(executor.map(load_process, folder_paths))
    
    return {name: result for name, result in zip(["crime", "legal", "med", "mental", "org"], results)}

folder_paths = [
    "/crime",
    "/legal",
    "/med",
    "/mental",
    "/org"
]

datasets_indexed = load_and_process_parallel(folder_paths)

# Function to search using the FAISS index from all indexed datasets
def search_all_datasets(query: str, k: int = 2):
    # Embed the query
    embedded_query = ST.encode(query, convert_to_tensor=True, device=device).cpu().numpy()

    # Dictionary to store search results from each dataset
    search_results = {}

    # Perform the search on each indexed dataset
    for dataset_name, indexed_dataset in datasets_indexed.items():
        scores, retrieved_examples = indexed_dataset.get_nearest_examples("embeddings", embedded_query, k=k)
        search_results[dataset_name] = {"scores": scores, "examples": retrieved_examples}

    return search_results

# Load the language model
nlp = spacy.load("en_core_web_sm") 

def format_prompt(prompt, retrieved_documents, k):
    """Using the retrieved documents, we will prompt the model to generate our responses."""
    PROMPT = f"Question: {prompt}\nContext:\n"
    for idx in range(k):
        PROMPT += f"{retrieved_documents['text'][idx]}\n"
    return PROMPT

# Load the tokenizer and model
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# # Configure BitsAndBytesConfig for 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.float16
)

# Load the model and explicitly set it to the GPU
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    torch_dtype=torch.float16
)

# Explicitly move the model to the correct device
model.to(device)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

SYS_PROMPT = (
    """You are integrated with a RAG-based knowledge extraction model. Silently categorize dialogue content
    related to crime, mental health, or Indian law during analysis but do not explicitly separate these
    categories in the output. For crime-related dialogue, identify specific offenses and link them to the
    relevant Indian Penal Codes. For mental health, pinpoint specific concerns or disorders. For Indian law,
    highlight pertinent statutes and penal codes. Utilize the knowledge base to match conversation embeddings
    with appropriate text files, ensuring a seamless integration of all pertinent details such as websites,
    links, phone numbers, locations, and procedural information into a unified, concise summary. Aim to deliver
    a response that is informative and engaging, presenting all relevant data in a flowing narrative format."""
)

def generate(formatted_prompt):
    formatted_prompt = formatted_prompt[:2000]  # To avoid GPU OOM

    input_text = SYS_PROMPT + "\n" + formatted_prompt
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )
    response = outputs[0][inputs['input_ids'].shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)

def select_relevant_sentences(prompt, response, max_sentences=2):
    # Encode the prompt for semantic similarity comparison
    prompt_embedding = ST.encode(prompt, convert_to_tensor=True, device=device)

    # Split response into sentences and process each sentence
    sentences = response.split('. ')
    sentence_embeddings = ST.encode(sentences, convert_to_tensor=True, device=device)
    similarities = util.cos_sim(prompt_embedding, sentence_embeddings)[0]

    # Named Entity Recognition and Keyword Extraction
    relevant_terms = set()
    doc = nlp(prompt)
    for ent in doc.ents:
        relevant_terms.add(ent.text.lower())
    for token in doc:
        if token.pos_ in ['NOUN', 'PROPN']: 
            relevant_terms.add(token.text.lower())

    # Filter sentences based on similarity and content relevance
    filtered_sentences = []
    for idx, sentence in enumerate(sentences):
        sentence_doc = nlp(sentence)
        sentence_terms = {token.text.lower() for token in sentence_doc if token.pos_ in ['NOUN', 'PROPN', 'VERB']}

        # Check if sentence contains relevant terms or entities
        if relevant_terms.intersection(sentence_terms):
            filtered_sentences.append((sentence, similarities[idx].item()))

    # Sort sentences based on similarity score and retain the top N
    filtered_sentences.sort(key=lambda x: x[1], reverse=True)
    top_sentences = [sent[0] for sent in filtered_sentences[:max_sentences]]

    return '. '.join(top_sentences) + '.'


def rag_chatbot(prompt: str, k: int = 2):
    # Replace the following lines with actual implementations of these functions
    retrieved_data = search_all_datasets(prompt, k)
    all_responses = {
        "crime": "",
        "legal": "",
        "med": "",
        "mental": "",
        "org": ""
    }

    # Iterate over each dataset's results
    for dataset_name, results in retrieved_data.items():
        retrieved_documents = results['examples']  
        scores = results['scores']  
        formatted_prompt = format_prompt(prompt, retrieved_documents, k)
        full_response = generate(formatted_prompt)
        relevant_response = select_relevant_sentences(prompt, full_response)

        if dataset_name in all_responses:
            all_responses[dataset_name] = relevant_response
        else:
            pass

    return all_responses

def process_json_input(input_file: str, output_file: str, k: int = 2):
  
    with open(input_file, 'r') as file:
        data = json.load(file)

    # Check if the data is a list of dictionaries
    if isinstance(data, list):
        # Iterate over each object (dictionary) in the list with a progress bar
        for item in tqdm(data, desc="Processing items"):
            if 'context' in item:
                # Combine the context messages into a single string to serve as the prompt
                prompt = ' '.join(item['context'])

                # Call the rag_chatbot function with the combined prompt
                responses = rag_chatbot(prompt, k)

                # Store the responses in the item (dictionary) under a new key
                item['rag_responses'] = [responses]
    else:
        if 'context' in data:
            prompt = ' '.join(data['context'])
            responses = rag_chatbot(prompt, k)
            data['rag_responses'] = [responses]

    # Write the modified data back to a new JSON file
    with open(output_file, 'w') as file:
        json.dump(data, file, indent=4)

    print(f"Processed data has been written to {output_file}")

# Specify the input and output file paths at the global scope
input_file_path = 'common_knn.json'  # Path to the input JSON file
output_file_path = 'domain_knn.json'  

# Call the function to process the JSON input
process_json_input(input_file_path, output_file_path)