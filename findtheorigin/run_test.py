#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from openai import OpenAI
import os
import requests
import csv
import time
import argparse
from dotenv import load_dotenv
from datetime import datetime
from .generate_prompt import generate_prompt
from .generate_prompt_shuffled import generate_prompt_shuffled
from .count_tokens import count_tokens
from .check_response import check_response
from llama_stack_client import LlamaStackClient

# Load environment variables
load_dotenv()

# API Endpoints
endpoint_claude = "https://api.anthropic.com/v1/messages"

# Initialize Llama client
LLAMA_API_KEY = os.getenv("LLAMA_API_KEY")
LLAMA_API_BASE_URL = os.getenv("LLAMA_API_BASE_URL")
llama_client = LlamaStackClient(base_url=LLAMA_API_BASE_URL, api_key=LLAMA_API_KEY)

# Logging function
LOG_FILE = "response_log.txt"

def log_response(provider, model, prompt, response):
    """Logs API responses to a file with timestamps."""
    with open(LOG_FILE, "a") as log_file:
        log_file.write(f"[{datetime.now()}] Provider: {provider}, Model: {model}\n")
        log_file.write(f"Prompt:\n{prompt}...\n")  # Log only the first 5000 chars of the prompt
        log_file.write(f"Response:\n{response}\n\n")

def query_llama_api(prompt, model="llama3.3-70b-llama_api", max_tokens=10000):
    """Function to query the Llama API with a specified model and context length."""
    truncated_prompt = prompt  # DO NOT Truncate prompt to fit context window
    try:
        response = llama_client.inference.chat_completion(
            model_id=model,
            messages=[{"role": "user", "content": truncated_prompt}],
        )
        response_text = response.completion_message.content.text if response else "Error: API request failed"
        log_response("llama", model, truncated_prompt, response_text)
        return response_text
    except Exception as e:
        error_message = f"Error: {str(e)}"
        log_response("llama", model, truncated_prompt, error_message)
        return error_message

def main():
    parser = argparse.ArgumentParser(description="Run the Find the Origin benchmark tests.")
    parser.add_argument('--provider', type=str, required=True, help='API provider (openai, anthropic, or llama)')
    parser.add_argument('--model_name', type=str, required=True, help='Model name (e.g., gpt-3.5-turbo-0125)')
    parser.add_argument('--d_parameter', type=int, required=True, help='Distance parameter')
    parser.add_argument('--max_lines', type=int, required=True, help='Maximum number of lines to insert')
    parser.add_argument('--step_lines', type=int, required=True, help='Step increment for number of lines')
    parser.add_argument('--shuffle', type=bool, required=False, default=False,
                        help='Randomize positioning of irrelevant vertices.')

    args = parser.parse_args()
    d, max_lines, step_lines, model, provider, shuffle = args.d_parameter, args.max_lines, args.step_lines, args.model_name, args.provider, args.shuffle

    n_tokens, model_responses = [], []
    client = OpenAI()

    for n in range(1 + abs(d), max_lines, step_lines):
        print(f'Running prompt for {n} lines of vertices')

        prompt = generate_prompt('vertices.txt', 'vertices_reorg.txt', d,
                                 n) if not shuffle else generate_prompt_shuffled('vertices.txt', 'vertices_reorg.txt',
                                                                                 d, n)
        num_tokens = count_tokens(prompt)
        n_tokens.append(num_tokens)

        if provider == 'openai':
            got_response = False
            while not got_response:
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": prompt},
                        ],
                        max_tokens=256,
                        temperature=0.0
                    )
                    got_response = True
                    response_text = response.choices[0].message.content
                    log_response("openai", model, prompt, response_text)
                except Exception as e:
                    error_message = f"Error accessing OpenAI: {str(e)}"
                    log_response("openai", model, prompt, error_message)
                    print(error_message)
                    print('Trying again in 10 seconds')
                    time.sleep(10)

            model_responses.append(check_response(response_text))

        elif provider == 'anthropic':
            api_key_claude = os.getenv('CLAUDE_API_KEY')
            headers_claude = {'Content-Type': 'application/json', 'x-api-key': api_key_claude,
                              'anthropic-version': '2023-06-01'}
            data_claude = {"model": model, "messages": [{"role": "user", "content": prompt}],
                           "system": "You are a helpful assistant.", "max_tokens": 256, "top_p": 1, "temperature": 0.0}
            got_response = False
            while not got_response:
                try:
                    response = requests.post(endpoint_claude, headers=headers_claude, json=data_claude)
                    if response.status_code == 200:
                        got_response = True
                        response_json = response.json()
                        text_content = response_json['content'][0]['text']
                        log_response("anthropic", model, prompt, text_content)
                    else:
                        error_message = f"Error accessing Anthropic: {response.text}"
                        log_response("anthropic", model, prompt, error_message)
                        print(error_message)
                        print('Trying again in 10 seconds')
                        time.sleep(10)
                except Exception as e:
                    error_message = f"Error accessing Anthropic: {str(e)}"
                    log_response("anthropic", model, prompt, error_message)
                    print(error_message)
                    print('Trying again in 10 seconds')
                    time.sleep(10)

            model_responses.append(check_response(text_content))

        elif provider == 'llama':
            response_text = query_llama_api(prompt, model)
            model_responses.append(check_response(response_text))

    # Saving results
    filename = f'results_d{d}_model_{model}_max_lines{max_lines}_steps{step_lines}.csv'
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['n_tokens', model])
        for values in zip(n_tokens, model_responses):
            writer.writerow(values)


if __name__ == "__main__":
    main()
