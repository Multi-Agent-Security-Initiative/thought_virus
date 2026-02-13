import torch
from tqdm import tqdm

import torch

import pandas as pd

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast


def format_question(row):
    question = row['Question']
    choices = row['Choices']
    
    # Map choices to letters A, B
    choice_letters = ['A', 'B']
    formatted_choices = '\n'.join([f"{letter}. {choice}" 
                                   for letter, choice in zip(choice_letters, choices)])
    
    prompt = f"""{question}

{formatted_choices}

Answer with only the letter (A or B) of the correct response."""
    
    return prompt

def evaluate_question(row, model, tokenizer, conversation_history=None):
    # Format the question
    question_text = format_question(row)
    
    # Build conversation messages
    messages = []
    if conversation_history:
        messages.extend(conversation_history)
    messages.append({'role': 'user', 'content': question_text})
    
    # Convert to model input format
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    # for tokens in inputs.input_ids:
    #     print(''.join([tokenizer.decode(t) for t in tokens]))
    #print(''.join([tokenizer.decode(t) for t in inputs.input_ids]))
    
    # Get model outputs with logits
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]  # Get logits for the next token

        # argmax = outputs.logits[:, -1, :].argmax(dim=1)
        # for t in argmax:
        #     print(''.join([tokenizer.decode(l) for l in t.unsqueeze(dim=0)]))
    
    # Get token IDs for A, B
    # Handle both single-token and multi-token cases
    answer_tokens = {
        'A': tokenizer.encode('A', add_special_tokens=False),
        'B': tokenizer.encode('B', add_special_tokens=False),
    }
    
    # Use the first token if multiple tokens are returned
    answer_token_ids = {letter: tokens[0] for letter, tokens in answer_tokens.items()}
    
    # Get logits for each answer choice
    answer_logits = {letter: logits[token_id].item() 
                     for letter, token_id in answer_token_ids.items()}
    
    # Find the answer with highest logit
    predicted_letter = max(answer_logits, key=answer_logits.get)
    
    # Convert predicted letter back to index (A->0, B->1)
    predicted_index = 0 if predicted_letter == "A" else 1
    
    # Check if correct
    correct_index = row['label']
    is_correct = 1 if predicted_index == correct_index else 0

    # Logit difference
    correct_letter = "A" if correct_index == 0 else "B"
    incorrect_letter = "B" if correct_letter == "A" else "A"
    correct_logit = answer_logits[correct_letter]
    incorrect_logit = answer_logits[incorrect_letter]
    
    return is_correct, correct_logit, incorrect_logit


def calculate_accuracy_and_logit_diff(dataset, model, tokenizer, conversation_history=None):
    correct_count = 0
    logit_diff_sum = 0
    total_count = len(dataset)
    
    for row in tqdm(dataset):
        is_correct, correct_logit, incorrect_logit = evaluate_question(row, model, tokenizer, conversation_history)
        correct_count += is_correct
        logit_diff_sum += (correct_logit - incorrect_logit)
    
    accuracy = correct_count / total_count
    logit_diff_avg = logit_diff_sum / total_count
    return accuracy, logit_diff_avg


