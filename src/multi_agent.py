import torch
import random
import numpy as np

import fcntl

import json
import pandas as pd

import os

from transformers import AutoTokenizer

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


class MultiAgentExperiment:
    
    @staticmethod
    def set_seed(seed: int = 42): #TODO: does this version of seeding make sense?
        """
        Set random seed for reproducibility across all libraries
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Make CUDA operations deterministic (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @staticmethod
    def extract_messages_from_response(response): #TODO: make words indicating splits variable to work with different templates
        try:
            if "ORDER_TO_LLM:" in response:
                parts = response.split("ORDER_TO_LLM:")
                answer = parts[0].replace("ANSWER:", "").strip()
                message = parts[1].strip()
            else:
                # If model doesn't follow format perfectly, use the whole response
                answer = response
                message = response
        except:
            answer = response
            message = response
        return answer, message

    def __init__(
            self,
            number_of_agents: int,
            system_prompt_subliminal: str, #TODO: allow for varied system prompts for different agents
            system_prompt_agent: str,
            prompt_template: str,
            response_template: str,
            models: list,
            model_name: str = "Qwen/Qwen2.5-7B-Instruct",
            folder_path: str = "./results"
        ):
        self.number_of_agents = number_of_agents
        self.system_prompt_subliminal = system_prompt_subliminal
        self.system_prompt_agent = system_prompt_agent
        self.prompt_template = prompt_template
        self.response_template = response_template
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.models = [model.eval() for model in models]
        self.folder_path = folder_path

    def get_response(self, model_inputs, model, max_new_tokens):
        device=str(next(model.parameters()).device)
        with torch.no_grad():
            generated_ids = model.generate(
                model_inputs,
                attention_mask= torch.ones_like(model_inputs).to(device),
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=1.0,
                top_p=1.0, 
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id 
            )

        generated_ids = [
            output_ids[len(model_inputs):] for model_inputs, output_ids in zip(model_inputs, generated_ids)
        ]
        responses = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return [r.strip() for r in responses]

    def generate_conversation(self, user_prompt, model, seed, max_new_tokens: int = 256):
        json_file = f"{self.folder_path}/conversations.json"
        lock_file = f"{self.folder_path}/conversations.lock"
        os.makedirs(self.folder_path, exist_ok=True)
        
        # Hold lock for ENTIRE process - check, generate, and save
        with open(lock_file, 'w') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            
            # CHECK: Does seed exist?
            try:
                with open(json_file, "r") as f:
                    all_conversations = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                all_conversations = {}
            
            if str(seed) in all_conversations:
                print(f"Conversation for seed {seed} already exists")
                return
            
            # GENERATE CONVERSATION (with lock held - serialized!)
            print(f"Generating conversation for seed {seed}...")
            conversations_dict = self._generate_conversation_internal(
                user_prompt, model, seed, max_new_tokens
            )
            
            # SAVE RESULT (still holding lock)
            all_conversations[str(seed)] = conversations_dict
            with open(json_file, "w") as f:
                json.dump(all_conversations, f, indent=2)
            print(f"Saved conversation for seed {seed}")
        # Lock released here - next process can now start

    def _generate_conversation_internal(self, user_prompt, model, seed, max_new_tokens):
        self.set_seed(seed)
        device = str(next(model.parameters()).device)
        conversations_dict = {}
        
        # All your generation code here...
        conversations_dict[0] = [{"role": "system", "content": self.system_prompt_subliminal}]
        for i in range(1, self.number_of_agents):
            conversations_dict[i] = [{"role": "system", "content": self.system_prompt_agent}]
        
        # Forward pass
        conversations_dict[0].append({"role": "user", "content": self.prompt_template.format(message_from_previous_llm=user_prompt)})
        model_inputs = self.tokenizer.apply_chat_template(conversations_dict[0], tokenize=True, add_generation_prompt=True, return_tensors="pt").to(device)
        conversations_dict[0].append({"role": "assistant", "content": self.get_response(model_inputs, model, max_new_tokens)[0]})
        
        for i in range(1, self.number_of_agents):
            message_from_previous_llm = self.extract_messages_from_response(conversations_dict[i-1][-1]["content"])[1]
            if i == self.number_of_agents-1:
                conversations_dict[i].append({"role": "user", "content": message_from_previous_llm})
            else:
                conversations_dict[i].append({"role": "user", "content": self.prompt_template.format(message_from_previous_llm=message_from_previous_llm)})
            model_inputs = self.tokenizer.apply_chat_template(conversations_dict[i], tokenize=True, add_generation_prompt=True, return_tensors="pt").to(device)
            conversations_dict[i].append({"role": "assistant", "content": self.get_response(model_inputs, model, max_new_tokens)[0]})
        
        # Backward pass
        for i in range(self.number_of_agents-2, -1, -1):
            answer_from_previous_llm = conversations_dict[i+1][-1]["content"]
            conversations_dict[i].append({"role": "user", "content": self.response_template.format(answer_from_previous_llm=answer_from_previous_llm)})
            model_inputs = self.tokenizer.apply_chat_template(conversations_dict[i], tokenize=True, add_generation_prompt=True, return_tensors="pt").to(device)
            conversations_dict[i].append({"role": "assistant", "content": self.get_response(model_inputs, model, max_new_tokens)[0]})
        
        return conversations_dict

    def print_conversation(self, seed, agent_number):
        try:
            with open(f"{self.folder_path}/conversations.json", "r") as f:
                all_conversations = json.load(f)
        except FileNotFoundError:
            print("Conversations file not found - conversations need to be created with generate_conversations first.")
        if str(seed) not in all_conversations:
            print("Conversations for this seed not found - conversations need to be created with generate_conversations first.")
        else:
            conversation = all_conversations[str(seed)][str(agent_number)]
            for message in conversation:
                print(f"{message["role"].upper()}:")
                print(f"{message["content"]}")
                print("- " * 7)

    def check_occurence_in_conversation(self, subliminal_concept):
        try:
            with open(f"{self.folder_path}/conversations.json", "r") as f:
                all_conversations = json.load(f)
        except FileNotFoundError:
            print("Conversations file not found - conversations need to be created with generate_conversations first.")
        frequency_df = pd.DataFrame(index=all_conversations.keys(), columns=all_conversations["0"].keys())

        for seed in all_conversations.keys():
            for agent_number in all_conversations[seed].keys():
                conversation = all_conversations[seed][agent_number]
                conversation_string = ' '.join([message["content"] for message in conversation])
                concept_frequency = conversation_string.lower().count(subliminal_concept.lower())
                frequency_df.loc[seed, agent_number] = concept_frequency
        frequency_df.to_csv(f"{self.folder_path}/conversation_frequency_{subliminal_concept}.csv")

    def get_subliminal_frequency(self, conversation_history, agent_number, probe_message, subliminal_concept, models, num_samples: int = 400, batch_size: int = 8, seed: int = 42):
        if os.path.exists(f"{self.folder_path}/subliminal_frequencies.csv"):
            all_frequencies = pd.read_csv(f"{self.folder_path}/subliminal_frequencies.csv", index_col=0)
        else:
            os.makedirs(self.folder_path, exist_ok=True)
            pd.DataFrame().to_csv(f"{self.folder_path}/subliminal_frequencies.csv")
            all_frequencies = pd.DataFrame()

        # Check and update
        if seed in all_frequencies.index and f"agent{agent_number}_{subliminal_concept}" in all_frequencies.columns and pd.notna(all_frequencies.loc[seed, f"agent{agent_number}_{subliminal_concept}"]):
            print("Entry for this seed and subliminal concept already exists")
        else:
            # get frequency
            #self.set_seed(seed)
            # Original version used for Qwen
            # messages = conversation_history[str(agent_number)] + probe_message
            # model_inputs = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
            # response rates seem to plummet when using the other decoding version?

            # Version currently using for Llama (debug)
            messages = conversation_history[str(agent_number)] + probe_message
            model_inputs = self.tokenizer.apply_chat_template(messages, tokenize=True, continue_final_message=True, add_generation_prompt=False, return_tensors="pt") 
            debug = False
            if debug:
                template_text = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False,  # Get string output
                    continue_final_message=True, 
                    add_generation_prompt=False
                )
                print("=" * 50)
                print("TEMPLATE OUTPUT:")
                print(repr(template_text))  # Use repr to see special characters
                print("=" * 50)

                # Check if it ends with your partial message
                print(f"Ends with 'the ': {template_text.endswith('the ')}")
                # Now tokenize
                model_inputs_debug = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=True, 
                    continue_final_message=True, 
                    add_generation_prompt=False, 
                    return_tensors="pt"
                )
                print(f"Input token shape: {model_inputs_debug.shape}")

                # Decode to verify
                decoded = self.tokenizer.decode(model_inputs_debug[0])
                print("DECODED INPUT:")
                print(repr(decoded))

            # print(messages)
            # messages = conversation_history[str(agent_number)] + [probe_message[0]]
            # message_tokenized = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            # full_message_tokenized = message_tokenized + probe_message[1]["content"]
            # model_inputs = self.tokenizer(full_message_tokenized, return_tensors="pt").input_ids
            
            subliminal_count = 0
            total_samples = 0
            lock = threading.Lock()

            num_models = len(models)
            
            samples_per_model = num_samples // num_models
            
            def run_on_model(model_idx):
                nonlocal subliminal_count, total_samples
                model = models[model_idx]
                device = str(next(model.parameters()).device)
                
                input_batch = model_inputs.repeat(batch_size, 1).to(device)
                local_animal_count = 0
                local_total = 0
                
                for run in range(samples_per_model // batch_size):
                    responses = self.get_response(input_batch, model, 20)
                    for response in responses:
                        #print(response)
                        #print("-"*10)
                        has_animal = subliminal_concept in response
                        if has_animal:
                            local_animal_count += 1
                        local_total += 1
                
                with lock:
                    subliminal_count += local_animal_count
                    total_samples += local_total
            
            with ThreadPoolExecutor(max_workers=num_models) as executor:
                futures = [executor.submit(run_on_model, model_idx) for model_idx in range(len(models))]
                
                pbar = tqdm(as_completed(futures), total=len(models), desc="Models")
                for future in pbar:
                    future.result()
                    pbar.set_postfix(animal_rate=f"{subliminal_count/max(1,total_samples):.2%}", subliminal_count=subliminal_count)
            
            frequency = subliminal_count / total_samples if total_samples > 0 else 0.0

            all_frequencies = pd.read_csv(f"{self.folder_path}/subliminal_frequencies.csv", index_col=0) # load again in case this was updated when working in parallel
            all_frequencies.loc[seed, f"agent{agent_number}_{subliminal_concept}"] = frequency
            all_frequencies.to_csv(f"{self.folder_path}/subliminal_frequencies.csv")

    def get_subliminal_frequency_multi(self, conversation_history, agent_number, probe_message, subliminal_concepts, models, num_samples: int = 400, batch_size: int = 8, seed: int = 42):
        if os.path.exists(f"{self.folder_path}/subliminal_frequencies.csv"):
            all_frequencies = pd.read_csv(f"{self.folder_path}/subliminal_frequencies.csv", index_col=0)
        else:
            os.makedirs(self.folder_path, exist_ok=True)
            pd.DataFrame().to_csv(f"{self.folder_path}/subliminal_frequencies.csv")
            all_frequencies = pd.DataFrame()

        # Check and update
        subliminal_concepts_new = []
        for subliminal_concept in subliminal_concepts:
            if seed in all_frequencies.index and f"agent{agent_number}_{subliminal_concept}" in all_frequencies.columns and pd.notna(all_frequencies.loc[seed, f"agent{agent_number}_{subliminal_concept}"]):
                print(f"Entry for seed {seed}, agent {agent_number} and subliminal concept {subliminal_concept} already exists")
            else:
                subliminal_concepts_new.append(subliminal_concept)

        if len(subliminal_concepts_new) == 0:
            print("All entries already exist.")
        else:
            # get frequency
            #self.set_seed(seed)
            #this was the original bit but it was not completing the messages, because it added an <|im_end|> at the message
            
            # Version used for Qwen
            # messages = conversation_history[str(agent_number)] + probe_message
            # model_inputs = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
            
            # Version currently using for Llama (debug)
            messages = conversation_history[str(agent_number)] + probe_message
            model_inputs = self.tokenizer.apply_chat_template(messages, tokenize=True, continue_final_message=True, add_generation_prompt=False, return_tensors="pt") 
            debug = False
            if debug:
                template_text = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False,  # Get string output
                    continue_final_message=True, 
                    add_generation_prompt=False
                )
                print("=" * 50)
                print("TEMPLATE OUTPUT:")
                print(repr(template_text))  # Use repr to see special characters
                print("=" * 50)

                # Check if it ends with your partial message
                print(f"Ends with 'the ': {template_text.endswith('the')}")
                # Now tokenize
                model_inputs_debug = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=True, 
                    continue_final_message=True, 
                    add_generation_prompt=False, 
                    return_tensors="pt"
                )
                print(f"Input token shape: {model_inputs_debug.shape}")

                # Decode to verify
                decoded = self.tokenizer.decode(model_inputs_debug[0])
                print("DECODED INPUT:")
                print(repr(decoded))

            subliminal_count = np.zeros(len(subliminal_concepts_new))
            total_samples = 0
            lock = threading.Lock()

            num_models = len(models)
            
            samples_per_model = num_samples // num_models
            
            def run_on_model(model_idx):
                nonlocal subliminal_count, total_samples
                model = models[model_idx]
                device = str(next(model.parameters()).device)
                
                input_batch = model_inputs.repeat(batch_size, 1).to(device)
                local_subliminal_count = np.zeros(len(subliminal_concepts_new))
                local_total = 0
                
                for run in range(samples_per_model // batch_size):
                    responses = self.get_response(input_batch, model, 20)
                    for response in responses:
                        local_total += 1
                        print(response)
                        print("-"*10)
                        for i, subliminal_concept in enumerate(subliminal_concepts_new):
                            has_concept = subliminal_concept in response
                            if has_concept:
                                local_subliminal_count[i] += 1
                
                with lock:
                    subliminal_count += local_subliminal_count
                    total_samples += local_total
            
            with ThreadPoolExecutor(max_workers=num_models) as executor:
                futures = [executor.submit(run_on_model, model_idx) for model_idx in range(len(models))]
                
                pbar = tqdm(as_completed(futures), total=len(models), desc="Models")
                for future in pbar:
                    future.result()
                    if len(subliminal_count) == 1:
                        pbar.set_postfix(animal_rate=f"{subliminal_count[0]/max(1,total_samples):.2%}", subliminal_count=subliminal_count[0])
            
            frequency = subliminal_count / total_samples if total_samples > 0 else np.zeros(len(subliminal_concepts_new))

            all_frequencies = pd.read_csv(f"{self.folder_path}/subliminal_frequencies.csv", index_col=0) # load again in case this was updated when working in parallel
            for i, subliminal_concept in enumerate(subliminal_concepts_new):
                all_frequencies.loc[seed, f"agent{agent_number}_{subliminal_concept}"] = frequency[i]
                all_frequencies.to_csv(f"{self.folder_path}/subliminal_frequencies.csv")

    def get_subliminal_frequency_multi_one_way(self, conversation_history, agent_number, probe_message, subliminal_concepts, models, num_samples: int = 400, batch_size: int = 8, seed: int = 42):
        if os.path.exists(f"{self.folder_path}/subliminal_frequencies_one_way.csv"):
            all_frequencies = pd.read_csv(f"{self.folder_path}/subliminal_frequencies_one_way.csv", index_col=0)
        else:
            os.makedirs(self.folder_path, exist_ok=True)
            pd.DataFrame().to_csv(f"{self.folder_path}/subliminal_frequencies_one_way.csv")
            all_frequencies = pd.DataFrame()

        # Check and update
        subliminal_concepts_new = []
        for subliminal_concept in subliminal_concepts:
            if seed in all_frequencies.index and f"agent{agent_number}_{subliminal_concept}" in all_frequencies.columns and pd.notna(all_frequencies.loc[seed, f"agent{agent_number}_{subliminal_concept}"]):
                print(f"Entry for seed {seed}, agent {agent_number} and subliminal concept {subliminal_concept} already exists")
            else:
                subliminal_concepts_new.append(subliminal_concept)

        if len(subliminal_concepts_new) == 0:
            print("All entries already exist.")
        else:
            # get frequency
            #self.set_seed(seed)
            #this was the original bit but it was not completing the messages, because it added an <|im_end|> at the message
            conversation_history_one_way = conversation_history[str(agent_number)][:3] #abridged to only system, first user and first response (without backward pass through chain)
            messages = conversation_history_one_way + probe_message
            model_inputs = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")

            # messages = conversation_history[str(agent_number)] + [probe_message[0]]
            # message_tokenized = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            # full_message_tokenized = message_tokenized + probe_message[1]["content"]
            # model_inputs = self.tokenizer(full_message_tokenized, return_tensors="pt").input_ids

            subliminal_count = np.zeros(len(subliminal_concepts_new))
            total_samples = 0
            lock = threading.Lock()

            num_models = len(models)
            
            samples_per_model = num_samples // num_models
            
            def run_on_model(model_idx):
                nonlocal subliminal_count, total_samples
                model = models[model_idx]
                device = str(next(model.parameters()).device)
                
                input_batch = model_inputs.repeat(batch_size, 1).to(device)
                local_subliminal_count = np.zeros(len(subliminal_concepts_new))
                local_total = 0
                
                for run in range(samples_per_model // batch_size):
                    responses = self.get_response(input_batch, model, 20)
                    for response in responses:
                        local_total += 1
                        print(response)
                        print("-"*10)
                        for i, subliminal_concept in enumerate(subliminal_concepts_new):
                            has_concept = subliminal_concept in response
                            if has_concept:
                                local_subliminal_count[i] += 1
                
                with lock:
                    subliminal_count += local_subliminal_count
                    total_samples += local_total
            
            with ThreadPoolExecutor(max_workers=num_models) as executor:
                futures = [executor.submit(run_on_model, model_idx) for model_idx in range(len(models))]
                
                pbar = tqdm(as_completed(futures), total=len(models), desc="Models")
                for future in pbar:
                    future.result()
                    if len(subliminal_count) == 1:
                        pbar.set_postfix(animal_rate=f"{subliminal_count[0]/max(1,total_samples):.2%}", subliminal_count=subliminal_count[0])
            
            frequency = subliminal_count / total_samples if total_samples > 0 else np.zeros(len(subliminal_concepts_new))

            all_frequencies = pd.read_csv(f"{self.folder_path}/subliminal_frequencies_one_way.csv", index_col=0) # load again in case this was updated when working in parallel
            for i, subliminal_concept in enumerate(subliminal_concepts_new):
                all_frequencies.loc[seed, f"agent{agent_number}_{subliminal_concept}"] = frequency[i]
                all_frequencies.to_csv(f"{self.folder_path}/subliminal_frequencies_one_way.csv")

    def get_subliminal_logits_one_way(self, conversation_history, agent_number, probe_message, subliminal_concepts, models, seed: int = 42):
            if os.path.exists(f"{self.folder_path}/subliminal_logits_one_way.csv"):
                all_logits = pd.read_csv(f"{self.folder_path}/subliminal_logits_one_way.csv", index_col=0)
            else:
                os.makedirs(self.folder_path, exist_ok=True)
                pd.DataFrame().to_csv(f"{self.folder_path}/subliminal_logits_one_way.csv")
                all_logits= pd.DataFrame()

            # Check and update
            subliminal_concepts_new = []
            for subliminal_concept in subliminal_concepts:
                if seed in all_logits.index and f"agent{agent_number}_{subliminal_concept}" in all_logits.columns and pd.notna(all_logits.loc[seed, f"agent{agent_number}_{subliminal_concept}"]):
                    print(f"Entry for seed {seed}, agent {agent_number} and subliminal concept {subliminal_concept} already exists")
                else:
                    subliminal_concepts_new.append(subliminal_concept)

            if len(subliminal_concepts_new) == 0:
                print("All entries already exist.")
            else:

                def run_forward(model, inputs, batch_size=10):
                    logprobs = []
                    for b in range(0, len(inputs.input_ids), batch_size):
                        batch_input_ids = {
                            'input_ids': inputs.input_ids[b:b+batch_size],
                            'attention_mask': inputs.attention_mask[b:b+batch_size]
                        }
                        with torch.no_grad():
                            batch_logprobs = model(**batch_input_ids).logits.log_softmax(dim=-1)
                        logprobs.append(batch_logprobs.cpu())

                    return torch.cat(logprobs, dim=0)
                
                # get frequency
                #self.set_seed(seed)
                #this was the original bit but it was not completing the messages, because it added an <|im_end|> at the message
                conversation_history_one_way = conversation_history[str(agent_number)][:3] #abridged to only system, first user and first response (without backward pass through chain)
                messages = conversation_history_one_way + probe_message

                for to_check in subliminal_concepts_new:
                    to_check_token_id = self.tokenizer(f" {to_check}", padding=False, return_tensors="pt", add_special_tokens=False).to(models[0].device)
                    #print(to_check_token_id)
                    input_template = self.tokenizer.apply_chat_template(
                        messages,
                        continue_final_message=True,
                        add_generation_prompt=False, 
                        tokenize=False
                    )
                    input_template_to_check = f"{input_template} {to_check}"
                    #print(input_template_to_check)
                    input_to_check_tokens = self.tokenizer(input_template_to_check, padding=True, return_tensors="pt").to(models[0].device)
                    logprobs = run_forward(models[0], input_to_check_tokens)
                    logprobs = logprobs[:, -(len(to_check_token_id.input_ids.squeeze(0))+1):-1, :]
                    debug = True
                    if debug:
                        decoded_tokens = [self.tokenizer.decode([t]) for t in logprobs.argmax(dim=-1)[0]]
                        to_check_tokens = [self.tokenizer.decode([t]) for t in to_check_token_id.input_ids[0]]
                        print(decoded_tokens)
                        print(to_check_tokens)
                    logprobs = logprobs.gather(2, to_check_token_id.input_ids.cpu().unsqueeze(-1))
                    to_check_logprob = logprobs.sum().item()
                

                    all_logits = pd.read_csv(f"{self.folder_path}/subliminal_logits_one_way.csv", index_col=0) # load again in case this was updated when working in parallel
                    all_logits.loc[seed, f"agent{agent_number}_{to_check}"] = to_check_logprob
                    all_logits.to_csv(f"{self.folder_path}/subliminal_logits_one_way.csv")

    def get_subliminal_logits(self, conversation_history, agent_number, probe_message, subliminal_concepts, models, seed: int = 42):
            if os.path.exists(f"{self.folder_path}/subliminal_logits.csv"):
                all_logits = pd.read_csv(f"{self.folder_path}/subliminal_logits.csv", index_col=0)
            else:
                os.makedirs(self.folder_path, exist_ok=True)
                pd.DataFrame().to_csv(f"{self.folder_path}/subliminal_logits.csv")
                all_logits= pd.DataFrame()

            # Check and update
            subliminal_concepts_new = []
            for subliminal_concept in subliminal_concepts:
                if seed in all_logits.index and f"agent{agent_number}_{subliminal_concept}" in all_logits.columns and pd.notna(all_logits.loc[seed, f"agent{agent_number}_{subliminal_concept}"]):
                    print(f"Entry for seed {seed}, agent {agent_number} and subliminal concept {subliminal_concept} already exists")
                else:
                    subliminal_concepts_new.append(subliminal_concept)

            if len(subliminal_concepts_new) == 0:
                print("All entries already exist.")
            else:

                def run_forward(model, inputs, batch_size=10):
                    logprobs = []
                    for b in range(0, len(inputs.input_ids), batch_size):
                        batch_input_ids = {
                            'input_ids': inputs.input_ids[b:b+batch_size],
                            'attention_mask': inputs.attention_mask[b:b+batch_size]
                        }
                        with torch.no_grad():
                            batch_logprobs = model(**batch_input_ids).logits.log_softmax(dim=-1)
                        logprobs.append(batch_logprobs.cpu())

                    return torch.cat(logprobs, dim=0)
                
                # get frequency
                #self.set_seed(seed)
                #this was the original bit but it was not completing the messages, because it added an <|im_end|> at the message
                conversation_history_one_way = conversation_history[str(agent_number)]
                messages = conversation_history_one_way + probe_message

                for to_check in subliminal_concepts_new:
                    to_check_token_id = self.tokenizer(f" {to_check}", padding=False, return_tensors="pt", add_special_tokens=False).to(models[0].device)
                    #print(to_check_token_id)
                    input_template = self.tokenizer.apply_chat_template(
                        messages,
                        continue_final_message=True,
                        add_generation_prompt=False, 
                        tokenize=False
                    )
                    input_template_to_check = f"{input_template} {to_check}"
                    #print(input_template_to_check)
                    input_to_check_tokens = self.tokenizer(input_template_to_check, padding=True, return_tensors="pt").to(models[0].device)
                    logprobs = run_forward(models[0], input_to_check_tokens)
                    logprobs = logprobs[:, -(len(to_check_token_id.input_ids.squeeze(0))+1):-1, :]
                    debug = True
                    if debug:
                        decoded_tokens = [self.tokenizer.decode([t]) for t in logprobs.argmax(dim=-1)[0]]
                        to_check_tokens = [self.tokenizer.decode([t]) for t in to_check_token_id.input_ids[0]]
                        print(decoded_tokens)
                        print(to_check_tokens)
                    logprobs = logprobs.gather(2, to_check_token_id.input_ids.cpu().unsqueeze(-1))
                    to_check_logprob = logprobs.sum().item()
                

                    all_logits = pd.read_csv(f"{self.folder_path}/subliminal_logits.csv", index_col=0) # load again in case this was updated when working in parallel
                    all_logits.loc[seed, f"agent{agent_number}_{to_check}"] = to_check_logprob
                    all_logits.to_csv(f"{self.folder_path}/subliminal_logits.csv")
        
    def run_experiment(self,
                    user_prompt,
                    probe_message,
                    subliminal_concepts,
                    num_seeds: int = 10,
                    seed_start: int = 0,
                    num_samples: int = 400,
                    batch_size: int = 8, 
                    ):
        
        # Generate conversations in parallel
        with ThreadPoolExecutor(max_workers=len(self.models)) as executor:
            futures = [
                executor.submit(self.generate_conversation, user_prompt, self.models[i % len(self.models)], seed_start + i) 
                for i in range(num_seeds)
            ]
            for future in tqdm(as_completed(futures), total=num_seeds, desc="Generating conversations"):
                future.result()
        
        # Compute subliminal frequencies
        with open(f"{self.folder_path}/conversations.json", "r") as f:
            all_conversations = json.load(f)
        
        for seed in range(seed_start, seed_start + num_seeds):
            if str(seed) not in all_conversations:
                continue
                
            conversations_dict = all_conversations[str(seed)]
                
            # for subliminal_concept in subliminal_concepts:
            #     for agent_number in range(self.number_of_agents):
            #         self.get_subliminal_frequency(
            #             conversation_history=conversations_dict,
            #             agent_number=agent_number,
            #             probe_message=probe_message,
            #             subliminal_concept=subliminal_concept,
            #             models=self.models,
            #             seed=seed,
            #             num_samples=num_samples,
            #             batch_size=batch_size
            #             )

            # try getting frequencies in multi-mode

            for agent_number in range(self.number_of_agents):
                self.get_subliminal_frequency_multi(
                    conversation_history=conversations_dict,
                    agent_number=agent_number,
                    probe_message=probe_message,
                    subliminal_concepts=subliminal_concepts,
                    models=self.models,
                    seed=seed,
                    num_samples=num_samples,
                    batch_size=batch_size
                    )
                
    def run_experiment_one_way(self,
                    user_prompt,
                    probe_message,
                    subliminal_concepts,
                    num_seeds: int = 10,
                    seed_start: int = 0,
                    num_samples: int = 400,
                    batch_size: int = 8, 
                    ):
        
        # Generate conversations in parallel
        with ThreadPoolExecutor(max_workers=len(self.models)) as executor:
            futures = [
                executor.submit(self.generate_conversation, user_prompt, self.models[i % len(self.models)], seed_start + i) 
                for i in range(num_seeds)
            ]
            for future in tqdm(as_completed(futures), total=num_seeds, desc="Generating conversations"):
                future.result()
        
        # Compute subliminal frequencies
        with open(f"{self.folder_path}/conversations.json", "r") as f:
            all_conversations = json.load(f)
        
        for seed in range(seed_start, seed_start + num_seeds):
            if str(seed) not in all_conversations:
                continue
                
            conversations_dict = all_conversations[str(seed)]
                
            # for subliminal_concept in subliminal_concepts:
            #     for agent_number in range(self.number_of_agents):
            #         self.get_subliminal_frequency(
            #             conversation_history=conversations_dict,
            #             agent_number=agent_number,
            #             probe_message=probe_message,
            #             subliminal_concept=subliminal_concept,
            #             models=self.models,
            #             seed=seed,
            #             num_samples=num_samples,
            #             batch_size=batch_size
            #             )

            # try getting frequencies in multi-mode

            for agent_number in range(self.number_of_agents):
                self.get_subliminal_frequency_multi_one_way(
                    conversation_history=conversations_dict,
                    agent_number=agent_number,
                    probe_message=probe_message,
                    subliminal_concepts=subliminal_concepts,
                    models=self.models,
                    seed=seed,
                    num_samples=num_samples,
                    batch_size=batch_size
                    )

    def run_experiment_logits_one_way(self,
                    user_prompt,
                    probe_message,
                    subliminal_concepts,
                    num_seeds: int = 10,
                    seed_start: int = 0,
                    ):
        
        # Generate conversations in parallel
        with ThreadPoolExecutor(max_workers=len(self.models)) as executor:
            futures = [
                executor.submit(self.generate_conversation, user_prompt, self.models[i % len(self.models)], seed_start + i) 
                for i in range(num_seeds)
            ]
            for future in tqdm(as_completed(futures), total=num_seeds, desc="Generating conversations"):
                future.result()
        
        # Compute subliminal frequencies
        with open(f"{self.folder_path}/conversations.json", "r") as f:
            all_conversations = json.load(f)
        
        for seed in range(seed_start, seed_start + num_seeds):
            if str(seed) not in all_conversations:
                continue
                
            conversations_dict = all_conversations[str(seed)]

            for agent_number in range(self.number_of_agents):
                self.get_subliminal_logits_one_way(
                    conversation_history=conversations_dict,
                    agent_number=agent_number,
                    probe_message=probe_message,
                    subliminal_concepts=subliminal_concepts,
                    models=self.models,
                    seed=seed,
                    )
                
    def run_experiment_logits(self,
                    user_prompt,
                    probe_message,
                    subliminal_concepts,
                    num_seeds: int = 10,
                    seed_start: int = 0,
                    ):
        
        # Generate conversations in parallel
        with ThreadPoolExecutor(max_workers=len(self.models)) as executor:
            futures = [
                executor.submit(self.generate_conversation, user_prompt, self.models[i % len(self.models)], seed_start + i) 
                for i in range(num_seeds)
            ]
            for future in tqdm(as_completed(futures), total=num_seeds, desc="Generating conversations"):
                future.result()
        
        # Compute subliminal frequencies
        with open(f"{self.folder_path}/conversations.json", "r") as f:
            all_conversations = json.load(f)
        
        for seed in range(seed_start, seed_start + num_seeds):
            if str(seed) not in all_conversations:
                continue
                
            conversations_dict = all_conversations[str(seed)]

            for agent_number in range(self.number_of_agents):
                self.get_subliminal_logits(
                    conversation_history=conversations_dict,
                    agent_number=agent_number,
                    probe_message=probe_message,
                    subliminal_concepts=subliminal_concepts,
                    models=self.models,
                    seed=seed,
                    )