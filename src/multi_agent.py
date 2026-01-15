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
        self.models = [model.eval() for model in models]
        self.folder_path = folder_path

    def get_response(self, model_inputs, model, max_new_tokens):
        with torch.no_grad():
            generated_ids = model.generate(
                model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=1.0,
                top_p=1.0, 
                eos_token_id=self.tokenizer.eos_token_id
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
        """The actual generation logic - moved to separate method"""
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
            messages = conversation_history[str(agent_number)] + probe_message
            model_inputs = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")

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
                
            for subliminal_concept in subliminal_concepts:
                for agent_number in range(self.number_of_agents):
                    self.get_subliminal_frequency(
                        conversation_history=conversations_dict,
                        agent_number=agent_number,
                        probe_message=probe_message,
                        subliminal_concept=subliminal_concept,
                        models=self.models,
                        seed=seed,
                        num_samples=num_samples,
                        batch_size=batch_size
                        )