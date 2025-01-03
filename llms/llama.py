import torch
import torch.nn as nn
import transformers
# from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from llms.system_prompt import system_messages

class LLama(nn.Module):
    def __init__(self, cache_dir, model_name, use_system_prompt=False) -> None:
        super().__init__()
        self.cache_dir = cache_dir
        self.model_name = model_name
        self.use_system_prompt = use_system_prompt

        self.set_model()

    def set_model(self):
        # Load the model with 8-bit quantization
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        if self.cache_dir is not None:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                quantization_config=quantization_config,
                device_map="auto",
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
            )

        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Move the model to evaluation mode
        self.model.eval()

        # Disable gradients for faster inference
        for param in self.model.parameters():
            param.requires_grad = False

    def system_prompt(self):
        return [{"role": "system", "content": system_messages}]

    def get_response(self, messages: list, max_tokens=512):
        if len(messages) > 1:
            if self.use_system_prompt:
                sys_prompt = self.system_prompt()
                batch_messages = [sys_prompt + messages[i] for i in range(len(messages))]
            else:
                batch_messages = messages
        else:
            batch_messages = messages

        # Prepare the input texts
        input_texts = []
        for message in batch_messages:
            # Concatenate all message contents
            text = ''
            for m in message:
                text += f"{m['role']}: {m['content']}\n"
            input_texts.append(text)

        # Tokenize the inputs
        inputs = self.tokenizer(
            input_texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=1024,  # Adjust max_length as needed
        ).to(self.model.device)

        # Generate responses
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=None,
                top_p=1.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode the outputs
        responses = []
        for i, output in enumerate(outputs):
            # Skip the input tokens to get only the generated text
            generated_text = self.tokenizer.decode(output[inputs['input_ids'].size(1):], skip_special_tokens=True)
            responses.append(generated_text.strip())

        return responses

# class LLama(nn.Module):
#     def __init__(self, cache_dir, model_name, use_system_prompt=False) -> None:
#         super().__init__()
#         self.cache_dir = cache_dir
#         self.model_name = model_name
#         self.use_system_prompt = use_system_prompt

#         self.set_model()

#     def set_model(self):
#         if self.cache_dir is not None:
#             self.model = AutoModelForCausalLM.from_pretrained(self.model_name, cache_dir=self.cache_dir, torch_dtype=torch.bfloat16, device_map="auto", token=True)
#             self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir, device_map="auto", token=True)
#         else:
#             self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.bfloat16, device_map="auto", token=True)
#             self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, device_map="auto", token=True)

#         self.tokenizer.padding_side = 'left'
#         if self.tokenizer.pad_token_id is None:
#             self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

#     def system_prompt(self):
#         return [{"role": "system", "content": system_messages}]

#     def get_response(self, messages: list, max_tokens=512):
#         if len (messages) > 1:
#             if self.use_system_prompt:
#                 sys_prompt = self.system_prompt()
#                 batch_messages = [sys_prompt + messages[i] for i in range(len(messages))]
#             else:
#                 batch_messages = messages

#         pipeline = transformers.pipeline(
#             "text-generation",
#             model=self.model,
#             tokenizer=self.tokenizer,
#             device_map='auto'
#         )

#         outputs = pipeline(
#             batch_messages,
#             batch_size=len(messages)//10,
#             max_new_tokens=max_tokens,
#             # eos_token_id=terminators,
#             do_sample=True,
#             temperature=1e-6,
#             top_p=1.0,
#         )

#         # Extract generated texts
#         responses = []
#         for output in outputs:
#             generated_text = output[0]['generated_text'][-1]['content']
#             responses.append(generated_text)
#         return responses