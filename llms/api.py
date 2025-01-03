import os
import time
import re
from groq import Groq, RateLimitError
from llms.system_prompt import system_messages

from openai import OpenAI

class LLM_API():
    def __init__(self, model_name, use_system_prompt) -> None:
        self.model_name = model_name
        self.use_system_prompt = use_system_prompt

        api_key = os.environ.get('NIM_API_KEY')
        self.client = OpenAI(
                            base_url = "https://integrate.api.nvidia.com/v1",
                            api_key=api_key
        )

    def system_prompt(self):
        return [{"role": "system", "content": system_messages}]

    def get_response(self, messages, max_tokens=512):
        if self.use_system_prompt:
            sys_prompt = self.system_prompt()
            input_messages = [sys_prompt + [messages[i]] for i in range(len(messages))]
        else:
            input_messages = messages

        completion = self.client.chat.completions.create(
            model="meta/llama-3.1-8b-instruct",
            messages=input_messages[0],
            top_p=1,
            max_tokens=1024,
            stream=False
            )
        
        return [completion.choices[0].message.content]

    def extract_retry_time(self, error_message):
        match = re.search(r'Please try again in (\d+\.?\d*)s', error_message)
        if match:
            return float(match.group(1))
        else:
            return 60