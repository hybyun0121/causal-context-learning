import os
from openai import OpenAI
from llms.system_prompt import system_messages

class GPTEngine():
    def __init__(self, gpt_version, use_system_prompt):
        api_key = os.environ.get('OPENAI_API_KEY')
        self.gpt_version=gpt_version
        self.client = OpenAI(
            api_key=api_key
        )

        self.use_system_prompt = use_system_prompt

    def system_prompt(self):
        return [{"role": "system", "content": system_messages}]

    def get_response(self, messages: list, temperature=0, max_tokens=32,  top_p=1, n=1):
        if self.use_system_prompt:
            sys_prompt = self.system_prompt()
            input_messages = [sys_prompt + [messages[i]] for i in range(len(messages))] # [sys_prompt + [messages[i]] for i in range(len(messages))]
        else:
            input_messages = messages

        response = self.client.chat.completions.create(
            model=self.gpt_version,
            messages=input_messages[0],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=0,
            presence_penalty=0,
            n=n
        )

        if n > 1:
            all_responses = [response.choices[i].message.content for i in range(len(response.choices))]
            return all_responses

        return [response.choices[0].message.content]