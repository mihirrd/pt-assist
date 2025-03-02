import os
import dotenv
import json
from typing import List
from openai import OpenAI
from dataclasses import dataclass

@dataclass
class Eval:
    history: str


class OpenAIProvider():
    def __init__ (self, api_key: str, model: str = "chatgpt-4o-latest"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def create_doc(self, history: str):
        prompt = f"""

        <Role>You are a physical therapist working in a clinic</Role>
        <Purpose>You have to write the history of a patient based on raw notes</Purpose>

        <History>
        <InputFormat>comma separated facts about patients. It might include - Since when the pain started, how it started etc.</InputFormat>
        <OutputFormat>Well written paragraph of the history of patient based on the comma separated facts</Output>
        <Constraint>Return only the paragraph</Constraints>
        <Input>
        {history}
        </Input>
        </History>
        """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return Eval(
            history=response.choices[0].message.content.strip(),
        )


class PTAssist():
    def __init__(self, provider: OpenAIProvider):
        self.provider = provider

    def create_eval(self):
        history = input("\n Enter history\n")
        evaluation = self.provider.create_doc(history)
        print(f"\n{evaluation.history}\n")


def main():
    dotenv.load_dotenv()
    provider = lambda: OpenAIProvider(os.getenv('OPENAI_API_KEY'))
    pt_assist = PTAssist(provider())
    pt_assist.create_eval()

if __name__ == "__main__":
    main()

