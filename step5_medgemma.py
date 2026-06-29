import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()


class MedGemmaClient:

    def __init__(self):

        self.endpoint = os.getenv(
            "MEDGEMMA_API_URL"
        )

        self.model = os.getenv(
            "MEDGEMMA_MODEL"
        )

        self.token = os.getenv(
            "MEDGEMMA_API_TOKEN"
        )

    def summarize(
        self,
        transcript,
        system_prompt
    ):

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        transcript,
                        ensure_ascii=False
                    )
                }
            ]
        }

        headers = {
            "Content-Type": "application/json"
        }

        if self.token:
            headers["Authorization"] = (
                f"Bearer {self.token}"
            )

        response = requests.post(
            self.endpoint,
            headers=headers,
            json=payload,
            timeout=300
        )

        response.raise_for_status()

        return response.json()