import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import os
import streamlit as st
from agno.models.ollama import Ollama
from agno.agent import Agent


class OllamaClient:
    """
    Centralized Ollama Cloud client for Streamlit deployment.
    
    Uses Agno's built-in Ollama Cloud support with API key from Streamlit secrets or env vars.
    """

    def __init__(self):
        self.model_name = "gemma4:31b-cloud"
        self.temperature = 0.3

    def get_model(self):
        # Fallback to environment variables if streamlit secrets are not loaded or lack the key
        api_key = os.getenv("OLLAMA_API_KEY")
        if not api_key:
            try:
                api_key = st.secrets.get("OLLAMA_API_KEY")
            except Exception:
                pass

        return Ollama(
            id=self.model_name,
            api_key=api_key,
            host="https://ollama.com",
            options={
                "temperature": self.temperature
            }
        )

    def get_chat_model(self):
        return self.get_model()

    def verify_connection(self) -> bool:
        try:
            client = self.get_model()
            if not client.api_key or client.api_key == "your_api_key_here":
                return False
            # Wrap the model in an Agent to invoke it correctly
            agent = Agent(model=client)
            test_response = agent.run("say ping", timeout=10)
            return test_response is not None
        except Exception as e:
            return False


ollama_client = OllamaClient()