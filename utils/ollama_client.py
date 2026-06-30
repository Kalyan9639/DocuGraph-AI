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

from agno.models.ollama import Ollama
import ollama



class OllamaClient:
    """
    Centralized Ollama client.

    Handles:
    - Model creation
    - Connection verification
    - Model existence verification
    """

    def __init__(self):
        self.model_name = "gemma4:31b-cloud"
        self.base_url = "http://localhost:11434"
        self.temperature = 0.3

    # ------------------------------------------------------------------
    # Return Agno Model
    # ------------------------------------------------------------------

    def get_model(self):
        return Ollama(
            id=self.model_name,
            host=self.base_url,
            options={
                "temperature": self.temperature
            }
        )

    # Compatibility
    def get_chat_model(self):
        return self.get_model()

    # ------------------------------------------------------------------
    # Verify Connection
    # ------------------------------------------------------------------

    def verify_connection(self) -> bool:

        try:

            response = ollama.list()

            models = []

            # Latest ollama python package
            if hasattr(response, "models"):
                models = response.models

            # Older package
            elif isinstance(response, dict):
                models = response.get("models", [])

            # Very old package
            elif isinstance(response, list):
                models = response

            else:
                print(f"Unknown response type : {type(response)}")
                return False

            available_models = []

            print("\n========== Installed Models ==========")

            for m in models:

                if isinstance(m, dict):
                    name = m.get("model") or m.get("name")

                else:
                    name = getattr(m, "model", None)

                    if name is None:
                        name = getattr(m, "name", None)

                if name is None:
                    continue

                name = str(name).strip()

                available_models.append(name)

                print(name)

            print("======================================\n")

            if self.model_name in available_models:
                print(f"✅ Found model : {self.model_name}")
                return True

            print(f"❌ Required model : {self.model_name}")
            print(f"Available models : {available_models}")

            return False

        except Exception as e:
            print(f"❌ Failed to connect to Ollama")
            print(e)
            return False


ollama_client = OllamaClient()