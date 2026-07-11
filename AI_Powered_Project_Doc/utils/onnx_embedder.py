import os
import urllib.request
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from agno.knowledge.embedder.base import Embedder
from typing import List, Tuple, Optional, Dict
import asyncio


class OnnxGraniteEmbedder(Embedder):
    """
    Lightweight, fast IBM Granite Embeddings using ONNX Runtime.
    Bypasses PyTorch/Transformers dependencies completely.
    """
    def __init__(self):
        super().__init__(dimensions=384)
        self.model_dir = "tmp/granite_onnx"
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.onnx_path = os.path.join(self.model_dir, "model_quantized.onnx")
        self.tokenizer_path = os.path.join(self.model_dir, "tokenizer.json")
        
        # Download ONNX weights and tokenizer config if not cached
        self._download_files()
        
        # Load the components
        self.tokenizer = Tokenizer.from_file(self.tokenizer_path)
        self.tokenizer.enable_truncation(max_length=512)
        
        self.session = ort.InferenceSession(self.onnx_path, providers=["CPUExecutionProvider"])
        
    def _download_files(self):
        base_url = "https://huggingface.co/onnx-community/granite-embedding-30m-english-ONNX/resolve/main"
        if not os.path.exists(self.onnx_path):
            print("📥 Downloading Granite ONNX model...")
            urllib.request.urlretrieve(f"{base_url}/onnx/model_quantized.onnx", self.onnx_path)
        if not os.path.exists(self.tokenizer_path):
            print("📥 Downloading Granite Tokenizer...")
            urllib.request.urlretrieve(f"{base_url}/tokenizer.json", self.tokenizer_path)

    def get_embedding(self, text: str) -> List[float]:
        if not text.strip():
            return [0.0] * self.dimensions
            
        encoding = self.tokenizer.encode(text)
        input_ids = np.array([encoding.ids], dtype=np.int64)
        attention_mask = np.array([encoding.attention_mask], dtype=np.int64)
        
        # Gather model entry nodes
        input_names = [inputs.name for inputs in self.session.get_inputs()]
        
        # Prepare inputs based on ONNX expectations
        onnx_inputs = {
            input_names[0]: input_ids,
            input_names[1]: attention_mask
        }
        
        outputs = self.session.run(None, onnx_inputs)
        
        # Mean pooling to extract sequence representation respecting attention mask
        last_hidden_state = outputs[0]  # Shape: (1, sequence_length, 384)
        
        mask = np.expand_dims(attention_mask, axis=-1)  # (1, sequence_length, 1)
        masked_states = last_hidden_state * mask
        sum_masked = np.sum(masked_states, axis=1)  # (1, 384)
        sum_mask = np.sum(mask, axis=1)  # (1, 1)
        sum_mask = np.maximum(sum_mask, 1e-9)  # Avoid div-by-zero
        
        mean_pooled = sum_masked / sum_mask
        return mean_pooled[0].tolist()

    def get_embedding_and_usage(self, text: str) -> Tuple[List[float], Optional[Dict]]:
        return self.get_embedding(text), None

    async def async_get_embedding(self, text: str) -> List[float]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.get_embedding, text)

    async def async_get_embedding_and_usage(self, text: str) -> Tuple[List[float], Optional[Dict]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.get_embedding_and_usage, text)
