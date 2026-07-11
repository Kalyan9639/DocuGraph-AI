import os
import urllib.request
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from agno.knowledge.embedder.base import Embedder
from typing import List, Tuple, Optional, Dict
import asyncio
import streamlit as st


MODEL_DIR = "tmp/granite_onnx"
ONNX_PATH = os.path.join(MODEL_DIR, "model_quantized.onnx")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.json")
BASE_URL = "https://huggingface.co/onnx-community/granite-embedding-30m-english-ONNX/resolve/main"


@st.cache_resource(show_spinner="📥 Loading Granite Embedding model (one-time)...")
def _load_granite_onnx():
    """
    Downloads and loads the Granite ONNX model and tokenizer.
    Cached by Streamlit so this runs only once per cold start, not per user session.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(ONNX_PATH):
        print("📥 Downloading Granite ONNX model weights...")
        urllib.request.urlretrieve(f"{BASE_URL}/onnx/model_quantized.onnx", ONNX_PATH)
    if not os.path.exists(TOKENIZER_PATH):
        print("📥 Downloading Granite Tokenizer...")
        urllib.request.urlretrieve(f"{BASE_URL}/tokenizer.json", TOKENIZER_PATH)

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    tokenizer.enable_truncation(max_length=512)
    session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    return tokenizer, session


class OnnxGraniteEmbedder(Embedder):
    """
    Lightweight, fast IBM Granite Embeddings using ONNX Runtime.
    Bypasses PyTorch/Transformers dependencies completely.
    Model is cached by Streamlit so it is only loaded once per cold start.
    """
    def __init__(self):
        super().__init__(dimensions=384)
        self.tokenizer, self.session = _load_granite_onnx()

    def get_embedding(self, text: str) -> List[float]:
        if not text.strip():
            return [0.0] * self.dimensions

        encoding = self.tokenizer.encode(text)
        input_ids = np.array([encoding.ids], dtype=np.int64)
        attention_mask = np.array([encoding.attention_mask], dtype=np.int64)

        input_names = [inp.name for inp in self.session.get_inputs()]
        onnx_inputs = {
            input_names[0]: input_ids,
            input_names[1]: attention_mask
        }

        outputs = self.session.run(None, onnx_inputs)

        # Mean pooling across token embeddings, respecting attention mask
        last_hidden_state = outputs[0]  # (1, seq_len, 384)
        mask = np.expand_dims(attention_mask, axis=-1)  # (1, seq_len, 1)
        masked_states = last_hidden_state * mask
        sum_masked = np.sum(masked_states, axis=1)  # (1, 384)
        sum_mask = np.maximum(np.sum(mask, axis=1), 1e-9)  # (1, 1)
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

