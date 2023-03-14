import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from typing import Union, Sequence, List, Tuple, AnyStr

class BaseEmbedding(object):
    def __init__(
            self,
            model_name='bert-large-cased',
            seed=42,
        ) -> None:
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def embed_input(
            self, 
            input: Sequence[AnyStr]
        ):
        # Tokenize the input input
        tokens = self.tokenizer(input, 
                                padding=True,
                                truncation=True, 
                                return_tensors='pt')
        # Get the embeddings for the tokens
        with torch.no_grad():
            output = self.model(**tokens)
            result_embeddings = output.last_hidden_state.mean(dim=1).numpy()
        return result_embeddings
