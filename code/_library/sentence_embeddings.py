import torch
from transformers import AutoModel, AutoTokenizer
from InstructorEmbedding import INSTRUCTOR
from sentence_transformers import SentenceTransformer
import numpy as np
from more_itertools import flatten, batched
from tqdm import tqdm

import torch.nn.functional as F



class SGPT_SentenceEmbedding():
    
    def __init__(self, model_version = 'base') -> None:
        model_version_names = {'base': '125M', 'large': '2.7B', 'xl': '5.8B'}
        
        self.model_path = f"Muennighoff/SGPT-{model_version_names[model_version]}-weightedmean-msmarco-specb-bitfit"
        self._target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModel.from_pretrained(self.model_path).to(self._target_device)
        
        self.max_seq_length = self.model.config.max_position_embeddings

        # Activate evaluation mode - Deactivate Dropout
        self.model.eval()
        
    def get_sentence_embedding_dimension(self):
        return self.model.config.hidden_size
        

    def _tokenize_with_specb(self, texts, is_query):
        
        SPECB_QUE_BOS = self.tokenizer.encode("[", add_special_tokens=False)[0]
        SPECB_QUE_EOS = self.tokenizer.encode("]", add_special_tokens=False)[0]

        SPECB_DOC_BOS = self.tokenizer.encode("{", add_special_tokens=False)[0]
        SPECB_DOC_EOS = self.tokenizer.encode("}", add_special_tokens=False)[0]
    
        # Tokenize without padding
        batch_tokens = self.tokenizer(texts, padding = False, truncation = True)
        
        # Add special brackets & pay attention to them
        for seq, att in zip(batch_tokens["input_ids"], batch_tokens["attention_mask"]): # type: ignore            
            if is_query:
                seq.insert(0, SPECB_QUE_BOS)
                seq.append(SPECB_QUE_EOS)
            else:
                seq.insert(0, SPECB_DOC_BOS)
                seq.append(SPECB_DOC_EOS) 
            att.insert(0, 1)
            att.append(1)
            
        # Add padding
        batch_tokens = self.tokenizer.pad(batch_tokens, padding=True, return_tensors="pt")
        
        # Send to device
        batch_tokens = batch_tokens.to(self._target_device)
        
        return batch_tokens
    
    def _get_weightedmean_embedding(self, batch_tokens):
        
        # Get the embeddings
        with torch.no_grad():
            
            # Get hidden state of shape [bs, seq_len, hid_dim]
            last_hidden_state = self.model(**batch_tokens, output_hidden_states=True, return_dict=True).last_hidden_state

        # Get weights of shape [bs, seq_len, hid_dim]
        weights = (
            torch.arange(start=1, end=last_hidden_state.shape[1] + 1)
            .unsqueeze(0)
            .unsqueeze(-1)
            .expand(last_hidden_state.size())
            .float().to(last_hidden_state.device)
        )

        # Get attn mask of shape [bs, seq_len, hid_dim]
        input_mask_expanded = (
            batch_tokens["attention_mask"]
            .unsqueeze(-1)
            .expand(last_hidden_state.size())
            .float()
        )

        # Perform weighted mean pooling across seq_len: bs, seq_len, hidden_dim -> bs, hidden_dim
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded * weights, dim=1)
        sum_mask = torch.sum(input_mask_expanded * weights, dim=1)

        embeddings = sum_embeddings / sum_mask

        return embeddings
    
    def encode(self, texts, is_query) -> torch.Tensor:
        if isinstance(texts, str):
            texts = [texts]
        
        batch_tokens = self._tokenize_with_specb(texts, is_query)
        embeddings = self._get_weightedmean_embedding(batch_tokens)
        return embeddings
    
    def encode_batch(self, texts, batch_size = 8, is_query = False) -> torch.Tensor:
        if isinstance(texts, str):
            texts = [texts]
        
        sentence_embeddings = []
        for batch in tqdm(batched(texts, batch_size), total = len(texts) // batch_size, desc = "Batch embedding"):
            sentence_embeddings.extend(self.encode(batch, is_query = is_query))
        sentence_embeddings = torch.vstack(sentence_embeddings)
        
        return sentence_embeddings

    
class SentenceInstructorEmbedding():
    
    FINANCE_QUERY_INSTUCTION = "Represent the Financial topic for retrieving supporting documents: "
    FINANCE_DOCUMENT_INSTUCTION = "Represent the Financial document for retrieval: "
    
    def __init__(self, model_version = 'base') -> None:
        model_path = f'hkunlp/instructor-{model_version}'
        self.model = INSTRUCTOR(model_path)
        self.max_seq_length = self.model.max_seq_length 
        self.tokenizer = self.model.tokenizer
        
        self._target_device = self.model._target_device
        
    def get_sentence_embedding_dimension(self):
        return self.model.get_sentence_embedding_dimension()
        
        
    def set_instructions(self, query_instruction = FINANCE_QUERY_INSTUCTION, document_instruction = FINANCE_DOCUMENT_INSTUCTION):
        self.query_instruction = query_instruction
        self.document_instruction = document_instruction
        
    def _build_inputs(self, instruction, data)  :
        
        if isinstance(data, str):
            data = [data]
            
        model_inputs = [[instruction, item] for item in data]
        return model_inputs
    
    def encode(self, texts, is_query = False) -> torch.Tensor:
        
        if (not hasattr(self, "query_instruction")) or (not hasattr(self, "document_instruction")):
            self.set_instructions()
        
        if isinstance(texts, str):
            texts = [texts]
        
        # Generate the inputs (instruction, sentence)
        instruction = self.query_instruction if is_query else self.document_instruction
        model_inputs = self._build_inputs(instruction, texts)
        
        # Get the embeddings for each batch
        sentence_embeddings = self.model.encode(model_inputs) # type: ignore
        sentence_embeddings = torch.as_tensor(sentence_embeddings)
        
        return sentence_embeddings
    
    def encode_batch(self, texts, batch_size = 8, is_query = False) -> torch.Tensor:
        
        if (not hasattr(self, "query_instruction")) or (not hasattr(self, "document_instruction")):
            self.set_instructions()
        
        # Generate the inputs (instruction, sentence)
        instruction = self.query_instruction if is_query else self.document_instruction
        model_inputs = self._build_inputs(instruction, texts)
        
        # Get the embeddings for each batch
        sentence_embeddings = []
        for batch in tqdm(batched(model_inputs, batch_size), total = len(model_inputs) // batch_size, desc = "Batch embedding"):
            sentence_embeddings.extend(self.model.encode(batch)) # type: ignore

        sentence_embeddings = torch.vstack(sentence_embeddings)
        #sentence_embeddings = torch.as_tensor(sentence_embeddings)
        
        return sentence_embeddings
    
    
class E5_SentenceEmbedding():
    def __init__(self, model_version = 'base') -> None:
        model_path = f'intfloat/e5-{model_version}' # small, base, large
        self._target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(self._target_device) 
        
        self.max_seq_length = self.model.config.max_position_embeddings
        
        # Activate evaluation mode - Deactivate Dropout
        self.model.eval()
        
        
    
    def get_sentence_embedding_dimension(self):
        return self.model.config.hidden_size
    
    def average_pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        last_hidden = last_hidden_states.masked_fill(~ attention_mask[..., None].bool(), 0.0)
        outputs = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]  
        return outputs
    
    def _build_inputs(self, texts, is_query)  :
        
        if isinstance(texts, str):
            texts = [texts]

        prefix = "query: " if is_query else "passage: "
        model_inputs = [prefix + item for item in texts]
        return model_inputs
    
    def encode(self, texts, is_query, normalize_embeddings = False)  -> torch.Tensor:
        input_texts = self._build_inputs(texts, is_query)
        
        # Tokenize the input texts
        batch_dict = self.tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt').to(self._target_device)
        
        outputs = self.model(**batch_dict)
        embeddings = self.average_pool(outputs.last_hidden_state, batch_dict['attention_mask']) # type: ignore
        
        if normalize_embeddings:
            embeddings = F.normalize(embeddings, p=2, dim=1)
            scores = (embeddings[:2] @ embeddings[2:].T) * 100
            print(scores.tolist())
            
        return embeddings
    
    def encode_batch(self, texts, is_query = False, normalize_embeddings = False,  batch_size = 8) -> torch.Tensor:
        sentence_embeddings = []
        for batch in tqdm(batched(texts, batch_size), total = len(texts) // batch_size, desc = "Batch embedding"):
            sentence_embeddings.extend(self.encode(batch, is_query = is_query, normalize_embeddings = normalize_embeddings))
        sentence_embeddings = torch.vstack(sentence_embeddings)
        return sentence_embeddings