import torch

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers.util import semantic_search

class SemanticSearcher:
    def __init__(self, query_embeddings, documents_embeddings, documents):
        self.documents = documents
        self.query_embeddings = query_embeddings
        self.documents_embeddings = documents_embeddings
    
    def _get_topK_documents(self, topK_inds, verbose = False) -> dict:
        
        related_sentences = dict()
        
        # For each query sentence (default = 1), find the most similar sentences in the corpus
        for related_documents in topK_inds:
            
            # For each id, retrive the corrisponding sentence
            for doc in related_documents:
                related_sentences[self.documents[doc['corpus_id']]] = doc['score']
                
                if verbose:
                    print(f"(id:{doc['corpus_id']})\t{round(doc['score'], 4) }\t{self.documents[doc['corpus_id']]}")
        
        return related_sentences
        
    def _get_cosine_similarity(self, query_embedding, document_embeddings):
        scores = cosine_similarity(query_embedding, document_embeddings).flatten()
        return torch.as_tensor(scores)
     
    def search(self, top_k = 5, visualize_findings = True):
        
        # Find the top k indexes
        topK_ids = semantic_search(
            query_embeddings = self.query_embeddings, 
            corpus_embeddings = self.documents_embeddings, 
            top_k = top_k
        ) # score_function,  corpus_chunk_size  

        # Get the top k sentences according to the indexes
        related_documents = self._get_topK_documents(topK_ids, verbose = False)
        
        # Visualize the related documents
        if visualize_findings:
            print("\n", '-' * 41, f'TOP {len(related_documents)} DOCUMENTS', '-' * 42)
            for idk, (document, score) in enumerate(related_documents.items()):
                print('-' * 45, f'[RANK: {idk + 1}]', '-' * 45)
                print(document)
                print('-' * 43, f'SCORE: {round(score, 4)}', '-' * 43, "\n")
        
        return related_documents