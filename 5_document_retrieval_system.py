import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import torch
from transformers import AutoTokenizer, AutoModel

class DocumentRetrievalSystem:
    def __init__(self, model_type='tfidf'):
        # Load processed documents
        self.documents = pd.read_csv('processed_documents.csv')
        
        # Model selection
        self.model_type = model_type
        if model_type == 'tfidf':
            self.load_tfidf_model()
        elif model_type == 'biobert':
            self.load_biobert_model()
        else:
            raise ValueError("Model type not supported. Choose 'tfidf' or 'biobert'.")

    def load_tfidf_model(self):
        # Load TF-IDF matrix and vectorizer
        with open('tfidf_matrix.pkl', 'rb') as f:
            self.tfidf_matrix = pickle.load(f)
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            self.vectorizer = pickle.load(f)

    def load_biobert_model(self):
        # Load BioBERT embeddings
        with open('biobert_embeddings.pkl', 'rb') as f:
            self.biobert_embeddings = pickle.load(f)
        # Initialize BioBERT model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-v1.1')
        self.model = AutoModel.from_pretrained('dmis-lab/biobert-v1.1')

    def preprocess_query(self, query):
        """Preprocess the query to match the document preprocessing."""
        query = query.lower()
        return ' '.join([word for word in query.split() if word.isalpha()])

    def get_query_embedding(self, query):
        """Get embedding for the query using BioBERT."""
        encoded_input = self.tokenizer(query, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**encoded_input)
        return outputs.last_hidden_state.mean(1).squeeze().numpy()

    def retrieve_by_tfidf(self, query):
        # Transform the query using TF-IDF vectorizer
        query_vec = self.vectorizer.transform([query])
        cosine_sim = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_docs_indices = np.argsort(cosine_sim)[-5:][::-1]  # Get top 5 documents, sorted by similarity
        return self.documents.iloc[top_docs_indices]

    def retrieve_by_biobert(self, query):
        # Get BioBERT embedding for the query
        query_embedding = self.get_query_embedding(query)
        cosine_sim = cosine_similarity([query_embedding], self.biobert_embeddings).flatten()
        top_docs_indices = np.argsort(cosine_sim)[-5:][::-1]  # Get top 5 documents, sorted by similarity
        return self.documents.iloc[top_docs_indices]

    def query(self, input_query):
        """Retrieve relevant documents based on the chosen model."""
        query = self.preprocess_query(input_query)
        
        if self.model_type == 'tfidf':
            results = self.retrieve_by_tfidf(query)
        elif self.model_type == 'biobert':
            results = self.retrieve_by_biobert(query)
        
        return results

# Example usage
if __name__ == "__main__":
    # Choose between 'tfidf' and 'biobert' models
    retrieval_system = DocumentRetrievalSystem(model_type='tfidf')  # or model_type='biobert'
    
    # Input a query
    query = "humans"
    results = retrieval_system.query(query)
    
    # Display results
    print("Top Search Results:")
    for index, row in results.iterrows():
        print(f"\nTitle: {row['Title']}")
        print(f"Abstract: {row['Abstract'][:200]}...")  # Displaying first 200 chars of abstract
        print(f"Keywords: {row['Keywords']}")
        print(f"Journal: {row['Journal']}")
