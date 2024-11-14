import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk
from transformers import AutoTokenizer, AutoModel
import torch
import pickle

# Ensure necessary NLTK resources are available
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Setup BioBERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-v1.1')
model = AutoModel.from_pretrained('dmis-lab/biobert-v1.1')

def preprocess_text(text):
    """ Tokenize, remove stopwords, apply stemming and lemmatization. """
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha() and t not in stopwords.words('english')]
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(stemmer.stem(t)) for t in tokens]
    return ' '.join(tokens)

def create_tfidf_features(corpus):
    """ Generate a TF-IDF matrix from preprocessed text. """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return tfidf_matrix, vectorizer.get_feature_names_out(), vectorizer  # Return vectorizer to save later

def get_biobert_embeddings(texts):
    """ Generate embeddings for a list of texts using BioBERT. """
    embeddings = []
    for text in texts:
        encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**encoded_input)
        embeddings.append(outputs.last_hidden_state.mean(1).squeeze().numpy())
    return embeddings

def main():
    # Load your dataset
    df = pd.read_csv('pubmed_articles_cleaned.csv')  # Ensure your dataset path is correct
    
    # Preprocess text fields for TF-IDF and embedding generation
    df['processed_text'] = df['Abstract'].apply(preprocess_text) + " " + df['Title'].apply(preprocess_text)
    
    # Generate a unique ID column if not present
    df['ID'] = df.index

    # Save processed document texts with additional metadata
    df[['ID', 'Title', 'Abstract', 'Keywords', 'Journal', 'processed_text']].to_csv('processed_documents.csv', index=False)
    
    # Generate and save TF-IDF features
    tfidf_matrix, tfidf_features, vectorizer = create_tfidf_features(df['processed_text'])
    print("TF-IDF Matrix Shape:", tfidf_matrix.shape)
    
    # Save TF-IDF matrix and vectorizer
    with open('tfidf_matrix.pkl', 'wb') as f:
        pickle.dump(tfidf_matrix, f)
    with open('tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    # Optionally, generate BioBERT embeddings (can be computationally intensive)
    biobert_embeddings = get_biobert_embeddings(df['processed_text'].sample(min(50, len(df))))
    with open('biobert_embeddings.pkl', 'wb') as f:
        pickle.dump(biobert_embeddings, f)
    print("BioBERT Embeddings Shape:", len(biobert_embeddings))

if __name__ == '__main__':
    main()
