import os
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter

def ensure_directory(directory_name):
    """Ensure the directory exists, and if not, create it."""
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    return directory_name

def load_data(filepath):
    return pd.read_csv(filepath)

def count_terms(data, column):
    terms = []
    for items in data[column].dropna():
        terms.extend([item.strip() for item in items.split(',')])
    return Counter(terms)

def generate_word_cloud(term_counts, title, filepath):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(term_counts)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title)
    plt.axis('off')
    plt.savefig(filepath)  # Save the figure with full path
    plt.show()

def plot_histogram(data, column, title, xlabel, ylabel, filepath):
    plt.figure(figsize=(10, 5))
    data[column].value_counts().sort_index().plot(kind='bar')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filepath)  # Save the figure with full path
    plt.show()

def plot_bar_chart(series, title, xlabel, ylabel, filepath):
    series.plot(kind='bar')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha='right')
    plt.savefig(filepath)  # Save the figure with full path
    plt.show()

def main():
    # Directory for saving visualization results
    vis_dir = ensure_directory('visualization_results')
    
    filepath = 'pubmed_articles_cleaned.csv'
    data = load_data(filepath)
    
    data['Publication Date'] = pd.to_datetime(data['Publication Date']).dt.year
    
    keyword_counts = count_terms(data, 'Keywords')
    mesh_counts = count_terms(data, 'MeSH Terms')

    # Generate and save visualizations with paths
    generate_word_cloud(keyword_counts, 'Most Common Keywords', os.path.join(vis_dir, 'keyword_cloud.png'))
    generate_word_cloud(mesh_counts, 'Most Common MeSH Terms', os.path.join(vis_dir, 'mesh_term_cloud.png'))
    
    plot_histogram(data, 'Publication Date', 'Publication Trend Over Years', 'Year', 'Number of Publications', os.path.join(vis_dir, 'publication_trend.png'))
    
    top_journals = data['Journal'].value_counts().head(10)
    plot_bar_chart(top_journals, 'Top Journals by Number of Publications', 'Journal', 'Number of Publications', os.path.join(vis_dir, 'top_journals.png'))

if __name__ == '__main__':
    main()
