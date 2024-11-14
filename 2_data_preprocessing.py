import pandas as pd

def load_data(file_path):
    """Load the data from a CSV file."""
    return pd.read_csv(file_path)

def check_data_integrity(data):
    """Print initial data and check for missing values."""
    print("Initial Data Overview:")
    print(data.head())
    print("\nMissing Values Check:")
    print(data.isnull().sum())

def handle_missing_values(data):
    """Handle missing values by filling them with placeholders or median values."""
    data['Authors'].fillna('No authors listed', inplace=True)
    data['Abstract'].fillna('No abstract available', inplace=True)
    data['Keywords'].fillna('No keywords', inplace=True)
    data['MeSH Terms'].fillna('No MeSH terms', inplace=True)
    data['DOI'].fillna('No DOI available', inplace=True)
    data['Author Affiliations'].fillna('No affiliations available', inplace=True)

def standardize_text_data(data):
    """Standardize text fields to ensure consistency."""
    data['Journal'] = data['Journal'].str.title().str.strip()
    data['Authors'] = data['Authors'].str.replace('.', '').str.title()  # Remove periods and standardize case
    data['Keywords'] = data['Keywords'].str.lower()
    data['MeSH Terms'] = data['MeSH Terms'].str.lower()

def correct_date_formats(data):
    """Correct and standardize date formats."""
    data['Publication Date'] = pd.to_datetime(data['Publication Date'], errors='coerce')

def remove_duplicates(data):
    """Remove duplicate entries based on DOI."""
    return data.drop_duplicates(subset=['DOI'])

def save_cleaned_data(data, output_path):
    """Save the cleaned data to a new CSV file."""
    data.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")

def main():
    # Set file paths
    input_file_path = 'pubmed_articles.csv'
    output_file_path = 'pubmed_articles_cleaned.csv'
    
    # Load data
    data = load_data(input_file_path)
    
    # Check data integrity
    check_data_integrity(data)
    
    # Handle missing values
    handle_missing_values(data)
    
    # Standardize text data
    standardize_text_data(data)
    
    # Correct date formats
    correct_date_formats(data)
    
    # Remove duplicates
    data = remove_duplicates(data)
    
    # Save cleaned data
    save_cleaned_data(data, output_file_path)

if __name__ == '__main__':
    main()
