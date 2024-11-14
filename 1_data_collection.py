from Bio import Entrez
import csv

# Configure your email and API key
Entrez.email = "atul2002chourasia@gmail.com"
Entrez.api_key = "f7a08174a11fe23e9efbe66947d1ff4e3208"  # Use your API key

def search(query):
    try:
        handle = Entrez.esearch(db='pubmed', 
                                sort='relevance', 
                                retmax='1000',  # Adjusted to retrieve 200 results
                                retmode='xml', 
                                term=query)
        results = Entrez.read(handle)
        handle.close()
        return results
    except Exception as e:
        print(f"An error occurred during search: {e}")
        return None

def fetch_details(id_list):
    try:
        ids = ','.join(id_list)
        handle = Entrez.efetch(db='pubmed',
                               retmode='xml',
                               id=ids)
        results = Entrez.read(handle)
        handle.close()
        return results
    except Exception as e:
        print(f"An error occurred while fetching details: {e}")
        return None

# Prepare CSV file to save the data
with open('pubmed_articles.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    # Write header
    writer.writerow(['Title', 'Authors', 'Abstract', 'Publication Date', 'Journal', 'Keywords', 'MeSH Terms', 'DOI', 'Author Affiliations'])

    # Example usage
    query_results = search('medical papers')
    if query_results:
        id_list = query_results['IdList']
        papers = fetch_details(id_list)
        if papers:
            # Process the papers and save data to CSV
            for paper in papers['PubmedArticle']:
                try:
                    title = paper['MedlineCitation']['Article']['ArticleTitle']
                    authors = ', '.join([author['ForeName'] + ' ' + author['LastName'] for author in paper['MedlineCitation']['Article']['AuthorList'] if 'ForeName' in author and 'LastName' in author]) if 'AuthorList' in paper['MedlineCitation']['Article'] else 'No authors listed'
                    abstract = paper['MedlineCitation']['Article']['Abstract']['AbstractText'][0] if 'Abstract' in paper['MedlineCitation']['Article'] and 'AbstractText' in paper['MedlineCitation']['Article']['Abstract'] else 'No abstract available'
                    journal = paper['MedlineCitation']['Article']['Journal']['Title']
                    doi = paper['MedlineCitation']['Article']['ELocationID'][0] if 'ELocationID' in paper['MedlineCitation']['Article'] and paper['MedlineCitation']['Article']['ELocationID'] else 'No DOI available'
                    keywords = ', '.join(paper['MedlineCitation']['KeywordList'][0]) if 'KeywordList' in paper['MedlineCitation'] and paper['MedlineCitation']['KeywordList'] and paper['MedlineCitation']['KeywordList'][0] else 'No keywords'
                    mesh_terms = ', '.join([mesh['DescriptorName'] for mesh in paper['MedlineCitation']['MeshHeadingList']]) if 'MeshHeadingList' in paper['MedlineCitation'] else 'No MeSH terms'
                    affiliations = ', '.join([author['AffiliationInfo'][0]['Affiliation'] for author in paper['MedlineCitation']['Article']['AuthorList'] if 'AffiliationInfo' in author and author['AffiliationInfo']]) if 'AuthorList' in paper['MedlineCitation']['Article'] else 'No affiliations available'
                    pub_date = paper['MedlineCitation']['Article']['ArticleDate'][0]['Year'] if 'ArticleDate' in paper['MedlineCitation']['Article'] and paper['MedlineCitation']['Article']['ArticleDate'] else 'No date available'
                    # Write to CSV
                    writer.writerow([title, authors, abstract, pub_date, journal, keywords, mesh_terms, doi, affiliations])
                except Exception as e:
                    print(f"An error occurred processing one of the papers: {e}")
    else:
        print("No results returned from query.")
