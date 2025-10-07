from arxiv_loader import *
from neo4j_loader import *
from arxiv_ner import *

def main():
    doc_list, doc_details = get_arxiv_documents()
    print("Arxiv documents loaded")
    doc_details = group_by_entity(doc_details)
    print("NER done")
    chunks = create_chunks(doc_list)
    print("Chunks created")
    driver = connect_to_neo4j()
    data_ingestion(doc_details,driver)
    print("Data ingested")
    create_vector_index(driver)
    print("Create vector index")
    ingest_chunks_embeddings(driver,chunks)
    print("Chunks embedding ingested")

if __name__ == "__main__":
    main()