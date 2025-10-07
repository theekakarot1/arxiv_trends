from retrieval.retriever import *
import os
from dotenv import load_dotenv
import cohere
from langchain_core.documents import Document
from ingestion.neo4j_loader import *
load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
co = cohere.ClientV2()
driver = connect_to_neo4j()


def retrieve_and_rerank_documents(driver, user_query):
    initial_chunks = get_chunks_from_neo4j(driver, user_query, k=50)

    if not initial_chunks:
        print("No chunks retrieved from Neo4j.")
        return []
    
    # rerank_client = RerankClient(api_key=COHERE_API_KEY)
    
    # Extract just the text of the chunks for the reranker
    document_texts = [chunk["text"] for chunk in initial_chunks]
    
    
    try:
        rerank_results = co.rerank(
                            model="rerank-english-v3.0", query=user_query, documents=document_texts, top_n=10
                                )
        
        # 3. Process the reranked results
        final_reranked_chunks = []
        for result in rerank_results.results:
            original_index = result.index
            original_chunk = initial_chunks[original_index]
            original_chunk["rerank_score"] = result.relevance_score
            final_reranked_chunks.append(original_chunk)
            
        # print(f"Successfully reranked and selected {len(final_reranked_chunks)} final chunks.")
        return final_reranked_chunks
        
    except Exception as e:
        print(f"Error during Cohere reranking: {e}")
        # Fallback to the top N of the initial retrieval
        return initial_chunks[:10]
    
def get_custom_retriever_documents(input_dict):
    # print("input_dict :",input_dict)
    user_query = input_dict
    retrieved_chunks = retrieve_and_rerank_documents(driver, user_query)
    docs = [Document(page_content=chunk['text'], metadata=chunk['metadata']) for chunk in retrieved_chunks]
    return docs