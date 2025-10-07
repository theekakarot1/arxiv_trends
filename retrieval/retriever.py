from retrieval.embedder import embedding_model
from ingestion.neo4j_loader import *

def get_chunks_from_neo4j(driver, user_query, k):
    """
    Performs a vector search in Neo4j to retrieve the top k initial chunks.
    """
    if not driver:
        print("Neo4j driver is not connected.")
        return []

    # 1. Generate embedding for the user query
    user_query_embedding = embedding_model.encode(user_query).tolist()

    # 2. Cypher query for vector search
    query = """
    CALL db.index.vector.queryNodes('chunk-embeddings', $k, $user_query_embedding)
    YIELD node AS chunk, score
    MATCH (chunk)<-[:HAS_CHUNK]-(p:Paper)
    RETURN
      chunk.text AS text,
      score,
      p.title AS paperTitle,
      p.entry_id AS paperId
    ORDER BY score DESC
    LIMIT $k;
    """
    parameters = {
        "user_query_embedding": user_query_embedding,
        "k": k
    }

    try:
        # print(f"Retrieving top {k} candidate chunks from Neo4j...")
        records, summary, keys = run_cypher_query(driver, query, parameters)
        
        results = [
            {
                "text": record["text"],
                "score": record["score"],
                "metadata": {
                    "paperTitle": record["paperTitle"],
                    "paperId": record["paperId"]
                }
            } for record in records
        ]
        
        # print(f"Successfully retrieved {len(results)} chunks.")
        return results
    except Exception as e:
        print(f"Error during Neo4j retrieval: {e}")
        return []