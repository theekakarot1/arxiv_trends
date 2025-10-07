from langchain_text_splitters import RecursiveCharacterTextSplitter
import nest_asyncio
from neo4j import GraphDatabase, exceptions
nest_asyncio.apply()
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from collections import defaultdict

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


def create_chunks(doc_list):
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=32768,
    chunk_overlap=500,
    length_function=len,
    separators=["\n\n", "\n", " ", ""],
    is_separator_regex=False
    )
    doc_list = [i[0] for i in doc_list]
    chunks = text_splitter.split_documents(doc_list)
    return chunks

def connect_to_neo4j():
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        print("Connection to Neo4j established successfully.")
        return driver
    except exceptions.ServiceUnavailable as e:
        print(f"Failed to connect to Neo4j at {NEO4J_URI}. Error: {e}")
        return None
    
def close_neo4j_driver(driver):
    if driver:
        driver.close()
        print("Neo4j connection closed.")

def run_cypher_query(driver, query, parameters=None):
    if not driver:
        return None, None, None
    records, summary, keys = driver.execute_query(
        query,
        parameters_=parameters or {},
        database_="neo4j",
    )
    return records, summary, keys

def data_ingestion(doc_details,driver):
    for doc in doc_details:
        doc = defaultdict(lambda: None,doc)
        entry_id = doc['entry_id']
        authors = doc['Authors'].split(", ")
        categories = doc['categories']
        model_algorithms = doc["Models & Algorithms"]
        datasets = doc["Datasets"]
        metrics = doc["Metrics"]
        libraries_frameworks = doc["Libraries & Frameworks"]
        tasks = doc["Tasks"]
        theories_concepts = doc["Theories & Concepts"]
        institutions = doc["Institutions"]

        # ingest paper node
        paper_query = """
            MERGE (p:Paper {entry_id: $entry_id})
            ON CREATE SET
            p.title = $title,
            p.published = datetime($published),
            p.summary = $summary,
            p.doi = $doi,
            p.journal_ref = $journal_ref
            """
        paper_params = {
            "entry_id": entry_id,
            "title": doc['Title'],
            "published": doc['Published'],
            "summary": doc['Summary'],
            "doi": doc['doi'],
            "journal_ref" : doc['journal_ref']

        }
        run_cypher_query(driver, paper_query, paper_params)

        # Ingest Authors and their relationship to the Paper
        if authors:
            authors_query = """
            MATCH (p:Paper {entry_id: $entry_id})
            UNWIND $authorNames AS authorName
            MERGE (a:Author {name: authorName})
            MERGE (p)-[:WRITTEN_BY]->(a)
            """
            authors_params = {"entry_id": entry_id, "authorNames": authors}
            run_cypher_query(driver, authors_query, authors_params)

        # Ingest Categories and their relationship to the Paper
        if categories:
            categories_query = """
            MATCH (p:Paper {entry_id: $entry_id})
            UNWIND $categoryNames AS categoryName
            MERGE (c:Category {name: categoryName})
            MERGE (p)-[:IN_CATEGORY]->(c)
            """
            categories_params = {"entry_id": entry_id, "categoryNames": categories}
            run_cypher_query(driver, categories_query, categories_params)
        
        if model_algorithms:
            model_algorithms_query = """
            MATCH (p:Paper {entry_id: $entry_id})
            UNWIND $model_algorithms AS model_algorithms_used
            MERGE (m:Model {name: model_algorithms_used})
            MERGE (p)-[:MODEL_ALGORITHM_USED]->(m)
            """
            model_algorithms_params = {"entry_id": entry_id, "model_algorithms": model_algorithms}
            run_cypher_query(driver, model_algorithms_query, model_algorithms_params)
        
        if datasets:
            datasets_query = """
            MATCH (p:Paper {entry_id: $entry_id})
            UNWIND $datasetUsed AS datasetUsed
            MERGE (d:Dataset {name: datasetUsed})
            MERGE (p)-[:DATASET_USED]->(d)
            """
            datasets_params = {"entry_id": entry_id, "datasetUsed": datasets}
            run_cypher_query(driver, datasets_query, datasets_params)
        
        if metrics:
            metrics_query = """
            MATCH (p:Paper {entry_id: $entry_id})
            UNWIND $metricsUsed AS metricsUsed
            MERGE (m:Metrics {name: metricsUsed})
            MERGE (p)-[:METRICS_USED]->(m)
            """
            metrics_params = {"entry_id": entry_id, "metricsUsed": metrics}
            run_cypher_query(driver, metrics_query, metrics_params)
        
        if libraries_frameworks:
            libraries_frameworks_query = """
            MATCH (p:Paper {entry_id: $entry_id})
            UNWIND $libraries_frameworks AS libraries_frameworks_used
            MERGE (f:Libraries {name: libraries_frameworks_used})
            MERGE (p)-[:LIBRARY_FRAMEWORK_USED]->(f)
            """
            libraries_frameworks_params = {"entry_id": entry_id, "libraries_frameworks": libraries_frameworks}
            run_cypher_query(driver, libraries_frameworks_query, libraries_frameworks_params)

        if tasks:
            tasks_query = """
            MATCH (p:Paper {entry_id: $entry_id})
            UNWIND $taskPerformed AS taskPerformed
            MERGE (t:Tasks {name: taskPerformed})
            MERGE (p)-[:TASK_PERFORMED]->(t)
            """
            tasks_params = {"entry_id": entry_id, "taskPerformed": tasks}
            run_cypher_query(driver, tasks_query, tasks_params)

        if theories_concepts:
            theories_concepts_query = """
            MATCH (p:Paper {entry_id: $entry_id})
            UNWIND $theories_concepts AS theories_concepts_used
            MERGE (c:Concepts {name: theories_concepts_used})
            MERGE (p)-[:THEORIES_CONCEPTS_USED]->(c)
            """
            theories_concepts_params = {"entry_id": entry_id, "theories_concepts": theories_concepts}
            run_cypher_query(driver, theories_concepts_query, theories_concepts_params)

        if institutions:
            institutions_query = """
            MATCH (p:Paper {entry_id: $entry_id})
            UNWIND $institutionName AS institutionName
            MERGE (i:Institute {name: institutionName})
            MERGE (p)-[:INSTITUTE]->(i)
            """
            institutions_params = {"entry_id": entry_id, "institutionName": institutions}
            run_cypher_query(driver, institutions_query, institutions_params)
        
    print("Ingestion complete")

def create_vector_index(driver):
    index_creation_query = """
    CREATE VECTOR INDEX `chunk-embeddings` IF NOT EXISTS
    FOR (c:Chunk) ON c.embedding
    OPTIONS {indexConfig: { `vector.dimensions`: 384, `vector.similarity_function`: 'cosine' }}
    """
    run_cypher_query(driver, index_creation_query)

def ingest_chunks_embeddings(driver,chunks):
    chunk_ingestion_query = """
    MATCH (p:Paper {entry_id: $parent_paper_id})
    CREATE (c:Chunk {
        text: $chunk_text,
        embedding: $chunk_embedding
    })
    CREATE (p)-[:HAS_CHUNK]->(c)
    """
    for i, chunk in enumerate(chunks):
        embedding = embedding_model.encode(chunk.page_content).tolist()
        
        chunk_params = {
            "parent_paper_id": chunk.metadata.get('entry_id'),
            "chunk_text": chunk.page_content,
            "chunk_embedding": embedding,
        }
        run_cypher_query(driver, chunk_ingestion_query, chunk_params)



