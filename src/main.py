import os
import pdfplumber
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from tqdm import tqdm
from src.config import OPENAI_API_KEY

print(f"Using OpenAI API Key: {OPENAI_API_KEY}")


# Step 1: Initialize Embedding Model and Vector DB
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Example pre-trained model
    
class EmbeddingFunctionWrapper:
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=False).tolist()  # Convert to list

    def embed_query(self, text):
        # Ensure the text is a string
        if not isinstance(text, str):
            raise ValueError("The query must be a string.")
        return self.model.encode([text], show_progress_bar=False)[0].tolist()  # Convert to list


# Initialize Chroma Vector DB
vector_db_dir = "./vector_db"
embedding_fn = EmbeddingFunctionWrapper(embedding_model)
vector_db = Chroma(persist_directory=vector_db_dir, embedding_function=embedding_fn)


# Initialize the Chroma client
client = chromadb.PersistentClient(path=vector_db_dir)

# Initialize the LangChain Chroma wrapper, passing the client
vector_db = Chroma(client=client,
                  embedding_function=embedding_fn,
                  persist_directory=vector_db_dir)

# Step 2: Document Scanner
def scan_document(file_path):
    """Reads content from a file (PDF or TXT)."""
    if file_path.endswith(".pdf"):
        with pdfplumber.open(file_path) as pdf:
            text = " ".join(page.extract_text() for page in pdf.pages)
    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
    else:
        raise ValueError("Unsupported file format. Use PDF or TXT.")
    return text

# Step 3: Chunkifier
def chunk_text(text):
    """Chunks text into smaller segments using a text splitter."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Adjust chunk size based on your use case
        chunk_overlap=200  # Overlap between chunks for context
    )
    chunks = text_splitter.split_text(text)
    return chunks

def process_and_store_document(file_path):

    # Scan the document
    text = scan_document(file_path)
    print(f"Scanned {len(text)} characters from {file_path}")

    # Chunk the text
    chunks = chunk_text(text)
    print(f"Chunked into {len(chunks)} segments")

    # ... (Your document scanning and chunking logic) ...

    for idx, chunk in enumerate(tqdm(chunks, desc="Adding chunks to vector DB")):
        metadata = {"source": file_path, "chunk_index": idx}
        vector_db.add_texts([chunk], metadatas=[metadata])

    # Persist changes using the client's persist method
    print("Document processed and stored in vector DB!")


def query_vector_db(query, top_k=5):
    """
    Retrieve the top_k most relevant documents for the given query.
    """
    print(f"Query type: {type(query)}")  # Debugging line
    # Embed the query
    #query_embedding = embedding_fn.embed_query(query)
    
    # Perform similarity search
    results = vector_db.similarity_search(query, k=top_k)
    
    for i, result in enumerate(results):
        print(f"Result {i + 1}:")
        print(f"Text: {result.page_content}")    # Access the text content of the Document
        print(f"Metadata: {result.metadata}")    # Access metadata of the Document
        print()
    
    return results


if __name__ == "__main__":
    # Example PDF or text file path
    file_path = "example_document.txt"  # Replace with your document
    process_and_store_document(file_path)
    
    # Example query
    query = "Explain the main concept discussed in the document."
    print("Querying the Vector DB...")
    retrieved_results = query_vector_db(query, top_k=5)
