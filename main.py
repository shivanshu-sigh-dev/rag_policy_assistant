"""
RAG-based Insurance Policy Assistant
=====================================

A Retrieval-Augmented Generation (RAG) system for querying insurance policy documents.

Features:
- PDF text extraction and semantic chunking
- Vector embeddings using OpenAI text-embedding-3-large
- Semantic query caching with ChromaDB
- Cross-encoder re-ranking for improved relevance
- OpenAI GPT-based response generation with streaming support

Author: Shivanshu Kumar Singh
Date: November 25, 2025
"""

###############################################################################
# 1. IMPORTS
###############################################################################
import os
import fitz  # PyMuPDF for PDF processing
import re
import ssl
import httpx
import tiktoken  # Token counting for OpenAI models
import json
import chromadb
import uuid
import time
from typing import List, Dict, Any, Tuple, Generator
from sentence_transformers import CrossEncoder
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# SSL Configuration (for development/testing environments with certificate issues)
# WARNING: Only use in trusted development environments
ssl._create_default_https_context = ssl._create_unverified_context
http_client = httpx.Client(verify=False)

# Initialize OpenAI client with custom HTTP client
openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    http_client=http_client
)

###############################################################################
# 2. EMBEDDING LAYER (EXTRACTION + CLEANING + CHUNKING + EMBEDDING)
###############################################################################

# Initialize OpenAI embedding function for vector generation
embedding_fn = OpenAIEmbeddingFunction(
    model_name="text-embedding-3-large",
    api_key=os.getenv("OPENAI_API_KEY"),
    api_base=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
)

# Initialize Cross-Encoder for re-ranking (ms-marco-electra-base provides best accuracy)
cross_encoder = CrossEncoder('cross-encoder/ms-marco-electra-base')

def extract_pdf_text(pdf_path: str) -> str:
    """
    Extract text content from a PDF file with page markers.
    
    Args:
        pdf_path: Absolute path to the PDF file
        
    Returns:
        Extracted text with <page=N> markers for each page
        
    Example:
        >>> text = extract_pdf_text("policy.pdf")
        >>> print(text[:100])
        <page=1>
        Insurance Policy Document...
    """
    doc = fitz.open(pdf_path)
    full_text = []
    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        full_text.append(f"<page={page_num+1}>\n{text}")
    return "\n".join(full_text)


def clean_text(text: str) -> str:
    """
    Clean and normalize extracted text by removing extra whitespace and fixing hyphenation.
    
    Args:
        text: Raw text extracted from PDF
        
    Returns:
        Cleaned and normalized text
        
    Transformations:
        - Collapse multiple spaces into single space
        - Reduce multiple newlines to double newlines
        - Join hyphenated words split across lines
    """
    text = re.sub(r"[ ]+", " ", text)  # Collapse multiple spaces
    text = re.sub(r"\n{2,}", "\n\n", text)  # Normalize paragraph breaks
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)  # Fix hyphenation
    return text.strip()


def chunk_semantic(text: str, max_tokens: int = 450) -> List[str]:
    """
    Split text into semantic chunks based on page and paragraph structure.
    
    This function implements a semantic chunking strategy that:
    1. Splits text by page markers
    2. Further splits by paragraph boundaries
    3. Combines blocks up to max_tokens while preserving semantic units
    4. Hard-splits oversized blocks that exceed max_tokens
    
    Args:
        text: Cleaned text with page markers
        max_tokens: Maximum tokens per chunk (default: 450)
        
    Returns:
        List of text chunks, each under max_tokens
        
    Example:
        >>> chunks = chunk_semantic(text, max_tokens=500)
        >>> len(chunks)
        57
    """
    enc = tiktoken.get_encoding("cl100k_base")
    
    # Split by page markers
    pages = re.split(r"(?=<page=\d+>)", text)
    blocks = []
    
    # Further split each page by paragraphs
    for page in pages:
        page = page.strip()
        if not page:
            continue
        paras = re.split(r"\n\s*\n+", page)
        for p in paras:
            p = p.strip()
            if p:
                blocks.append(p)

    chunks = []
    current = ""
    current_toks = 0

    # Combine blocks into chunks respecting token limit
    for block in blocks:
        block_toks = len(enc.encode(block))
        
        # Hard-split oversized blocks
        if block_toks > max_tokens:
            if current.strip():
                chunks.append(current.strip())
                current = ""
                current_toks = 0
            tokens = enc.encode(block)
            for i in range(0, len(tokens), max_tokens):
                slice_tokens = tokens[i:i + max_tokens]
                chunks.append(enc.decode(slice_tokens).strip())
            continue

        # Add block to current chunk if it fits
        if current_toks + block_toks <= max_tokens:
            if current:
                current += "\n" + block
            else:
                current = block
            current_toks += block_toks
        else:
            if current.strip():
                chunks.append(current.strip())
            current = block
            current_toks = block_toks

    if current.strip():
        chunks.append(current.strip())

    # Final safety pass: ensure no chunk exceeds max_tokens
    safe_chunks = []
    for c in chunks:
        tok_len = len(enc.encode(c))
        if tok_len <= max_tokens:
            safe_chunks.append(c)
        else:
            tokens = enc.encode(c)
            for i in range(0, len(tokens), max_tokens):
                safe_chunks.append(enc.decode(tokens[i:i + max_tokens]).strip())

    return [c for c in safe_chunks if c.strip()]


def embed_text(texts: List[str], batch_size: int = 50, max_context_tokens: int = 8000) -> List[List[float]]:
    """
    Generate embeddings for text chunks using OpenAI's embedding API.
    
    Handles oversized texts by splitting them and processes in batches to avoid API limits.
    
    Args:
        texts: List of text strings to embed
        batch_size: Number of texts to embed per API call (default: 50)
        max_context_tokens: Maximum tokens per text (default: 8000 for text-embedding-3-large)
        
    Returns:
        List of embedding vectors (each is a list of floats)
        
    Raises:
        ValueError: If no valid text to embed after filtering
        
    Example:
        >>> embeddings = embed_text(["Hello world", "Another text"])
        >>> len(embeddings)
        2
        >>> len(embeddings[0])
        3072
    """
    enc = tiktoken.get_encoding("cl100k_base")
    normalized = []
    
    # Filter and split oversized texts
    for t in texts:
        if not t or not t.strip():
            continue
        toks = enc.encode(t)
        if len(toks) > max_context_tokens:
            for i in range(0, len(toks), max_context_tokens):
                part = enc.decode(toks[i:i + max_context_tokens]).strip()
                if part:
                    normalized.append(part)
        else:
            normalized.append(t.strip())
            
    if not normalized:
        raise ValueError("No valid text to embed")

    # Process in batches to avoid API rate limits
    embeddings = []
    for i in range(0, len(normalized), batch_size):
        batch = normalized[i:i + batch_size]
        batch_embeddings = embedding_fn(batch)
        embeddings.extend(batch_embeddings)
    return embeddings

###############################################################################
# 3. SEMANTIC QUERY CACHE WITH CHROMADB
###############################################################################

# Configuration constants
QUERY_CACHE_TTL_SECONDS = 3600  # 1 hour cache lifetime
SEMANTIC_SIMILARITY_THRESHOLD = 0.88  # Threshold for semantic similarity matching

class QueryCache:
    """
    Semantic query cache using ChromaDB for similarity-based query matching.
    
    Instead of exact string matching, this cache uses vector similarity to match
    semantically similar queries (e.g., "What is premium?" and "How much is premium?")
    
    Attributes:
        collection: ChromaDB collection for storing query cache
        ttl_seconds: Time-to-live for cached entries in seconds
        
    Example:
        >>> cache = QueryCache(chroma_client)
        >>> query_embedding = embedding_fn(["What is premium?"])[0]
        >>> cache.set("What is premium?", query_embedding, ["result1", "result2"])
        >>> cached = cache.get(query_embedding)
        >>> print(cached["original_query"])
        What is premium?
    """
    
    def __init__(self, chroma_client: chromadb.Client, collection_name: str = "query_cache", ttl_seconds: int = QUERY_CACHE_TTL_SECONDS):
        """
        Initialize the semantic query cache.
        
        Args:
            chroma_client: ChromaDB client instance
            collection_name: Name for the cache collection (default: "query_cache")
            ttl_seconds: Cache entry lifetime in seconds (default: 3600)
        """
        self.collection = chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )
        self.ttl_seconds = ttl_seconds
    
    def get(self, query_embedding: List[float], threshold: float = SEMANTIC_SIMILARITY_THRESHOLD) -> Dict[str, Any]:
        """
        Retrieve cached results for semantically similar queries.
        
        Args:
            query_embedding: Vector embedding of the query
            threshold: Minimum similarity score for cache hit (0.0-1.0)
            
        Returns:
            Dictionary containing:
                - original_query: The cached query text
                - cached_results: List of retrieved documents
                - similarity: Similarity score (0.0-1.0)
                - cached_time: Unix timestamp of cache creation
            Returns None if no similar query found or cache expired
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=1,
            include=["documents", "metadatas", "distances"]
        )
        
        if not results["ids"][0]:
            return None
        
        # Convert cosine distance to similarity: similarity = 1 - (distance/2)
        # Distance ranges: 0 (identical) to 2 (opposite)
        distance = results["distances"][0][0]
        similarity = 1 - (distance / 2)
        
        metadata = results["metadatas"][0][0]
        
        # Check TTL expiration
        cached_time = metadata.get("timestamp", 0)
        if time.time() - cached_time > self.ttl_seconds:
            print(f"✗ Cache expired (age: {time.time() - cached_time:.0f}s).")
            return None
        
        # Check similarity threshold
        if similarity >= threshold:
            print(f"✓ Similarity: {similarity:.4f} >= {threshold:.4f}")
            return {
                "original_query": results["documents"][0][0],
                "cached_results": json.loads(metadata["results"]),
                "similarity": similarity,
                "cached_time": cached_time
            }
        else:
            print(f"✗ Similarity: {similarity:.4f} < {threshold:.4f} (too low for cache hit)")
        
        return None
    
    def set(self, query: str, query_embedding: List[float], results: List[str]) -> None:
        """
        Store query results in the cache.
        
        Args:
            query: Original query text
            query_embedding: Vector embedding of the query
            results: List of retrieved document chunks to cache
        """
        cache_id = str(uuid.uuid4())
        self.collection.add(
            documents=[query],
            embeddings=[query_embedding],
            metadatas=[{
                "timestamp": time.time(),
                "results": json.dumps(results)
            }],
            ids=[cache_id]
        )
    
    def cleanup_expired(self) -> None:
        """
        Remove expired cache entries to free up storage.
        
        Scans all cached entries and deletes those exceeding TTL.
        """
        all_items = self.collection.get(include=["metadatas"])
        expired_ids = []
        now = time.time()
        
        for i, metadata in enumerate(all_items["metadatas"]):
            if now - metadata.get("timestamp", 0) > self.ttl_seconds:
                expired_ids.append(all_items["ids"][i])
        
        if expired_ids:
            self.collection.delete(ids=expired_ids)
            print(f"✓ Cleaned up {len(expired_ids)} expired cache entries.")

###############################################################################
# 4. CROSS-ENCODER RE-RANKING
###############################################################################

def rerank_results(query: str, documents: List[str], top_k: int = 10) -> List[Tuple[str, float]]:
    """
    Re-rank documents using a cross-encoder model for improved relevance.
    
    Cross-encoders evaluate query-document pairs together, capturing interaction
    features that bi-encoders (used in initial retrieval) miss. This two-stage
    approach (fast bi-encoder + accurate cross-encoder) balances speed and accuracy.
    
    Args:
        query: User query string
        documents: List of candidate documents from initial retrieval
        top_k: Number of top results to return after re-ranking (default: 10)
        
    Returns:
        List of (document, relevance_score) tuples, sorted by score (descending)
        
    Example:
        >>> docs = ["Doc A about premium", "Doc B about coverage"]
        >>> reranked = rerank_results("What is premium?", docs, top_k=5)
        >>> for doc, score in reranked:
        ...     print(f"Score: {score:.4f} - {doc[:50]}")
    """
    if not documents:
        return []
    
    # Create query-document pairs for cross-encoder
    pairs = [[query, doc] for doc in documents]
    
    # Get relevance scores
    scores = cross_encoder.predict(pairs)
    
    # Combine and sort by relevance
    doc_score_pairs = list(zip(documents, scores))
    doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
	
    return doc_score_pairs[:top_k]

###############################################################################
# 5. RAG (RETRIEVAL-AUGMENTED GENERATION) WORKFLOW
###############################################################################

def build_rag_prompt(query: str, context_chunks: List[str], max_context_length: int = 3000) -> str:
    """
    Build a prompt for RAG by combining user query with retrieved context.
    
    Creates a structured prompt that:
    - Provides relevant document context
    - Instructs the model to answer based on context only
    - Encourages citation of specific documents
    - Sets a professional, helpful tone
    
    Args:
        query: User's question
        context_chunks: List of relevant document chunks from retrieval
        max_context_length: Maximum characters for context to avoid token limits (default: 3000)
        
    Returns:
        Formatted prompt string ready for LLM input
        
    Example:
        >>> prompt = build_rag_prompt("What is premium?", ["Premium is...", "Cost is..."])
        >>> print(prompt[:100])
        You are a helpful insurance policy assistant...
    """
    # Combine chunks into context, respecting max length
    context = ""
    for i, chunk in enumerate(context_chunks, 1):
        chunk_text = f"\n\n[Document {i}]\n{chunk}"
        if len(context) + len(chunk_text) <= max_context_length:
            context += chunk_text
        else:
            break
    
    prompt = f"""
    You are a helpful insurance policy assistant. Use the following document excerpts to answer the user's question accurately and concisely.
    **CONTEXT:**
    {context}

    **QUESTION:**
    {query}

    **INSTRUCTIONS:**
    - Answer based ONLY on the provided context
    - If the context doesn't contain enough information, say "I don't have enough information in the provided documents to answer that question."
    - Cite specific document sections or page numbers when possible (e.g., "According to Page 2...")
    - Be concise but complete
    - Use a professional and helpful tone

    ANSWER:
    """
    return prompt

def generate_rag_response(query: str, context_chunks: List[str], model: str = "gpt-4o-mini", temperature: float = 0.3, max_tokens: int = 500) -> Dict[str, Any]:
    """
    Generate a response using OpenAI's chat completion API with RAG.
    
    Args:
        query: User's question
        context_chunks: List of relevant document chunks from retrieval
        model: OpenAI model identifier (default: "gpt-4o-mini")
            Options: "gpt-4o-mini" (fast, cheap), "gpt-4o" (better quality), "gpt-3.5-turbo" (legacy)
        temperature: Controls randomness (0.0-1.0, lower = more focused, default: 0.3)
        max_tokens: Maximum tokens in response (default: 500)
        
    Returns:
        Dictionary containing:
            - answer: Generated response text
            - model: Model used
            - tokens_used: Total tokens consumed
            - prompt_tokens: Tokens in prompt
            - completion_tokens: Tokens in response
            - finish_reason: Completion status ("stop", "length", etc.)
            - error: Error message if generation failed
            
    Example:
        >>> result = generate_rag_response("What is premium?", ["Premium is $100"])
        >>> print(result["answer"])
        According to the documents, the premium is $100.
        >>> print(f"Tokens used: {result['tokens_used']}")
        Tokens used: 245
    """
    prompt = build_rag_prompt(query, context_chunks)
    
    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a knowledgeable insurance policy assistant. Provide accurate, helpful answers based on the provided document context."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.9,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        
        return {
            "answer": response.choices[0].message.content.strip(),
            "model": model,
            "tokens_used": response.usage.total_tokens,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "finish_reason": response.choices[0].finish_reason
        }
    except Exception as e:
        return {
            "answer": f"Error generating response: {str(e)}",
            "model": model,
            "tokens_used": 0,
            "error": str(e)
        }


def stream_rag_response(query: str, context_chunks: List[str], model: str = "gpt-4o-mini", temperature: float = 0.3, max_tokens: int = 500) -> Generator[str, None, None]:
    """
    Generate a streaming response using OpenAI's chat completion API.
    
    Yields text chunks as they arrive from the API, allowing for real-time display
    of the response as it's being generated.
    
    Args:
        query: User's question
        context_chunks: List of relevant document chunks from retrieval
        model: OpenAI model identifier (default: "gpt-4o-mini")
        temperature: Controls randomness (0.0-1.0, default: 0.3)
        max_tokens: Maximum tokens in response (default: 500)
    
    Yields:
        Text chunks as they arrive from the API
        
    Example:
        >>> for chunk in stream_rag_response("What is premium?", ["Premium is $100"]):
        ...     print(chunk, end="", flush=True)
        According to the documents, the premium is $100.
    """
    prompt = build_rag_prompt(query, context_chunks)
    
    try:
        stream = openai_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a knowledgeable insurance policy assistant. Provide accurate, helpful answers based on the provided document context."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.9,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
    
    except Exception as e:
        yield f"\n\nError: {str(e)}"

###############################################################################
# 6. MAIN EXECUTION
###############################################################################

if __name__ == "__main__":
    """
    Main execution flow for the RAG system.
    
    Command-line options:
        --perform-embedding : Index the PDF document (required for first run)
        --no-cache         : Bypass semantic query cache
        --no-rerank        : Skip cross-encoder re-ranking (faster but less accurate)
        --show-chunks      : Display retrieved document chunks
        --show-scores      : Show re-ranking scores for top results
        --show-stats       : Display detailed token usage statistics
        --stream           : Stream the response as it's generated
        --cleanup-cache    : Remove expired cache entries
        
    Workflow:
        1. Initialize ChromaDB collections (documents & cache)
        2. Optionally: Extract, chunk, and embed PDF (--perform-embedding)
        3. Accept user query
        4. Check semantic cache for similar queries
        5. If cache miss: Retrieve candidates from vector DB
        6. Re-rank results using cross-encoder
        7. Generate answer using OpenAI GPT with retrieved context
        8. Display answer and optional statistics
    """
    
    # Initialize ChromaDB with persistent storage
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    
    # Document collection for policy chunks
    chroma_db_collection = chroma_client.get_or_create_collection(
        name="insurance_policy_collection",
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )
    
    # Query cache collection for semantic caching
    query_cache = QueryCache(chroma_client)

    # =========================================================================
    # OPTIONAL: Document indexing (run once or when documents change)
    # =========================================================================
    if "--perform-embedding" in os.sys.argv:
        PDF_FILE_NAME = "Principal-Sample-Life-Insurance-Policy.pdf"
        PDF_DOCUMENT_PATH = os.path.join(os.getcwd(), "pdfs", PDF_FILE_NAME)
        print(f"✓ Document: {PDF_DOCUMENT_PATH}")

        raw_pdf_text = extract_pdf_text(PDF_DOCUMENT_PATH)
        cleaned_text = clean_text(raw_pdf_text)
        print(f"✓ Cleaned text length: {len(cleaned_text)} characters")

        text_chunks = chunk_semantic(cleaned_text, max_tokens=500)
        print(f"✓ Total chunks created: {len(text_chunks)}")

        ids = [str(uuid.uuid4()) for _ in text_chunks]
        chroma_db_collection.add(documents=text_chunks, ids=ids)
        print("✓ Chunks stored in ChromaDB.")
    else:
        print("✗ Skipping embedding step. Use '--perform-embedding' to run.")

    # =========================================================================
    # OPTIONAL: Cache cleanup
    # =========================================================================
    if "--cleanup-cache" in os.sys.argv:
        query_cache.cleanup_expired()

    # =========================================================================
    # RAG Query Processing
    # =========================================================================
    query = input("Enter your query: ").strip()
    use_cache = "--no-cache" not in os.sys.argv
    use_rerank = "--no-rerank" not in os.sys.argv
    use_streaming = "--stream" in os.sys.argv
    
    # Generate query embedding for retrieval and cache lookup
    query_embedding = embedding_fn([query])[0]
    
    # Step 1: Check semantic cache
    cached_entry = query_cache.get(query_embedding) if use_cache else None
    
    if cached_entry:
        print(f"✓ Cache HIT! (similarity: {cached_entry['similarity']:.3f})")
        print(f"✓ Original cached query: '{cached_entry['original_query']}'")
        documents = cached_entry["cached_results"]
    else:
        print("✗ Cache MISS. Querying ChromaDB...")
        
        # Step 2: Retrieve candidates (more than needed for re-ranking)
        initial_k = 20 if use_rerank else 10
        
        results = chroma_db_collection.query(
            query_embeddings=[query_embedding],
            n_results=initial_k
        )
        documents = results["documents"][0]
        
        # Step 3: Apply cross-encoder re-ranking
        if use_rerank and documents:
            print(f"✓ Re-ranking top {initial_k} results using cross-encoder...")
            reranked = rerank_results(query, documents, top_k=10)
            documents = [doc for doc, score in reranked]
            
            # Optionally show re-ranking scores
            if "--show-scores" in os.sys.argv:
                print("\n--- Re-ranking Scores ---")
                for i, (doc, score) in enumerate(reranked[:5]):
                    print(f"{i+1}. Score: {score:.4f} | {doc[:80]}...")
                print("---\n")
        
        # Step 4: Cache results for future similar queries
        if use_cache:
            query_cache.set(query, query_embedding, documents)
            print("✓ Results cached for future queries.")
    
    # =========================================================================
    # Display Retrieved Chunks (Optional)
    # =========================================================================
    if "--show-chunks" in os.sys.argv:
        print("\n" + "="*80)
        print("RETRIEVED CHUNKS")
        print("="*80)
        for i, chunk in enumerate(documents[:5], 1):  # Show top 5
            print(f"\n[Chunk {i}]")
            print(chunk)
            print("-" * 80)
    
    # =========================================================================
    # Generate RAG Response
    # =========================================================================
    print("\n" + "="*80)
    print("GENERATING ANSWER")
    print("="*80 + "\n")
    
    if use_streaming:
        # Streaming mode: Display response as it's generated
        print("Answer: ", end="", flush=True)
        for chunk in stream_rag_response(query, documents, model="gpt-4o-mini"):
            print(chunk, end="", flush=True)
        print("\n")
    else:
        # Non-streaming mode: Get complete response then display
        rag_result = generate_rag_response(query, documents, model="gpt-4o-mini")
        
        print(f"Answer:\n{rag_result['answer']}\n")
        
        # Optionally show detailed statistics
        if "--show-stats" in os.sys.argv:
            print("\n" + "="*80)
            print("STATISTICS")
            print("="*80)
            print(f"Model: {rag_result['model']}")
            print(f"Total Tokens: {rag_result['tokens_used']}")
            print(f"Prompt Tokens: {rag_result['prompt_tokens']}")
            print(f"Completion Tokens: {rag_result['completion_tokens']}")
            print(f"Finish Reason: {rag_result['finish_reason']}")
            print(f"Retrieved Chunks: {len(documents)}")

    # =========================================================================
    # Display Available Options
    # =========================================================================
    print("\n" + "="*80)
    print("Options:")
    print("  --perform-embedding  : Index the PDF document")
    print("  --no-cache          : Bypass query cache")
    print("  --no-rerank         : Skip cross-encoder re-ranking")
    print("  --show-chunks       : Display retrieved chunks")
    print("  --show-scores       : Show re-ranking scores")
    print("  --show-stats        : Display token usage statistics")
    print("  --stream            : Stream the response")
    print("  --cleanup-cache     : Remove expired cache entries")
    print("="*80)