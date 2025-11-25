import fitz
import re
import uuid
import tiktoken
import pandas as pd
import chromadb
from openai import OpenAI
from sklearn.metrics import ndcg_score
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

###############################################################################
# 1. PDF EXTRACTION + CLEANING
###############################################################################

def extract_pdf_text(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = []
    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        full_text.append(f"<page={page_num+1}>\n{text}")
    return "\n".join(full_text)

def clean_text(text):
    text = re.sub(r"[ ]+", " ", text)  # kill extra spaces
    text = re.sub(r"\n{2,}", "\n\n", text)  # normalize newlines
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)  # fix hyphenation
    return text.strip()

###############################################################################
# 2. CHUNKING STRATEGIES
###############################################################################

def count_tokens(text):
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


# Strategy A: Fixed-size tokens (baseline)
def chunk_fixed_size(text, max_tokens=350, overlap=80):
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)

    chunks = []
    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunk_tokens = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens)
        chunks.append(chunk_text)
        start += (max_tokens - overlap)
    return chunks


# Strategy B: Semantic chunking using structure
def chunk_semantic(text, max_tokens=450):
    enc = tiktoken.get_encoding("cl100k_base")

    # Split primarily by page markers first
    pages = re.split(r"(?=<page=\d+>)", text)
    blocks = []
    for page in pages:
        page = page.strip()
        if not page:
            continue
        # Further split page into paragraph blocks (double newline or line breaks)
        paras = re.split(r"\n\s*\n+", page)
        for p in paras:
            p = p.strip()
            if p:
                blocks.append(p)

    chunks = []
    current = ""
    current_toks = 0

    for block in blocks:
        block_toks = len(enc.encode(block))
        # If single block already exceeds max_tokens, hard-split it
        if block_toks > max_tokens:
            # Flush current first
            if current.strip():
                chunks.append(current.strip())
                current = ""
                current_toks = 0
            # Split this large block into token slices
            tokens = enc.encode(block)
            for i in range(0, len(tokens), max_tokens):
                slice_tokens = tokens[i:i + max_tokens]
                chunks.append(enc.decode(slice_tokens).strip())
            continue

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


# Strategy C: Hybrid semantic + sliding window
def chunk_hybrid(text, semantic_chunks, window_size=500, stride=200):
    enc = tiktoken.get_encoding("cl100k_base")

    hybrid_chunks = []
    for chunk in semantic_chunks:
        tokens = enc.encode(chunk)
        for i in range(0, len(tokens), stride):
            window = tokens[i:i + window_size]
            if not window:
                continue
            hybrid_chunks.append(enc.decode(window))

    return hybrid_chunks

###############################################################################
# 3. EMBEDDING + CHROMADB SETUP
###############################################################################

def embed_text(texts, model="text-embedding-3-large", batch_size=50, max_context_tokens=8000):
    enc = tiktoken.get_encoding("cl100k_base")
    valid_texts = []
    for t in texts:
        if not t or not t.strip():
            continue
        toks = enc.encode(t)
        if len(toks) > max_context_tokens:
            # Hard split overly large text before embedding
            for i in range(0, len(toks), max_context_tokens):
                part = enc.decode(toks[i:i + max_context_tokens]).strip()
                if part:
                    valid_texts.append(part)
        else:
            valid_texts.append(t.strip())

    if not valid_texts:
        raise ValueError("No valid text to embed")

    all_embeddings = []
    for i in range(0, len(valid_texts), batch_size):
        batch = valid_texts[i:i + batch_size]
        vectors = client.embeddings.create(model=model, input=batch)
        all_embeddings.extend([v.embedding for v in vectors.data])
    return all_embeddings

def create_chroma_collection(name):
    chroma_client = chromadb.PersistentClient(
        path=f"./chroma_{name}"
    )
    return chroma_client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"}
    )

###############################################################################
# 4. LOADING EVALUATION QUERY SET
###############################################################################

def load_ground_truth():
    """
    Define your evaluation set here.
    Each entry has:
    - query text
    - relevant keywords
    - (optional) expected section or page
    """
    return [
        {
            "query": "What is the date of issue?",
            "keywords": ["date", "policy", "force"]
        },
        {
            "query": "What is meant by full time student?",
            "keywords": ["dependent", "child", "school"]
        },
        {
            "query": "When the policy has been updated?",
            "keywords": ["effective"]
        },
        {
            "query": "What are the premium rates?",
            "keywords": ["premium", "rates", "cost"]
        },
        {
            "query": "What are the consequences of failure to pay premium?",
            "keywords": ["grace", "policyholder", "end"]
        },
        {
            "query": "How can one meet good health requirements?",
            "keywords": ["health", "requirements", "good", "days"]
        },
        {
            "query": "Is the dependent also insured?",
            "keywords": ["dependent", "insured", "coverage"]
        },
        {
            "query": "Is the coverage worldwide?",
            "keywords": ["outside", "travel", "United States"]
        }
    ]

###############################################################################
# 5. RETRIEVAL + METRICS
###############################################################################

def compute_hit_at_k(results, keywords, k):
    top_k = " ".join(r["text"] for r in results[:k]).lower()
    return int(any(kw.lower() in top_k for kw in keywords))

def compute_mrr(results, keywords):
    for i, r in enumerate(results):
        if any(kw.lower() in r["text"].lower() for kw in keywords):
            return 1 / (i + 1)
    return 0

def compute_ndcg(results, keywords):
    relevance = [1 if any(kw.lower() in r["text"].lower() for kw in keywords) else 0 for r in results]
    ideal = sorted(relevance, reverse=True)
    return ndcg_score([ideal], [relevance])

###############################################################################
# 6. MAIN EVALUATION PIPELINE
###############################################################################

def evaluate_strategy(name, chunks, queries):
    print(f"\n=== Evaluating: {name} ===")
    
    # Filter out empty chunks
    chunks = [c.strip() for c in chunks if c and c.strip()]
    print(f"Total chunks after filtering: {len(chunks)}")

    collection = create_chroma_collection(name)
    ids = [str(uuid.uuid4()) for _ in chunks]
    embeddings = embed_text(chunks)

    collection.add(documents=chunks, ids=ids, embeddings=embeddings)

    eval_rows = []

    for q in queries:
        query_embedding = embed_text([q["query"]])[0]

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=8,
            include=["documents"]
        )

        retrieved = [
            {"text": doc}
            for doc in results["documents"][0]
        ]

        hit1  = compute_hit_at_k(retrieved, q["keywords"], 1)
        hit3  = compute_hit_at_k(retrieved, q["keywords"], 3)
        hit5  = compute_hit_at_k(retrieved, q["keywords"], 5)
        mrr   = compute_mrr(retrieved, q["keywords"])
        ndcg  = compute_ndcg(retrieved, q["keywords"])

        eval_rows.append({
            "strategy": name,
            "query": q["query"],
            "hit@1": hit1,
            "hit@3": hit3,
            "hit@5": hit5,
            "MRR": mrr,
            "nDCG": ndcg
        })

    return pd.DataFrame(eval_rows)


###############################################################################
# 7. EXECUTION
###############################################################################

if __name__ == "__main__":
    pdf_path = ".\\data\\Principal-Sample-Life-Insurance-Policy.pdf"

    print("Extracting PDF…")
    raw = extract_pdf_text(pdf_path)
    cleaned = clean_text(raw)

    print("Chunking strategies…")
    chunks_A = chunk_fixed_size(cleaned)
    chunks_B = chunk_semantic(cleaned)
    chunks_C = chunk_hybrid(cleaned, chunk_semantic(cleaned))

    queries = load_ground_truth()

    df_A = evaluate_strategy("fixed_size", chunks_A, queries)
    df_B = evaluate_strategy("semantic", chunks_B, queries)
    df_C = evaluate_strategy("hybrid", chunks_C, queries)

    full_results = pd.concat([df_A, df_B, df_C], ignore_index=True)
    full_results.to_csv("embedding_evaluation_results.csv", index=False)

    print("\n=== Evaluation Complete ===")
    print(full_results)
