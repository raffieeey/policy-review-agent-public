# Policy Review Agent

An agentic AI system that reviews, compares, and improves policy documents using hybrid retrieval (BM25 + dense vectors + RRF fusion) and LLM-powered analysis.

## Features

- **Document Ingestion** — Upload current and historical policies (PDF, DOCX, MD, TXT), parsed via Docling
- **Contextual Chunking** — RecursiveCharacterTextSplitter with section-aware prefixes
- **Hybrid Retrieval** — BM25Okapi (sparse) + Qdrant (dense) + Reciprocal Rank Fusion
- **LLM-Powered Pipeline** — Compare → Position → Identify Issues → Rewrite → Rate → Grammar Review
- **Weighted Scoring Rubric** — Structure (20%), Clarity (25%), Consistency (20%), Alignment (25%), Language (10%)
- **Streamlit UI** — Upload, compare, rewrite, rate, and download improved drafts

## Architecture

```
Upload → Docling Parser → Contextual Chunker → Hybrid Retriever
                                              ↓
                                    LLM Pipeline (linear)
                                    ├── Compare policies
                                    ├── Build positioning
                                    ├── Identify issues (with evidence)
                                    ├── Rewrite draft
                                    ├── Rate with rubric
                                    └── Grammar review
                                              ↓
                                    FinalPolicyPackage output
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set required environment variables
cp .env.example .env
# Edit .env with your ANTHROPIC_API_KEY and Qdrant settings

# Start Qdrant (Docker)
docker run -d -p 6333:6333 qdrant/qdrant

# Run the app
streamlit run app.py
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | ✅ | — | Anthropic API key for Claude |
| `QDRANT_URL` | ❌ | `http://localhost:6333` | Qdrant instance URL |
| `QDRANT_API_KEY` | ❌ | — | Qdrant API key (for cloud) |
| `COLLECTION_NAME` | ❌ | `policy_documents` | Qdrant collection prefix |
| `EMBEDDING_MODEL` | ❌ | `all-MiniLM-L6-v2` | Sentence transformer model |
| `LLM_MODEL` | ❌ | `claude-sonnet-4-20250514` | Anthropic model name |

## Project Structure

```
src/
├── config/settings.py          # Pydantic settings with .env support
├── schemas/
│   ├── documents.py             # PolicyMetadata, PolicyChunk, RetrievalResult
│   └── outputs.py               # FinalPolicyPackage, PolicyIssue, RatingScorecard
├── ingestion/
│   ├── parser.py                # Docling document parser
│   └── chunker.py               # Contextual chunking (prefix-only)
├── retrieval/
│   └── hybrid_retriever.py      # BM25 + Dense + RRF fusion
├── agents/
│   └── pipeline.py              # Linear review pipeline
└── rating/
    └── rubric.py                 # Weighted scoring rubric
```

## How It Works

1. **Upload** — Upload your current policy and historical reference documents
2. **Index** — Documents are parsed (Docling), chunked with context prefixes, and embedded into Qdrant
3. **Analyze** — The LLM pipeline runs: compare against historical evidence, recommend positioning, identify issues
4. **Rewrite** — LLM generates an improved draft addressing all identified issues
5. **Rate** — Weighted rubric scores the improved draft across 5 dimensions
6. **Review** — Final grammar and language copyedit
7. **Download** — Export the final polished policy as Markdown

## License

MIT License — see [LICENSE](LICENSE) for details.
