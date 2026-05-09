# Hybrid Chunking

Relevant source files

## Purpose and Scope

This page documents the HybridChunker class, which implements a sophisticated multi-stage chunking strategy that combines document structure awareness with token-based constraints. The hybrid chunker is designed for RAG (Retrieval-Augmented Generation) and LLM applications where chunks must respect both semantic boundaries and strict token limits.

For basic structure-aware chunking without token limits, see Hierarchical Chunking. For tokenization implementation details, see Tokenization. For the chunking framework and base classes, see Chunking Overview.

Sources:
docling_core/transforms/chunker/hybrid_chunker.py1-60

## Architecture Overview

The HybridChunker implements a four-stage refinement pipeline that progressively splits document content to satisfy token constraints while preserving semantic coherence:

### Multi-Stage Refinement Pipeline

The hybrid chunker refines chunks through successive stages, each applying a different splitting strategy when token limits are exceeded. This approach preserves document structure where possible while guaranteeing compliance with token constraints.

Sources:
docling_core/transforms/chunker/hybrid_chunker.py50-329
docling_core/transforms/chunker/hybrid_chunker.py301-329

## Configuration Parameters

### Core Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| tokenizer | BaseTokenizer | HuggingFaceTokenizer (all-MiniLM-L6-v2) | Tokenizer for counting tokens and determining chunk boundaries |
| max_tokens | int | Derived from tokenizer | Maximum tokens per chunk (read-only property) |
| merge_peers | bool | True | Whether to merge undersized chunks with matching metadata |
| always_emit_headings | bool | False | Whether to emit heading-only chunks for empty sections |
| serializer_provider | BaseSerializerProvider | ChunkingSerializerProvider() | Provider for document serialization |
| repeat_table_header | bool | True | Whether to repeat table headers in chunked tables |
| omit_header_on_overflow | bool | False | Omit table headers if they would cause a chunk overflow |

Sources:
docling_core/transforms/chunker/hybrid_chunker.py50-68

### Tokenizer Property

The max_tokens property is computed from the tokenizer configuration rather than being directly settable:

```python
@property
def max_tokens(self) -> int:
    """Get maximum number of tokens allowed."""
    return self.tokenizer.get_max_tokens()
```

This ensures consistency between the tokenizer's capabilities and the chunking behavior.

Sources:
docling_core/transforms/chunker/hybrid_chunker.py102-105

### Legacy Parameter Handling

The chunker includes backward-compatibility logic for deprecated initialization patterns via a @model_validator:

Sources:
docling_core/transforms/chunker/hybrid_chunker.py69-100
test/test_hybrid_chunker.py80-98
test/test_hybrid_chunker.py157-175

## Multi-Stage Processing Pipeline

### Stage 1: Hierarchical Chunking

The first stage delegates to HierarchicalChunker to create structure-aware initial chunks. This internal chunker is initialized with the same serializer_provider and always_emit_headings settings as the parent HybridChunker.

Sources:
docling_core/transforms/chunker/hybrid_chunker.py107-113
docling_core/transforms/chunker/hybrid_chunker.py301-320

### Stage 2: Item-Based Splitting

The second stage splits chunks that exceed token limits by attempting to fit DocItem objects into smaller windows. The _split_by_doc_items method iterates through the document items within a chunk, attempting to create new DocChunk instances that fit within the max_tokens limit.

If a single item (like a very large paragraph) exceeds the token limit on its own, it is passed to Stage 3 for text-based splitting.

Sources:
docling_core/transforms/chunker/hybrid_chunker.py172-221
docling_core/transforms/chunker/hybrid_chunker.py143-170

### Stage 3: Text-Based Splitting

For chunks containing a single item that still exceeds the token limit, the chunker employs semchunk for semantic text splitting. This stage calculates the "overhead" (tokens used by headings, captions, and metadata) and uses the remaining token budget to split the item's text.

```python
available_length = self.max_tokens - lengths.other_len
sem_chunker = semchunk.chunkerify(
    self.tokenizer.get_tokenizer(), chunk_size=available_length
)
text = doc_chunk.text
segments = sem_chunker.chunk(text)
```

Sources:
docling_core/transforms/chunker/hybrid_chunker.py223-250
docling_core/transforms/chunker/hybrid_chunker.py130-141

### Stage 4: Peer Merging (Optional)

If merge_peers is enabled, the final stage merges adjacent chunks that share the same heading context and fit together within the token limit. This reduces fragmentation in RAG systems.

Sources:
docling_core/transforms/chunker/hybrid_chunker.py252-299
test/test_hybrid_chunker.py32-54

## Token Counting Implementation

### Token Counting Methods

| Method | Purpose |
|---|---|
| _count_text_tokens | Counts tokens in raw text or a list of text strings. |
| _count_chunk_tokens | Serializes the chunk with context and counts the resulting tokens. |
| _doc_chunk_length | Returns a _ChunkLengthInfo object containing total, text, and metadata lengths. |

Sources:
docling_core/transforms/chunker/hybrid_chunker.py115-141

### Line-Based Token Chunking

The LineBasedTokenChunker is a specialized token-aware chunker used internally (or independently) that preserves line boundaries. It is particularly useful for structured text like code or tables where line breaks carry semantic meaning.

#### Key Features

- Prefix Support: Can prepend a repeated prefix (e.g., table headers) to every chunk.
- Overflow Handling: If a line exceeds the limit, it can be split, or the prefix can be omitted via omit_prefix_on_overflow.
- Large Prefix Handling: If the prefix itself is larger than max_tokens, it is split into standalone chunks at the beginning.

Sources:
docling_core/transforms/chunker/line_chunker.py20-69
docling_core/transforms/chunker/line_chunker.py71-113
test/test_line_chunker.py44-71

## Tokenizer Integration

### HuggingFace Integration

The HuggingFaceTokenizer wraps transformers tokenizers. It attempts to resolve max_tokens automatically from sentence_bert_config.json if not provided.

Sources:
docling_core/transforms/chunker/tokenizer/huggingface.py19-52

### OpenAI Integration

The OpenAITokenizer uses tiktoken for token counting and is typically used when chunking for OpenAI models like GPT-4.

Sources:
test/test_hybrid_chunker.py16-18
test/test_hybrid_chunker.py316-338

## Usage Patterns

### Standard Hybrid Chunking

```python
chunker = HybridChunker(
    tokenizer=HuggingFaceTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2"),
    merge_peers=True
)
chunks = list(chunker.chunk(dl_doc))
```

### Line-Based Chunking

```python
line_chunker = LineBasedTokenChunker(
    tokenizer=my_tokenizer,
    prefix="Table Header Context: ",
    omit_prefix_on_overflow=True
)
```

Sources:
test/test_hybrid_chunker.py63-77
test/test_line_chunker.py25-42

---

# Document Chunking

Relevant source files

## Purpose and Scope

This page describes Docling's document chunking system, which splits DoclingDocument objects into smaller, semantically meaningful pieces suitable for downstream tasks like vector embeddings, retrieval-augmented generation (RAG), and indexing. The system is designed to preserve document hierarchy while adhering to strict token budgets required by LLMs and embedding models.

## Overview

Document chunking addresses the challenge of processing long documents that exceed the context windows of AI models. Docling's native chunking system operates directly on the DoclingDocument representation rather than raw text or Markdown, allowing it to:

- Respect document structure (sections, lists, tables)
  docs/concepts/chunking.md109-114
- Maintain semantic coherence by keeping related elements together.
- Provide tokenization-aware refinements (splitting oversized chunks and merging undersized peers)
  docs/concepts/chunking.md61-72
- Attach rich metadata, including hierarchical headings and captions
  docs/concepts/chunking.md109-114

### Document Chunking Data Flow

Sources:
docs/concepts/chunking.md5-14
docs/concepts/chunking.md61-72
docs/concepts/chunking.md109-114

## Chunker Hierarchy

Docling defines a class hierarchy for chunkers to enable both flexibility and out-of-the-box utility. The base interface is BaseChunker, which defines the contract for all implementations
docs/concepts/chunking.md21-22

### BaseChunker

The BaseChunker defines two primary methods:

- chunk(self, dl_doc: DoclingDocument, **kwargs) -> Iterator[BaseChunk]: Returning the chunks for the provided document
  docs/concepts/chunking.md32-33
- contextualize(self, chunk: BaseChunk) -> str: Returning the potentially metadata-enriched serialization of the chunk, typically used to feed an embedding model
  docs/concepts/chunking.md34-36

### HierarchicalChunker

The HierarchicalChunker uses the structural information in a DoclingDocument to create chunks for individual document elements.

- Default Behavior: Creates one chunk per element
  docs/concepts/chunking.md109-111
- Merging: By default, it merges list items together unless merge_list_items is set to False
  docs/concepts/chunking.md111-112
- Metadata: Automatically attaches headers and captions to the chunk metadata
  docs/concepts/chunking.md112-114

### HybridChunker

The HybridChunker is the recommended implementation for RAG applications. It applies tokenization-aware refinements on top of the hierarchical output
docs/concepts/chunking.md61-62

- Splitting Pass: Splits chunks only when they exceed the token limit of the user-provided tokenizer
  docs/concepts/chunking.md68-69
- Merging Pass: Merges successive undersized chunks that share the same headings and captions
  docs/concepts/chunking.md70-71 This can be disabled via the merge_peers parameter (default True)
  docs/concepts/chunking.md71-72

### Line-Based Token Chunker

The LineBasedTokenChunker is designed for structured content like tables, code, and logs where line boundaries are semantically important
docs/examples/line_based_chunking.ipynb9-14

- Line Preservation: It attempts to keep entire lines within a single chunk, only splitting a line if it exceeds the maximum token limit on its own
  docs/concepts/chunking.md99
- Prefix Support: Supports adding a repeated prefix (e.g., table headers) to each chunk to maintain context
  docs/concepts/chunking.md103-104
- Overflow Handling: The omit_prefix_on_overflow parameter (default False) allows omitting the prefix if a line fits alone but would overflow with the prefix
  docs/concepts/chunking.md105-106

Sources:
docs/concepts/chunking.md28-36
docs/concepts/chunking.md61-106
docs/examples/line_based_chunking.ipynb9-14

## Tokenization and Configuration

In a RAG context, it is critical that the chunker uses a tokenizer aligned with the embedding model
docs/examples/hybrid_chunking.ipynb21-23

### Tokenizer Integration

Docling supports various tokenizer backends via docling-core extras:

- HuggingFace: pip install 'docling-core[chunking]'
  docs/concepts/chunking.md46-49
- OpenAI (tiktoken): pip install 'docling-core[chunking-openai]'
  docs/concepts/chunking.md51-54

```python
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer
from docling.chunking import HybridChunker

# Align tokenizer with embedding model
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = HuggingFaceTokenizer(
    tokenizer=AutoTokenizer.from_pretrained(EMBED_MODEL_ID),
)
chunker = HybridChunker(tokenizer=tokenizer)
```

Sources:
docs/concepts/chunking.md40-59
docs/examples/advanced_chunking_and_serialization.ipynb65-75

### Contextualization Strategy

The contextualize() method transforms a raw chunk into a string that includes its document context, such as parent headings. This enriched text is typically what is passed to embedding models
docs/examples/hybrid_chunking.ipynb123-126

### Chunk Entity Relationship

| Feature | Description |
|---|---|
| Heading Inclusion | Prepends the hierarchy of headings (e.g., "IBM\n1910s–1950s\n...") to the chunk text docs/examples/hybrid_chunking.ipynb151-152 |
| Caption Inclusion | Ensures captions for tables or figures are preserved within the chunk context docs/concepts/chunking.md112-114 |
| Table Header Repetition | HybridChunker can repeat table headers at the beginning of each chunk when a table spans multiple chunks (repeat_table_header=True) docs/concepts/chunking.md77-78 |
| Overflow Control | omit_header_on_overflow (default: False) provides flexibility for wide tables where rows might not fit with the header included docs/concepts/chunking.md80-84 |

Sources:
docs/examples/hybrid_chunking.ipynb123-152
docs/concepts/chunking.md34-36
docs/concepts/chunking.md74-85

## Advanced Serialization and Enrichment

Docling allows customizing how complex elements are serialized into text before or during chunking.

### Table Serialization

Users can customize serialization strategies. For example, wide tables can be handled with omit_header_on_overflow=True, which omits the header if a row would otherwise exceed the token limit
docs/concepts/chunking.md80-84

### Picture Enrichment

When a document has been enriched with picture descriptions (e.g., via DocumentPictureClassifier), these descriptions are included in the DoclingDocument and can be surfaced during chunking
docs/examples/enrich_doclingdocument.py38-41 This ensures that visual content is represented in the text chunks for retrieval
docs/examples/advanced_chunking_and_serialization.ipynb35-36

Sources:
docs/concepts/chunking.md80-84
docs/examples/enrich_doclingdocument.py138-147
docs/examples/advanced_chunking_and_serialization.ipynb35-36

## Framework Integrations

Docling's BaseChunker interface serves as the integration point for third-party AI frameworks.

- LlamaIndex: Integration is done using the BaseChunker interface, allowing users to plug in any Docling chunker implementation
  docs/concepts/chunking.md24-26
- LangChain: Users can either export the document to Markdown and use LangChain's native splitters or use the Docling native chunkers for structured retrieval
  docs/concepts/chunking.md13-14

Sources:
docs/concepts/chunking.md24-26
docs/concepts/chunking.md13-14
