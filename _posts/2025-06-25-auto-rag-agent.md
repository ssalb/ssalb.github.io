---
layout: post
title: "From Docs to Answers: My Smolagents + Docling + DuckDB Experiment 🤖"
date: 2025-06-25 09:00:00
description: Testing smolagents, docling, and duckdb through an agent that RAGs for you.
tags: agents RAG llms transformers
# categories: transformers
thumbnail: assets/img/blog/2025-06-25-auto-rag/article-image.png
images:
  slider: true
featured: false
---

To the surprise of no one, **retrieval-augmented generation (RAG)** and **agents** have been the biggest buzz in the AI world for over a year now. 
Today, you can choose from loads of libraries, vector databases, models, and frameworks to build your own applications, and more keep popping up all the time!

Among the multitude of options, three particular new contenders caught my eye:

* Hugging Face’s [smolagents](https://github.com/huggingface/smolagents), especially their *code agent* (more on that later).
* [DuckDB](https://duckdb.org/), which—at the risk of underselling it—I will be using it as an *OLAP SQLite*. 
It’s recently added support for a vector data type and similarity search, making it very interesting for this kind of project.
* IBM’s [Docling](https://docling-project.github.io/docling/)—a simple, flexible, and powerful document processor/parser.

So there I was, curious about a lightweight, in-process vector DB, a promising document parser, and a new agents framework... 
what could I possibly do with them? 🤔 **An agent that ingests documents and *RAGs* for you** 💡

This article shares my first experience and impressions while building a *smol* prototype agent that ingests your documents, 
indexes them for retrieval, and answers questions about their content.

## The Stack

This project uses several well-known packages—including *transformers* and *sentence-transformers* for LLMs and embeddings, plus *Gradio* for the UI. 
I won’t detail all of them here, but I want to highlight our three main players.

#### Smolagents

Smolagents is a minimalist AI agent framework from Hugging Face. As far as I know (and I don’t follow every framework), they were the first 
to push the idea of a *code agent* in an open‑source setting. Instead of interacting with tools via JSON, the code agent writes Python code 
that executes tools inside a sandbox. That makes it both more flexible and more efficient, since it can tackle complex workflows in fewer steps.
For example, it could execute something like the following in a single step:

```python
result_1 = tool_1(args)
if result_1:
    result_2 = tool_2(result_1)
print(result_2)
```

Like most HF projects, it’s model‑agnostic—you can plug in your favorite LLM (open‑source or API‑based) and it integrates seamlessly with 
the *transformers* library.

At the time of writing, they’ve also added Vision–Language models (VLMs) and *Computer Use* support in the last few weeks, opening up even 
more possibilities 🚀

#### Docling

I don’t think Docling has received the hype it deserves. IBM open‑sourced it a while back, yet I haven’t seen many people talking about it. 
Out of the box, it takes tons of document formats and parses them into JSON or Markdown. What used to require multiple libraries and custom 
parsers is now a one‑stop shop. It’s so straightforward that I barely had to tweak anything 😅—which says a lot about its defaults.

You can also supercharge it with VLMs, though I found the out‑of‑the‑box pipeline already covers the “80%” of most RAG needs.

#### DuckDB

DuckDB delivered exactly what it promised: an in‑process OLAP database—think “SQLite for analytics”—and it recently added experimental support 
for fixed‑size arrays and vector similarity search. That means you can store embeddings directly in a column and run nearest‑neighbor queries 
with plain SQL.

With the `vss` extension, building an `HNSW` index and performing similarity search takes just a few lines. No servers, no extra services, just 
a local file and your queries. For this prototype, that meant everything stayed self‑contained: ingest, embed, store, and search. Super convenient.

## Building the RAG Agent

The core idea is simple:

* A chat UI built on Gradio
* A code agent following the [ReAct framework](https://www.ibm.com/think/topics/react-agent)
* Two initial tools:

  1. **Indexer**: ingests, parses, and indexes documents
  2. **Retriever**: embeds queries and performs similarity search

During a conversation, the user can ask the agent to index new documents—either by upload or URL—or to answer questions about any indexed content.

In practice (no surprise to anyone building agents), it didn’t work perfectly at first. For this reason I ended up adding a third tool:

3. **Summarizer**: condenses one or more text chunks, either generally or tailored to a query.

All tools are invoked via generated Python code. The same LLM powers both the agent’s reasoning and the summarization, keeping the architecture simple.

### Indexing

For each document, the indexer:

1. **Parses** the file with Docling
2. **Extracts named entities** via an NER‑tuned model
3. **Computes** an embedding vector
4. **Inserts** a row per chunk into DuckDB, storing:

   * Document name
   * Chunk text
   * Named entities
   * Embedding vector

### Retrieval

The retriever:

1. **Extracts named entities** from the query
2. **Embeds** the query
3. **Retrieves** chunks via similarity search (optionally filtered by document name)
4. **Reranks** based on shared named entities between chunk and query

This quick entity‑based reranking boosted relevance without requiring expensive cross‑encoders or reranker models. It’s not perfect, but it’s surprisingly effective.
An obvious but much more intricate extension to this approach would be to build a knowledge graph using these names entities.

## Observations & Takeaways

While I hinted at a few challenges above, the overall experience with these tools has been very positive. Here are my main takeaways:

<swiper-container keyboard="true" navigation="true" pagination="true" pagination-clickable="true" pagination-dynamic-bullets="true" rewind="true">
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/blog/2025-06-25-auto-rag/slider-1.jpg" class="img-fluid rounded z-depth-1" %}</swiper-slide>
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/blog/2025-06-25-auto-rag/slider-2.jpg" class="img-fluid rounded z-depth-1" %}</swiper-slide>
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/blog/2025-06-25-auto-rag/slider-3.jpg" class="img-fluid rounded z-depth-1" %}</swiper-slide>
  <swiper-slide>{% include figure.liquid loading="eager" path="assets/img/blog/2025-06-25-auto-rag/slider-4.jpg" class="img-fluid rounded z-depth-1" %}</swiper-slide>
</swiper-container>

#### Agent bias toward tool usage

By default, the code agent strongly prefers using tools—even inventing them—instead of solving tasks directly as an LLM. To work around this, 
you can tweak the system prompt or add deliberately goofy tools. I opted for the latter, introducing the **summarizer**. This played to the 
agent’s tool‑centric tendencies while leveraging the LLM’s strengths in summarization.

#### Model size matters

Earlier this year, I wrote about [test‑time compute](https://ssalb.github.io/blog/2025/test-time-compute-story-generation/) and how smaller 
LLMs can outperform expectations. In this case, though, model size really did matter. Models in the \~7B–11B range struggled with open‑ended 
tasks, needing explicit instructions to use specific tools. Swapping up to \~30B–70B turned that around: the larger models handled ambiguous 
requests and self‑corrected much better. The trade‑off was losing local inferencing and moving to cloud endpoints.

#### Final verdict

All in all, I came away with a positive impression of the three tools that motivated this project. Docling is incredibly simple yet powerful
—I’ll definitely reach for it again when processing documents. DuckDB is great, but I see its real potential more as a Delta Lake alternative 
than for simple, local storage (see their [Duck Lake post](https://duckdb.org/2025/05/27/ducklake.html)). And smolagents? I’m excited to take 
it beyond this PoC, especially if async support is added. It’s shaping up to be a solid production contender.
