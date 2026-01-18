"""RAG (Retrieval-Augmented Generation) engine for email search."""

import logging
from typing import Optional

from .embeddings import VectorStore
from .llm_client import LLMClient
from .models import SearchQuery, SearchResult

logger = logging.getLogger(__name__)

RAG_SYSTEM_PROMPT = """You are an intelligent email search assistant. Your job is to help users find information in their emails based on natural language queries.

You will be given:
1. A user's search query
2. Relevant email excerpts retrieved from their mailbox

Your task is to:
1. Analyze the retrieved emails to find the most relevant information
2. Provide a clear, concise answer to the user's query
3. Cite specific emails when providing information (mention sender, subject, date if available)
4. If the retrieved emails don't contain the answer, say so clearly

Be helpful, accurate, and concise. Focus on answering the user's actual question."""

SEARCH_REFINEMENT_PROMPT = """Based on the user's natural language query, extract key search terms and filters.

User query: {query}

Respond with a JSON object containing:
- keywords: list of important search terms
- date_range: "recent" (last month), "this_year", "all", or null
- attachment_filter: true if looking for attachments, false otherwise
- person_filter: email address or name if looking for emails from/to someone, null otherwise

Example response:
{{"keywords": ["invoice", "payment"], "date_range": "recent", "attachment_filter": true, "person_filter": null}}

Respond ONLY with the JSON object, no other text."""


class RAGEngine:
    """RAG engine combining vector search with LLM generation."""

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        llm_client: Optional[LLMClient] = None,
    ):
        """Initialize RAG engine."""
        self.vector_store = vector_store or VectorStore()
        self.llm_client = llm_client or LLMClient()

    async def search(
        self,
        query: str,
        limit: int = 10,
        use_llm: bool = True,
    ) -> dict:
        """Search emails using natural language.

        Args:
            query: Natural language search query
            limit: Maximum results to return
            use_llm: Whether to use LLM for answer generation

        Returns:
            Dict with search results and optional LLM-generated answer
        """
        # Retrieve relevant chunks
        results = self.vector_store.search(query, limit=limit * 2)

        # Deduplicate by email ID and aggregate
        email_chunks: dict[str, list[dict]] = {}
        for result in results:
            email_id = result["metadata"]["email_id"]
            if email_id not in email_chunks:
                email_chunks[email_id] = []
            email_chunks[email_id].append(result)

        # Sort emails by best chunk score
        sorted_emails = sorted(
            email_chunks.items(),
            key=lambda x: max(c["score"] for c in x[1]),
            reverse=True,
        )[:limit]

        # Build search results
        search_results = []
        for email_id, chunks in sorted_emails:
            best_chunk = max(chunks, key=lambda c: c["score"])
            search_results.append(
                {
                    "email_id": email_id,
                    "score": best_chunk["score"],
                    "subject": best_chunk["metadata"].get("subject", ""),
                    "from_address": best_chunk["metadata"].get("from_address", ""),
                    "date": best_chunk["metadata"].get("date", ""),
                    "folder": best_chunk["metadata"].get("folder", ""),
                    "matched_text": best_chunk["document"],
                    "all_chunks": [c["document"] for c in chunks],
                }
            )

        response = {
            "query": query,
            "results": search_results,
            "total_found": len(search_results),
        }

        # Generate LLM answer if requested
        if use_llm and search_results:
            answer = await self._generate_answer(query, search_results)
            response["answer"] = answer

        return response

    async def _generate_answer(
        self,
        query: str,
        results: list[dict],
    ) -> str:
        """Generate a natural language answer using retrieved context."""
        # Build context from top results
        context_parts = []
        for i, result in enumerate(results[:5], 1):
            context = f"Email {i}:\n"
            if result["subject"]:
                context += f"Subject: {result['subject']}\n"
            if result["from_address"]:
                context += f"From: {result['from_address']}\n"
            if result["date"]:
                context += f"Date: {result['date']}\n"
            context += f"Content:\n{result['matched_text']}\n"
            context_parts.append(context)

        context = "\n---\n".join(context_parts)

        prompt = f"""User's question: {query}

Retrieved emails:
{context}

Based on the emails above, please answer the user's question. If the emails don't contain enough information to answer, say so."""

        try:
            answer = await self.llm_client.generate(
                prompt=prompt,
                system_prompt=RAG_SYSTEM_PROMPT,
                temperature=0.3,
            )
            return answer
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return f"Error generating answer: {e}"

    async def search_with_filters(
        self,
        query: SearchQuery,
    ) -> dict:
        """Search with structured filters.

        Args:
            query: SearchQuery with filters

        Returns:
            Search results dict
        """
        # Build ChromaDB where clause
        where = {}

        if query.from_address:
            where["from_address"] = {"$eq": query.from_address}

        if query.folder:
            where["folder"] = {"$eq": query.folder}

        if query.has_attachments is not None:
            where["has_attachments"] = {"$eq": query.has_attachments}

        if query.attachment_type:
            where["attachment_types"] = {"$contains": query.attachment_type.value}

        # Search with filters
        results = self.vector_store.search(
            query.query,
            limit=query.limit * 2,
            where=where if where else None,
        )

        # Apply date filters in post-processing (ChromaDB doesn't handle dates well)
        if query.from_date or query.to_date:
            filtered_results = []
            for result in results:
                date_str = result["metadata"].get("date", "")
                if date_str:
                    from datetime import datetime

                    try:
                        date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                        if query.from_date and date < query.from_date:
                            continue
                        if query.to_date and date > query.to_date:
                            continue
                    except ValueError:
                        pass
                filtered_results.append(result)
            results = filtered_results

        # Deduplicate and format
        email_chunks: dict[str, list[dict]] = {}
        for result in results:
            email_id = result["metadata"]["email_id"]
            if email_id not in email_chunks:
                email_chunks[email_id] = []
            email_chunks[email_id].append(result)

        sorted_emails = sorted(
            email_chunks.items(),
            key=lambda x: max(c["score"] for c in x[1]),
            reverse=True,
        )[: query.limit]

        search_results = []
        for email_id, chunks in sorted_emails:
            best_chunk = max(chunks, key=lambda c: c["score"])
            search_results.append(
                {
                    "email_id": email_id,
                    "score": best_chunk["score"],
                    "subject": best_chunk["metadata"].get("subject", ""),
                    "from_address": best_chunk["metadata"].get("from_address", ""),
                    "date": best_chunk["metadata"].get("date", ""),
                    "folder": best_chunk["metadata"].get("folder", ""),
                    "matched_text": best_chunk["document"],
                }
            )

        return {
            "query": query.query,
            "filters": {
                "from_date": query.from_date.isoformat() if query.from_date else None,
                "to_date": query.to_date.isoformat() if query.to_date else None,
                "from_address": query.from_address,
                "folder": query.folder,
                "has_attachments": query.has_attachments,
                "attachment_type": query.attachment_type.value if query.attachment_type else None,
            },
            "results": search_results,
            "total_found": len(search_results),
        }

    async def ask(self, question: str) -> str:
        """Simple Q&A interface - search and generate answer.

        Args:
            question: Natural language question

        Returns:
            Generated answer string
        """
        result = await self.search(question, limit=5, use_llm=True)
        return result.get("answer", "No answer could be generated.")
