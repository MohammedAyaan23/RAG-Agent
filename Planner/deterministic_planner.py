import json
from typing import Dict
from groq import AsyncGroq
import os
from dotenv import load_dotenv, find_dotenv

from dotenv import load_dotenv, find_dotenv
import logging

load_dotenv(find_dotenv())
logger = logging.getLogger(__name__)


class DeterministicPlanner:
    """
    Deterministic planner for RAG.

    Responsibilities:
    - Analyze query intent
    - Estimate complexity
    - Decide number of chunks to retrieve
    - Return a structured plan
    """

    def __init__(self, model: AsyncGroq | None = None):
        self.model = model or AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

        # Base chunk sizes per intent
        self.base_chunks = {
            "factual": 3,
            "explanation": 8,
            "comparison": 12,
            "creative": 6,
        }

    async def run(self, query: str) -> Dict:
        """
        Analyze the query and return a deterministic retrieval plan.

        Args:
            query (str): The user query to analyze.

        Returns:
            Dict: A structured plan containing:
                - planner: The planner type
                - query: The original query
                - intent: The detected intent
                - complexity: The estimated complexity
                - recommended_chunks: The recommended number of chunks to retrieve
        """
        logger.info("Running run method")

        
        analysis = await self._analyze_query(query)
        logger.info(f"Planner analysis: {analysis}")

        intent = analysis["intent"]
        complexity = analysis["complexity"]

        base = self.base_chunks.get(intent, 6)

        # Deterministic chunk computation
        recommended_chunks = max(
            3,
            min(15, int(base * (0.8 + complexity * 0.5)))
        )

        return {
            "planner": "deterministic",
            "query": query,
            "intent": intent,
            "complexity": complexity,
            "recommended_chunks": recommended_chunks,
        }

    async def _analyze_query(self, query: str) -> Dict:
        """
        Uses LLM to extract intent and complexity.

        Args:
            query (str): The user query to analyze.

        Returns:
            Dict: A structured plan containing:
                - intent: The detected intent
                - complexity: The estimated complexity
        """

        prompt = f"""
Analyze the following user query and return ONLY a valid JSON object
with the following fields:

- intent: one of ["factual", "explanation", "comparison", "creative"]
- complexity: a float between 0 and 1

Query:
{query}
"""

        response = await self.model.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0,
        )

        raw = response.choices[0].message.content.strip()

        try:
            clean = raw.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean)

            intent = data.get("intent", "factual")
            complexity = float(data.get("complexity", 0.5))

            # Guardrails
            if intent not in self.base_chunks:
                intent = "factual"

            complexity = max(0.0, min(1.0, complexity))

            return {
                "intent": intent,
                "complexity": complexity,
            }

        except Exception:
            # Safe fallback
            return {
                "intent": "factual",
                "complexity": 0.5,
            }
