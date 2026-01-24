import json
from typing import Dict
from groq import AsyncGroq
import os
from dotenv import load_dotenv, find_dotenv

from dotenv import load_dotenv, find_dotenv
import logging

load_dotenv(find_dotenv())
logger = logging.getLogger(__name__)


class PseudoDeterministicPlanner:
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

        

        

        return {
            "planner": "pseudo_deterministic",
            "query": query,
            "intent_scores": analysis["intent_scores"],
            "sub_queries": analysis["sub_queries"],
        }

    async def _analyze_query(self, query: str) -> Dict:
        """
        Uses LLM to extract intent and complexity.

        Args:
            query (str): The user query to analyze.

        Returns:
            Dict: A structured plan containing:
                - intent_scores: The detected intent scores
                - sub_queries: The detected sub queries
        """

        combined_prompt = """
        ### SYSTEM INSTRUCTION ###
        You are a Query Analyzer for a RAG system. Your goal is to decompose a user query into intent scores and specific search queries. 

        ### OUTPUT FORMAT ###
        Return ONLY a valid JSON object. Do not include preamble or markdown formatting like ```json.

        ### EXAMPLE ###
        User Query: "How does the Go backend's rate limiting compare to NGINX's approach?"
        Output: 
        {
            "intent_scores": {
                "factual": 0.1,
                "explanation": 0.2,
                "comparison": 0.6,
                "creative": 0.1
            },
            "sub_queries": {
                "factual_query": "Go backend rate limiting implementation details",
                "explanation_query": "How NGINX handles traffic limits at the edge",
                "comparison_query": "Difference between Token Bucket in Go and NGINX reverse proxy rate limiting",
                "creative_query": "Hypothetical scenario of moving rate limiting from application to infrastructure layer"
            }
        }

        ### USER QUERY ###
        Query: """
        
        # Single call to the AI
        response = await self.model.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": combined_prompt + query,
            }
        ],
        model="llama-3.3-70b-versatile" # Specify model here
    )
        
        # Simple parsing (using a try/except or json.loads)
        try:
            # Cleaning the response text in case AI adds markdown ```json blocks
            clean_text = response.choices[0].message.content.replace('```json', '').replace('```', '').strip()
            data = json.loads(clean_text)
        except:
            data = {
                "intent_scores": {"factual": 1.0, "explanation": 0.0, "comparison": 0.0, "creative": 0.0},
                "sub_queries": {
                    "factual_query": query,
                    "explanation_query": query,
                    "comparison_query": query,
                    "creative_query": query
                }
            }
        

        return data
