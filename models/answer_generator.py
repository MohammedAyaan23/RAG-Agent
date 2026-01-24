import os
import logging
from typing import List, AsyncGenerator

logger = logging.getLogger(__name__)
from groq import AsyncGroq
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


class AnswerGenerator:
    """
    Generates a streamed RAG answer using retrieved context.
    """

    def __init__(self, model: AsyncGroq | None = None):
        self.model = model or AsyncGroq(
            api_key=os.getenv("GROQ_API_KEY")
        )

    async def generate(
        self,
        query: str,
        context_chunks: List[str],
    ) -> AsyncGenerator[str, None]:
        """
        Stream the answer token-by-token (or chunk-by-chunk).
        """
        logger.info("Starting answer generation...")

        context_text = "\n---\n".join(context_chunks)

        prompt = f"""
### ROLE:
You are answering as the author of the project.

### CONTEXT:
{context_text}

### INSTRUCTIONS:
1. Answer the query: "{query}"
2. Use FIRST PERSON ("I", "my") where appropriate.
3. ONLY use information present in the CONTEXT.
4. If the answer is not present in the CONTEXT, say:
   "I haven't documented that specific detail yet, but based on my project logs I can explain related components."
5. Do NOT add external knowledge.
6. Be concise but technically accurate.

### RESPONSE:
"""

        response = await self.model.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            stream=True,
        )

        async for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
