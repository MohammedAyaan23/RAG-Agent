import chromadb
from chromadb.config import Settings
import asyncio
from groq import AsyncGroq
import os
from dotenv import load_dotenv, find_dotenv
import time
import json
import hashlib
from upstash_redis.asyncio import Redis
import re
import unicodedata


load_dotenv(find_dotenv())

# Load the model
brain = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

strategy_chunks = {
    "factual": 3,
    "explanation": 8,
    "comparison": 12,
    "creative": 6
}

def deduplicate_chunks(chunks):
    """
    Removes duplicate chunks based on their unique ID.
    This prevents the LLM from processing the same information multiple times.
    """
    print("Starting deduplication")
    seen_ids = set()
    unique_chunks = []
    for chunk in chunks:
        # Attempt to get ID from attribute (object) or key (dict)
        cid = chunk.get('id')
        print(cid)
        
        if cid and cid not in seen_ids:
            unique_chunks.append(chunk)
            seen_ids.add(cid)
    print("Deduplication completed")
    return unique_chunks


def normalize_prompt(text: str) -> str:
    # 1. Convert to lowercase
    text = text.lower()
    
    # 2. Normalize Unicode (e.g., converts 'é' to 'e')
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    
    # 3. Remove punctuation and special characters (keeping letters/numbers)
    text = re.sub(r'[^\w\s]', '', text)
    
    # 4. Remove extra whitespaces, tabs, and newlines
    text = " ".join(text.split())
    
    # 5. Strip leading/trailing spaces
    return text.strip()


class RedisSessionStore:
    def __init__(self, ttl=86400):
        self.redis = Redis(
            url = os.getenv("UPSTASH_REST_URL"),
            token = os.getenv("UPSTASH_REST_TOKEN")
        )
        self.ttl = ttl
    def _get_key(self, prompt):
        """
        Creates a unique key using haashlib
        """
        clean_prompt = normalize_prompt(prompt)
        return f"prompt_cache:{hashlib.md5(clean_prompt.encode()).hexdigest()}"
    async def get_key(self,prompt):

        try:
            key = self._get_key(prompt)
            return await self.redis.get(key)
        except Exception as e:
            print(f"Error getting key: {e}")
            return None
    async def set_key(self, prompt: str, response: str):
        try:
            key = self._get_key(prompt)
            await self.redis.set(key, response, ex=self.ttl)
        except Exception as e:
            print(f"⚠️ Cache write error: {e}")
        










# Configure ChromaDB persistent client ## will be using psql for session storage
chroma_client = chromadb.PersistentClient(
    path="./utils/chroma_db",  # Directory where ChromaDB will persist data
    settings=Settings(
        anonymized_telemetry=False,
        allow_reset=True
    )
)

# Create or get collection
collection_name = "my_documents_personal_project"
chroma_collection = chroma_client.get_or_create_collection(
    name=collection_name,
    metadata={"hnsw:space": "cosine"}
)




class QueryAnalyzer:
    def __init__(self, model = brain):
        self.model = model
        

    async def query_analyzer(self, query: str) -> dict:
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

        # base_chunks = {"factual": 3, "explanation": 8, "comparison": 12, "creative": 6}
        # base_val = base_chunks.get(data['intent'], 6)
        
        # chunks_needed = max(3, min(15, int(base_val * (0.8 + data['complexity'] * 0.5))))

        # return {
        #     "intent": data['intent'],
        #     "recommended_chunks": chunks_needed,
        #     "complexity": data['complexity']
        # }

class Retriever:
        def __init__(self, chroma_collection):
            self.collection = chroma_collection


        async def retrieve(self, query: str, no_of_chunks: int) -> list:

            results = self.collection.query(
                query_texts=[query],
                n_results=no_of_chunks,
                include=['documents', 'metadatas'] 
            )
    
            formatted_chunks = []
    
            # ✅ Access the IDs normally (Chroma includes them automatically)
            if results.get('documents') and results['documents']:
                ids = results['ids'][0]
                docs = results['documents'][0]
                metas = results['metadatas'][0]

            for i in range(len(ids)):
                formatted_chunks.append({
                    'id': ids[i],           
                    'text': docs[i],        
                    'metadata': metas[i]     
                })            
            return formatted_chunks
            
class RagMonitor:
    async def log_retrieval(self, query, no_of_chunks, context_chunks, latency, analysis, full_response):
        with open("metrics.txt", "a") as f:
            f.write("Logs of the query----------")
            f.write(f"Query: {query}\n")
            f.write(f"Chunks_retrieved: {no_of_chunks}\n")
            f.write(f"Context_chunks: {context_chunks}\n")
            f.write(f"factual score: {analysis['intent_scores']['factual']}, factual_query: {analysis['sub_queries']['factual_query']}\n")
            f.write(f"explanation score: {analysis['intent_scores']['explanation']}, explanation_query: {analysis['sub_queries']['explanation_query']}\n")
            f.write(f"comparison score: {analysis['intent_scores']['comparison']}, comparison_query: {analysis['sub_queries']['comparison_query']}\n")
            f.write(f"creative score: {analysis['intent_scores']['creative']}, creative_query: {analysis['sub_queries']['creative_query']}\n")
            f.write(f"Latency_ms: {latency}\n")
            f.write(f"Answer:{full_response}\n")
            f.write("\n")
    async def log_cache(self,query,response, latency):
        with open("metrics.txt", "a") as f:
            f.write("Logs of the cache-hit query----------")
            f.write(f"Query: {query}\n")
            f.write(f"Latency_ms: {latency}\n")
            f.write(f"Answer:{response}\n")
            f.write("\n")







async def asyn_main():
    analyzer = QueryAnalyzer()
    retriever = Retriever(chroma_collection)
    monitor = RagMonitor()
    upstash_session_store = RedisSessionStore()

    while True:
        user_query = input("\nAsk a question (or type 'exit'): ")
        if user_query.lower() == 'exit':
            break
        
        start_time = time.time()
        cache_response = await upstash_session_store.get_key(user_query)

        if cache_response:
            print("Cache HIT!\n")
            print("="*20)
            print("CACHE HIT ANSWER")
            print("="*20)
            print(f"\n{cache_response}")
            print("="*50)
            await monitor.log_cache(user_query, cache_response, (time.time() - start_time) * 1000)
            continue

        print("cache miss, looking into vector_db")

        # 1. Analyze Query
        print(f"🔍 Analyzing query intent...")
        analysis = await analyzer.query_analyzer(user_query)
        print("Intent analysis completed, starting retrieval")
        print(f"factual score: {analysis['intent_scores']['factual']}, factual_query: {analysis['sub_queries']['factual_query']}")
        print(f"explanation score: {analysis['intent_scores']['explanation']}, explanation_query: {analysis['sub_queries']['explanation_query']}")
        print(f"comparison score: {analysis['intent_scores']['comparison']}, comparison_query: {analysis['sub_queries']['comparison_query']}")
        print(f"creative score: {analysis['intent_scores']['creative']}, creative_query: {analysis['sub_queries']['creative_query']}")
        # retrieval_tasks = []
        total_chunks = []
        intents = ['factual', 'explanation', 'comparison', 'creative']

        for intent in intents:
            print(f"🔍 Processing {intent} intent...")
            score = analysis['intent_scores'].get(intent, 0)
            query = analysis['sub_queries'].get(f"{intent}_query")

            if score > 0.1 and query:
                # Calculate dynamic k based on score
                k = max(3, int(score * strategy_chunks.get(intent, 5)))
                
                # In sync dev mode, we await and extend immediately
                # retrieve() returns a list, so .extend() keeps the main list flat
                new_chunks = await retriever.retrieve(query, k)
                total_chunks.extend(new_chunks)

        # No need to flatten 'total_chunks' anymore because we used .extend()
        context_chunks = total_chunks

        if context_chunks:
            print(f"📦 Total chunks collected: {len(context_chunks)}")
    
        # 3. Apply Final Deduplication
        # This is critical because different sub-queries often find the same source text
            print("🧹 Deduplicating chunks...")
            context_chunks = deduplicate_chunks(context_chunks)
            print(f"✅ Final context size: {len(context_chunks)} unique chunks")
        else:
            print("⚠️ No relevant chunks found for the given intent scores.")
            context_chunks = []
        #  Retrieve Chunks
        # print(f"📥 Retrieving {analysis['recommended_chunks']} chunks for '{analysis['intent']}' intent...")
        # context_chunks = retriever.retrieve(user_query, analysis['recommended_chunks'])
        # print(f"Retrieved {len(context_chunks)} chunks for query: {user_query}, context_chunks: {context_chunks}")
        
        #  Generate Final Answer (RAG)
        context_text = "\n\n".join([c['text'] for c in context_chunks])


        rag_prompt = f"""
    ### ROLE:
You are Ayaan, an Associate Software Developer. You are answering a query based EXCLUSIVELY on your internal project documentation provided in the "CONTEXT" section below.

### CONTEXT:
{context_text}

### INSTRUCTIONS:
1. Answer the query: "{user_query}"
2. Use the FIRST PERSON ("I", "my") as if you are describing your own work.
3. STRICTNESS: You must only use information found in the CONTEXT. 
4. If the information is not there, say: "I haven't documented that specific detail in my project logs yet, but I can tell you about the [Mention a related tech from context] I used instead."
5. DO NOT provide general "typical" microservice architectures. Only describe YOUR architecture from the context.

### RESPONSE:
    """
        
        response = await brain.chat.completions.create(
            model="llama-3.3-70b-versatile", # Groq's flagship model
            messages=[
                {"role": "user", "content": rag_prompt}
            ],
            temperature=0,
            stream=True # Highly recommended for Groq's speed
        )
        latency = int((time.time() - start_time) * 1000)
        full_response = ""

        print("\n" + "="*20 + " AI RESPONSE " + "="*20)
        async for chunk in response:
            chunk_text = chunk.choices[0].delta.content
            if chunk_text:
                full_response += chunk_text
                print(chunk_text, end="", flush=True)
        print("\n" + "="*53)

        #  Log Metrics
        await monitor.log_retrieval(user_query, len(context_chunks), context_chunks, latency, analysis,full_response)
        print(f"📊 Logged metrics (Latency: {latency}ms)")
        try:
            await upstash_session_store.set_key(prompt= user_query, response=full_response)
            print("Cache set successfuly")
        except Exception as e:
            print(f"Failed to cache response: {e}")
        
        
        
        
       


if __name__ == "__main__":
    # Run the agent
    print("RAG Agent with ChromaDB Persistent Session")
    print("=" * 50)
    # Debug Peek
    # peek = chroma_collection.peek(limit=10)
    # print(f"🔍 Peek at DB Content: {peek['documents']}")
    try:
        asyncio.run(asyn_main())
    except KeyboardInterrupt:
        print("\nShutdown gracefully.")
