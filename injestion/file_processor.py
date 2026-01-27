import re
import numpy as np
from typing import List, Dict, Tuple, Optional
import chromadb
from chromadb.config import Settings
# from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import nltk
from chromadb.utils import embedding_functions
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())



# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab') # Add this line
except:
    nltk.download('punkt')
    nltk.download('punkt_tab') # Add this line

class EnhancedDynamicContentAwareChunker:
    """
    chroma_path: Path to ChromaDB storage
    collection_name: Name of ChromaDB collection    
    """
    
    def __init__(self, 
                 chroma_path: str = "./chroma_db",
                 collection_name: str = "document_chunks_personal_project"):
        """
        Initialize chunker with embedding model and ChromaDB connection
        
        Args:
            embedding_model_name: Sentence transformer model for semantic chunking
            chroma_path: Path to ChromaDB storage
            collection_name: Name of ChromaDB collection
        """
        print("[LOG] __init__: Initializing EnhancedDynamicContentAwareChunker...")
        self.embedding_model = embedding_functions.DefaultEmbeddingFunction()
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(
            path=chroma_path,
            settings=Settings(anonymized_telemetry=False)
        )

        try:
            self.chroma_client.delete_collection(collection_name)
            print(f"[LOG] Collection {collection_name} deleted.")
        except Exception:
            print(f"[LOG] Collection {collection_name} did not exist.")

        # Then create it again
        print("[LOG] Creating collection...")
        self.collection = self.chroma_client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print("[LOG] Collection created successfully.")
        
        # Try to get or create collection
        # try:
        #     self.collection = self.chroma_client.get_collection(
        #         name=collection_name, 
        #         embedding_function=self.embedding_model
        #     )
        # except:
        #     self.collection = self.chroma_client.create_collection(
        #         name=collection_name,
        #         embedding_function=self.embedding_model,
        #         metadata={"hnsw:space": "cosine"}  # Optimize for semantic search
        #     )
        


        
        # Enhanced strategies with ChromaDB metadata
        self.strategies = {
            'narrative': self._semantic_chunking,
            'technical': self._hierarchical_chunking,
            'tabular': self._table_aware_chunking,
            'code': self._syntax_aware_chunking,
            'mixed': self._adaptive_hybrid_chunking,
            'qa': self._qa_optimized_chunking  # New: For Q&A documents
        }
        
        # Content type heuristics with confidence scores
        self.content_detectors = [
            (self._is_code, 'code', 0.95),
            (self._is_tabular, 'tabular', 0.7),
            (self._is_technical, 'technical', 0.8),
            (self._is_narrative, 'narrative', 0.85),
            (self._is_qa, 'qa', 0.75)
        ]
        
        # Store chunking metrics
        self.metrics = {
            'total_chunks_created': 0,
            'chunks_by_type': {},
            'avg_chunk_size': {},
            'chunking_time': {}
        }
    
    def detect_content_type(self, text: str, use_embedding: bool = False) -> Tuple[str, float]:
        """
        Enhanced content detection with multiple heuristics and embedding fallback
        
        Returns:
            Tuple of (content_type, confidence_score)
        """
        print(f"[LOG] detect_content_type: Detecting content type (use_embedding={use_embedding})...")
        # Quick text stats
        lines = text.strip().split('\n')
        word_count = len(text.split())
        char_count = len(text)
        
        # Try heuristic detectors first
        for detector, content_type, base_confidence in self.content_detectors:
            if detector(text, lines, word_count):
                # Adjust confidence based on text length
                length_confidence = min(1.0, word_count / 1000)
                confidence = base_confidence * (0.3 + 0.7 * length_confidence)
                return content_type, confidence
        
        # Fallback: Use embedding-based similarity if enabled
        if use_embedding and word_count > 100:
            return self._embedding_based_detection(text)
        
        # Default to mixed with low confidence
        return 'mixed', 0.5
    
    def _is_code(self, text: str, lines: List[str], word_count: int) -> bool:
        """Detect code content"""
        print("[LOG] _is_code: Checking if content is code...")
        code_patterns = [
            r'(def\s+\w+\s*\(|class\s+\w+|import\s+|from\s+\w+\s+import)',
            r'(function\s+\w+|const\s+\w+|let\s+\w+|var\s+\w+)',
            r'(\{\s*|\}\s*|;\s*$|//|/\*|\*/)',
            r'(if\s*\(|for\s*\(|while\s*\(|return\s+)'
        ]
        
        code_lines = 0
        for line in lines[:50]:  # Check first 50 lines
            line_stripped = line.strip()
            if any(re.match(pattern, line_stripped) for pattern in code_patterns):
                code_lines += 1
        
        return code_lines >= 2 or code_lines / min(len(lines), 50) > 0.1
    
    def _is_tabular(self, text: str, lines: List[str], word_count: int) -> bool:
        table_indicators = 0
        for line in lines[:30]:
            line_stripped = line.strip()
            # Markdown tables must have pipes
            if line_stripped.count('|') >= 2:
                table_indicators += 1
            # CSV check: Only trigger if the line is relatively short 
            # (Real CSV rows rarely have 50+ words in a single 'cell')
            elif line_stripped.count(',') >= 3 and len(line_stripped.split()) < 40:
                table_indicators += 1
                
        return table_indicators >= 3 # Increase threshold to 3 lines
    
    def _is_technical(self, text: str, lines: List[str], word_count: int) -> bool:
        """Detect technical/scientific content"""
        print("[LOG] _is_technical: Checking if content is technical...")
        technical_terms = {
            'therefore', 'hence', 'thus', 'consequently',
            'algorithm', 'theorem', 'proof', 'equation',
            'hypothesis', 'experiment', 'results', 'analysis'
        }
        
        text_lower = text.lower()
        term_count = sum(1 for term in technical_terms if term in text_lower)
        
        # Check for LaTeX/math patterns
        math_patterns = [r'\$\$.*?\$\$', r'\$.*?\$', r'\\begin\{equation\}']
        math_count = sum(len(re.findall(pattern, text)) for pattern in math_patterns)
        
        return term_count >= 3 or math_count >= 1
    
    def _is_narrative(self, text: str, lines: List[str], word_count: int) -> bool:
        """Detect narrative/prose content"""
        print("[LOG] _is_narrative: Checking if content is narrative...")
        if word_count < 200:
            return False
            
        # Check sentence structure
        sentences = sent_tokenize(text[:1000])  # First 1000 chars
        if len(sentences) < 3:
            return False
            
        # Average sentence length
        avg_sentence_len = sum(len(s.split()) for s in sentences) / len(sentences)
        
        # Check for narrative markers
        narrative_markers = {'said', 'asked', 'replied', 'thought', 'felt'}
        marker_count = sum(1 for marker in narrative_markers if marker in text.lower())
        
        return (10 <= avg_sentence_len <= 25) or marker_count >= 2
    
    def _is_qa(self, text: str, lines: List[str], word_count: int) -> bool:
        """Detect Q&A format content"""
        print("[LOG] _is_qa: Checking if content is Q&A format...")
        q_patterns = [r'^Q[:.] ', r'^Question[:.] ', r'^\d+[\.\)]\s']
        a_patterns = [r'^A[:.]', r'^Answer[:.]', r'^Solution[:.]']
        
        q_count = sum(1 for line in lines if any(re.match(p, line.strip()) for p in q_patterns))
        a_count = sum(1 for line in lines if any(re.match(p, line.strip()) for p in a_patterns))
        
        return q_count >= 2 and a_count >= 2
    
    def _embedding_based_detection(self, text: str) -> Tuple[str, float]:
        """Use embeddings to detect content type"""
        print("[LOG] _embedding_based_detection: Using embeddings for content detection...")
        # Reference embeddings for each content type
        reference_texts = {
            'narrative': "Once upon a time, in a land far away, there lived a king.",
            'technical': "The algorithm operates with O(n log n) time complexity.",
            'code': "def calculate_sum(a, b):\n    return a + b",
            'tabular': "| Name | Age | City |\n|------|-----|------|\n| John | 25  | NY   |",
            'qa': "Q: What is photosynthesis?\nA: The process plants use to convert light into energy."
        }
        
        # Get embeddings
        text_embedding = np.array(self.embedding_model([text])[0], dtype=np.float32)
        similarities = {}
        
        for content_type, ref_text in reference_texts.items():
            ref_embedding = np.array(self.embedding_model([ref_text])[0], dtype=np.float32)
            similarity = np.dot(text_embedding, ref_embedding) / (
                np.linalg.norm(text_embedding) * np.linalg.norm(ref_embedding)
            )
            similarities[content_type] = similarity
        
        # Return type with highest similarity
        best_type = max(similarities, key=similarities.get)
        confidence = similarities[best_type]
        
        return best_type, confidence
    
    def _semantic_chunking(self, text: str, target_tokens: int = 110, 
                          overlap: int = 30) -> List[Dict]:
        """
        Semantic chunking with sentence boundaries and overlap
        
        Returns:
            List of chunk dictionaries with metadata
        """
        print(f"[LOG] _semantic_chunking: Starting semantic chunking (target_tokens={target_tokens}, overlap={overlap})...")
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for i, sentence in enumerate(sentences):
            sentence_tokens = len(sentence.split())
            
            if current_tokens + sentence_tokens > target_tokens and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'metadata': {
                        'chunk_type': 'semantic',
                        'sentences': len(current_chunk),
                        'tokens': current_tokens,
                        'has_overlap': False
                    }
                })
                
                # Start new chunk with overlap
                overlap_sentences = current_chunk[-min(len(current_chunk), overlap//20):]
                current_chunk = overlap_sentences + [sentence]
                current_tokens = sum(len(s.split()) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'metadata': {
                    'chunk_type': 'semantic',
                    'sentences': len(current_chunk),
                    'tokens': current_tokens,
                    'has_overlap': len(chunks) > 0
                }
            })
        
        return chunks
    
    def _hierarchical_chunking(self, text: str) -> List[Dict]:
        """
        Hierarchical chunking for technical documents
        Preserves section hierarchy for better context
        """
        print("[LOG] _hierarchical_chunking: Starting hierarchical chunking...")
        # Split by major sections (h1, h2, h3)
        sections = re.split(r'(\n#{1,3}\s+.+?\n)', text)
        
        chunks = []
        current_section = ""
        
        for i in range(0, len(sections), 2):
            if i + 1 < len(sections):
                section_header = sections[i + 1].strip()
                section_content = sections[i] if i > 0 else ""
                
                # Combine with next content section
                if i + 2 < len(sections):
                    section_content += sections[i + 2]
                
                full_section = section_header + "\n" + section_content
                
                # If section is too large, split further
                if len(full_section.split()) > 1000:
                    # Split by subsections or paragraphs
                    sub_chunks = self._split_large_section(full_section)
                    for sub_chunk in sub_chunks:
                        chunks.append({
                            'text': sub_chunk,
                            'metadata': {
                                'chunk_type': 'hierarchical',
                                'level': section_header.count('#'),
                                'parent_section': section_header.strip('#').strip()
                            }
                        })
                else:
                    chunks.append({
                        'text': full_section,
                        'metadata': {
                            'chunk_type': 'hierarchical',
                            'level': section_header.count('#'),
                            'tokens': len(full_section.split())
                        }
                    })
        
        return chunks
    
    def _table_aware_chunking(self, text: str) -> List[Dict]:
        """
        Preserve table structure while chunking
        Tables are kept as single chunks when possible
        """
        print("[LOG] _table_aware_chunking: Starting table-aware chunking...")
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        current_type = None  # 'table', 'text', or 'mixed'
        
        for line in lines:
            line_stripped = line.strip()
            
            # Detect table lines
            is_table_line = ('|' in line and '--' in line) or \
                           (line.count('|') >= 2 and len(line.split('|')) >= 3)
            
            if is_table_line:
                # If we were in text mode, save current chunk
                if current_type == 'text' and current_chunk:
                    chunks.append(self._create_chunk_dict('\n'.join(current_chunk), 'text'))
                    current_chunk = []
                
                current_type = 'table'
                current_chunk.append(line)
                
            else:
                # If we were in table mode, save the table
                if current_type == 'table' and current_chunk:
                    # Check if table is complete
                    if line_stripped == '' or not any('|' in l for l in current_chunk[-3:]):
                        chunks.append(self._create_chunk_dict('\n'.join(current_chunk), 'table'))
                        current_chunk = []
                
                current_type = 'text'
                if line_stripped:  # Skip empty lines for text
                    current_chunk.append(line)
                
                # If text chunk is getting large, save it
                if len(' '.join(current_chunk).split()) > 400:
                    chunks.append(self._create_chunk_dict('\n'.join(current_chunk), 'text'))
                    current_chunk = []
        
        # Save final chunk
        if current_chunk:
            chunks.append(self._create_chunk_dict('\n'.join(current_chunk), 
                                                current_type or 'mixed'))
        
        return chunks
    
    def _syntax_aware_chunking(self, text: str) -> List[Dict]:
        """
        Code-aware chunking that preserves logical blocks
        """
        print("[LOG] _syntax_aware_chunking: Starting syntax-aware chunking...")
        # Try to detect language
        language = self._detect_programming_language(text)
        
        chunks = []
        
        if language == 'python':
            # Split by functions and classes
            pattern = r'(?:^|\n)((?:class|def)\s+\w+.*?)(?=\n\s*(?:class|def)|\Z)'
            matches = re.findall(pattern, text, re.DOTALL | re.MULTILINE)
            
            for match in matches:
                if len(match.split()) > 800:
                    # Large function/class - split by methods or logical blocks
                    sub_chunks = self._split_python_code(match)
                    chunks.extend(sub_chunks)
                else:
                    chunks.append({
                        'text': match,
                        'metadata': {
                            'chunk_type': 'code',
                            'language': 'python',
                            'block_type': 'function' if 'def ' in match else 'class'
                        }
                    })
        
        elif language == 'javascript':
            # Similar pattern for JS
            pattern = r'(?:^|\n)((?:function|class|const|let|var)\s+\w+.*?)(?=\n\s*(?:function|class|const|let|var)|\Z)'
            matches = re.findall(pattern, text, re.DOTALL | re.MULTILINE)
            
            for match in matches:
                chunks.append({
                    'text': match,
                    'metadata': {
                        'chunk_type': 'code',
                        'language': 'javascript'
                    }
                })
        
        else:
            # Generic code chunking by logical blocks
            chunks = self._generic_code_chunking(text, language)
        
        return chunks
    
    def _adaptive_hybrid_chunking(self, text: str) -> List[Dict]:
        """
        Adaptive hybrid approach for mixed content
        Uses multiple strategies based on local content
        """
        print("[LOG] _adaptive_hybrid_chunking: Starting adaptive hybrid chunking...")
        # First, try to identify sections with different content types
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_segment = []
        current_type = None
        
        for para in paragraphs:
            para_type, _ = self.detect_content_type(para, use_embedding=False)
            
            if current_type and para_type != current_type:
                # Type change - process current segment
                if current_segment:
                    segment_text = '\n\n'.join(current_segment)
                    if current_type in self.strategies:
                        typed_chunks = self.strategies[current_type](segment_text)
                        chunks.extend(typed_chunks)
                    else:
                        # Fallback to semantic chunking
                        chunks.extend(self._semantic_chunking(segment_text))
                
                current_segment = [para]
                current_type = para_type
            else:
                if not current_type:
                    current_type = para_type
                current_segment.append(para)
        
        # Process final segment
        if current_segment:
            segment_text = '\n\n'.join(current_segment)
            if current_type in self.strategies:
                typed_chunks = self.strategies[current_type](segment_text)
                chunks.extend(typed_chunks)
            else:
                chunks.extend(self._semantic_chunking(segment_text))
        
        return chunks
    
    def _qa_optimized_chunking(self, text: str) -> List[Dict]:
        """
        Chunking optimized for Q&A pairs
        Each Q&A pair stays together
        """
        print("[LOG] _qa_optimized_chunking: Starting Q&A optimized chunking...")
        lines = text.split('\n')
        chunks = []
        current_qa = []
        in_question = False
        
        q_patterns = [r'^Q[:.]', r'^Question[:.]', r'^\d+[\.\)]\s']
        a_patterns = [r'^A[:.]', r'^Answer[:.]', r'^Solution[:.]']
        
        for line in lines:
            line_stripped = line.strip()
            
            # Check if this is a question
            is_question = any(re.match(p, line_stripped) for p in q_patterns)
            is_answer = any(re.match(p, line_stripped) for p in a_patterns)
            
            if is_question:
                # Save previous Q&A if exists
                if current_qa and len(current_qa) > 1:
                    chunks.append(self._create_qa_chunk(current_qa))
                    current_qa = []
                
                in_question = True
                current_qa.append(line)
            
            elif is_answer:
                in_question = False
                current_qa.append(line)
            
            else:
                # Continuation of current Q or A
                if current_qa:
                    current_qa.append(line)
        
        # Save final Q&A
        if current_qa:
            chunks.append(self._create_qa_chunk(current_qa))
        
        return chunks
    
    def _create_chunk_dict(self, text: str, chunk_type: str) -> Dict:
        """Helper to create standardized chunk dictionary"""
        print(f"[LOG] _create_chunk_dict: Creating chunk dictionary (type={chunk_type})...")
        return {
            'text': text,
            'metadata': {
                'chunk_type': chunk_type,
                'tokens': len(text.split()),
                'characters': len(text)
            }
        }
    
    def _create_qa_chunk(self, lines: List[str]) -> Dict:
        """Create a Q&A chunk with metadata"""
        print("[LOG] _create_qa_chunk: Creating Q&A chunk...")
        chunk_text = '\n'.join(lines)
        
        # Extract question and answer
        question = next((l for l in lines if any(re.match(p, l.strip()) 
                       for p in [r'^Q[:.]', r'^Question[:.]'])), lines[0] if lines else "")
        
        return {
            'text': chunk_text,
            'metadata': {
                'chunk_type': 'qa',
                'has_question': 'Q:' in chunk_text or 'Question:' in chunk_text,
                'has_answer': 'A:' in chunk_text or 'Answer:' in chunk_text,
                'tokens': len(chunk_text.split())
            }
        }
    
    def chunk_document(self, 
                       document_id: str,
                       text: str, 
                       metadata: Optional[Dict] = None) -> List[Dict]:
        """
        Main chunking method with ChromaDB storage
        
        Args:
            document_id: Unique ID for the document
            text: Document text to chunk
            metadata: Additional document metadata
            
        Returns:
            List of chunk dictionaries ready for ChromaDB
        """
        print(f"[LOG] chunk_document: Chunking document '{document_id}'...")
        import time
        start_time = time.time()
        
        # Detect content type
        content_type, confidence = self.detect_content_type(text, use_embedding=True)
        print(f"[LOG] chunk_document: Detected content type '{content_type}' with confidence {confidence:.2f}")
        
        # Choose chunking strategy
        chunking_strategy = self.strategies.get(content_type, self._adaptive_hybrid_chunking)
        
        # Apply chunking
        chunks = chunking_strategy(text)
        
        # Add document-level metadata to each chunk
        final_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = chunk.get('metadata', {})
            
            # Enhanced metadata for ChromaDB
            enhanced_metadata = {
                **chunk_metadata,
                'document_id': document_id,
                'chunk_id': f"{document_id}_chunk_{i:04d}",
                'chunk_index': i,
                'total_chunks': len(chunks),
                'content_type': content_type,
                'confidence': confidence,
                'original_document_metadata': metadata or {}
            }
            
            final_chunks.append({
                'id': enhanced_metadata['chunk_id'],
                'text': chunk['text'],
                'metadata': enhanced_metadata
            })
        
        # Update metrics
        chunking_time = time.time() - start_time
        self._update_metrics(content_type, len(final_chunks), chunking_time)
        
        return final_chunks
    
    def store_in_chromadb(self, chunks: List[Dict], batch_size: int = 100):
        """
        Store chunks in ChromaDB with embeddings
        
        Args:
            chunks: List of chunk dictionaries
            batch_size: Batch size for adding to ChromaDB
        """
        print(f"[LOG] store_in_chromadb: Storing {len(chunks) if chunks else 0} chunks...")
        if not chunks:
            return
    
        texts = [chunk['text'] for chunk in chunks]
        ids = [chunk['id'] for chunk in chunks]
    
        # --- FIX START: Sanitize Metadatas ---
        cleaned_metadatas = []
        for chunk in chunks:
            raw_meta = chunk.get('metadata', {})
            clean_meta = {}
            for key, value in raw_meta.items():
                # ChromaDB only allows str, int, float, bool
                if isinstance(value, (str, int, float, bool)):
                    clean_meta[key] = value
                elif isinstance(value, dict):
                    # If it's a dict (like original_document_metadata), flatten it or stringify it
                    clean_meta[key] = str(value) 
                elif value is None:
                    continue
                else:
                    # This handles the 'set' issue by converting it to a string
                    clean_meta[key] = str(value)
            cleaned_metadatas.append(clean_meta)
        # --- FIX END ---

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            batch_metadatas = cleaned_metadatas[i:i+batch_size]
        
            try:
                self.collection.add(
                    documents=batch_texts,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
            except Exception as e:
                print("⚠️ Quota exceeded. Waiting 60 seconds to retry...")
                time.sleep(60)  # Wait for the window to reset
                # Retry the same batch
                self.collection.add(
                    documents=batch_texts,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
    
        print(f"✅ Stored {len(chunks)} chunks in ChromaDB")
    
    def _update_metrics(self, content_type: str, num_chunks: int, chunking_time: float):
        """Update chunking metrics"""
        print(f"[LOG] _update_metrics: Updating metrics (type={content_type}, chunks={num_chunks}, time={chunking_time:.4f}s)...")
        self.metrics['total_chunks_created'] += num_chunks
        self.metrics['chunks_by_type'][content_type] = \
            self.metrics['chunks_by_type'].get(content_type, 0) + num_chunks
        self.metrics['chunking_time'][content_type] = \
            self.metrics['chunking_time'].get(content_type, 0) + chunking_time
    
    def get_metrics(self) -> Dict:
        """Get chunking metrics"""
        print("[LOG] get_metrics: Retrieving chunking metrics...")
        # Calculate averages
        for content_type in self.metrics['chunks_by_type']:
            count = self.metrics['chunks_by_type'][content_type]
            if count > 0:
                # This would require tracking sizes - simplified for now
                self.metrics['avg_chunk_size'][content_type] = "N/A"
        
        return self.metrics
    
    def process_document_file(self, 
                             file_path: str, 
                             document_id: Optional[str] = None,
                             metadata: Optional[Dict] = None):
        """
        Process a document file and store in ChromaDB
        
        Args:
            file_path: Path to document file
            document_id: Optional document ID (defaults to filename)
            metadata: Optional document metadata
        """
        print(f"[LOG] process_document_file: Processing file '{file_path}'...")
        import os
        
        # Read file
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Generate document ID if not provided
        if not document_id:
            document_id = os.path.basename(file_path).split('.')[0]
        
        # Add file metadata
        file_metadata = {
            'file_path': file_path,
            'file_size': os.path.getsize(file_path),
            'file_name': os.path.basename(file_path),
            **(metadata or {})
        }
        
        # Chunk document
        print(f"📄 Processing document: {document_id}")
        chunks = self.chunk_document(document_id, text, file_metadata)
        
        # Store in ChromaDB
        self.store_in_chromadb(chunks)
        
        print(f"✅ Created {len(chunks)} chunks from {document_id}")
        return chunks

    # Helper methods for code chunking
    def _detect_programming_language(self, text: str) -> str:
        """Detect programming language from code"""
        print("[LOG] _detect_programming_language: Detecting programming language...")
        if 'def ' in text and 'import ' in text:
            return 'python'
        elif 'function ' in text or 'const ' in text or 'let ' in text:
            return 'javascript'
        elif '#include' in text or 'int main' in text:
            return 'c'
        elif 'public class' in text or 'void main' in text:
            return 'java'
        else:
            return 'unknown'
    
    def _split_python_code(self, code: str) -> List[Dict]:
        """Split large Python code blocks"""
        print("[LOG] _split_python_code: Splitting large Python code...")
        chunks = []
        
        # Try to split by methods within class
        if 'class ' in code:
            class_match = re.match(r'(class\s+\w+.*?:\s*\n)(.*)', code, re.DOTALL)
            if class_match:
                class_declaration = class_match.group(1)
                class_body = class_match.group(2)
                
                # Add class declaration as first chunk
                chunks.append({
                    'text': class_declaration.strip(),
                    'metadata': {'chunk_type': 'code', 'block_type': 'class_declaration'}
                })
                
                # Split methods
                method_pattern = r'(\s+def\s+\w+.*?:\s*\n)(.*?)(?=\n\s+def|\n\s*$|\Z)'
                method_matches = list(re.finditer(method_pattern, class_body, re.DOTALL))
                
                for match in method_matches:
                    method_text = match.group(0)
                    chunks.append({
                        'text': method_text,
                        'metadata': {'chunk_type': 'code', 'block_type': 'method'}
                    })
        else:
            # Just a function - split by logical sections (comments, blocks)
            lines = code.split('\n')
            current_section = []
            
            for line in lines:
                current_section.append(line)
                # Split at significant comments or blank lines after code
                if line.strip().startswith('# ') and len(current_section) > 10:
                    chunks.append({
                        'text': '\n'.join(current_section),
                        'metadata': {'chunk_type': 'code', 'block_type': 'section'}
                    })
                    current_section = []
            
            if current_section:
                chunks.append({
                    'text': '\n'.join(current_section),
                    'metadata': {'chunk_type': 'code', 'block_type': 'section'}
                })
        
        return chunks
    
    def _generic_code_chunking(self, text: str, language: str) -> List[Dict]:
        """Generic code chunking for unknown languages"""
        print(f"[LOG] _generic_code_chunking: Chunking {language} code...")
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        current_lines = 0
        
        for line in lines:
            current_chunk.append(line)
            current_lines += 1
            
            # Split at logical points
            if current_lines >= 50 and (line.strip() == '' or line.strip().endswith('}')):
                chunks.append({
                    'text': '\n'.join(current_chunk),
                    'metadata': {
                        'chunk_type': 'code',
                        'language': language,
                        'lines': current_lines
                    }
                })
                current_chunk = []
                current_lines = 0
        
        if current_chunk:
            chunks.append({
                'text': '\n'.join(current_chunk),
                'metadata': {
                    'chunk_type': 'code',
                    'language': language,
                    'lines': current_lines
                }
            })
        
        return chunks
    
    def _split_large_section(self, section: str) -> List[str]:
        """Split large sections into paragraphs or smaller units"""
        print("[LOG] _split_large_section: Splitting large section...")
        # Try to split by paragraphs first
        paragraphs = section.split('\n\n')
        
        if len(paragraphs) > 1:
            return paragraphs
        
        # If no paragraphs, split by sentences
        sentences = sent_tokenize(section)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sent_length = len(sentence.split())
            if current_length + sent_length > 300 and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sent_length
            else:
                current_chunk.append(sentence)
                current_length += sent_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks


# Usage Example
def process_file(path : str, metadata : str = None):
    print("Strating the chunking process")
    # Initialize chunker
    file_name = path.split("/")[-1]
    if metadata is None:
        metadata = file_name

    chunker = EnhancedDynamicContentAwareChunker(
        chroma_path="./chroma_db",
        collection_name="my_documents_personal_project"
    )

    print("Chunker initialized successfully.")
    
    # Example document
    file_path = path
    
    
    with open(file_path, "r") as f:
        sample_text_1 = f.read()

    

    if sample_text_1:
        print("Sample text-1 read successfully.")
    else:
        print("Failed to read sample text-1.")
    
    # Process document
    print("Processing document-1...")
    chunks = chunker.chunk_document(
        document_id=file_name,
        text=sample_text_1,
        metadata={metadata}
    )
    print("Document-1 processed successfully.")
    
    # Store in ChromaDB
    print("Storing document-1 in ChromaDB...")
    chunker.store_in_chromadb(chunks)
    print("Document-1 stored in ChromaDB successfully.")
    
    # Get metrics
    metrics = chunker.get_metrics()
    print(f"\n📊 Metrics of document-1: {metrics}")

    
    
   