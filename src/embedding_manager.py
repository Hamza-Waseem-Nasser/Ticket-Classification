"""
Embedding Manager for Saber Ticket Classification
Handles multiple embedding models, FAISS indexing, and LLM enrichment
"""

import numpy as np
import pandas as pd
import faiss
import json
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union
import logging
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class EmbeddingManager:
    """Manages embedding generation, storage, and similarity search using FAISS with LLM enrichment"""
    
    def __init__(self, config: dict):
        self.config = config
        self.models = {}
        self.faiss_indices = {}
        self.category_data = None
        self.embeddings = {}
        self.results_dir = Path("results/experiments/phase2_embeddings")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize Gemini if API key is available
        self.gemini_model = None
        if os.getenv('GEMINI_API_KEY'):
            try:
                genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
                self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
                self.logger.info("Gemini model initialized for LLM enrichment")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Gemini: {e}")
                self.gemini_model = None
        
        # Initialize OpenAI if API key is available
        self.openai_client = None
        if os.getenv('OPENAI_API_KEY'):
            try:
                self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
                self.logger.info("OpenAI client initialized for embeddings")
            except Exception as e:
                self.logger.warning(f"Failed to initialize OpenAI: {e}")
                self.openai_client = None
        
    def enrich_with_llm(self, subcategory_prefix: str, subcategory_keywords: str, 
                       subcategory2_prefix: str, subcategory2_keywords: str) -> str:
        """Use LLM to generate enhanced Arabic descriptions for better embedding"""
        
        if not self.gemini_model:
            # Fallback: simple concatenation if LLM not available
            return f"{subcategory_prefix} {subcategory_keywords} {subcategory2_prefix} {subcategory2_keywords}"
        
        try:
            prompt = f"""
أنت خبير في تصنيف طلبات العملاء لمنصة سابر السعودية. 

المعلومات المتوفرة:
- وصف الفئة: {subcategory_prefix}
- كلمات مفتاحية للفئة: {subcategory_keywords}
- وصف الفئة الفرعية: {subcategory2_prefix}
- كلمات مفتاحية للفئة الفرعية: {subcategory2_keywords}

المطلوب: اكتب وصفاً شاملاً باللغة العربية (مع الاحتفاظ بالمصطلحات الإنجليزية كما هي) يوضح:
1. نوع المشكلة أو الطلب الذي يواجهه العميل
2. الأعراض أو المؤشرات التي قد يذكرها العميل
3. السياق والحالات المشابهة

اجعل الوصف طبيعياً وقريباً من طريقة تعبير العملاء العاديين. لا تذكر أسماء الفئات مباشرة، بل اوصف المشكلة.

مثال للأسلوب المطلوب: "العميل يواجه صعوبات في..." أو "يحدث هذا عندما..."

الوصف (100-150 كلمة):
"""
            
            response = self.gemini_model.generate_content(prompt)
            enhanced_description = response.text.strip()
            
            # Clean up the response
            enhanced_description = enhanced_description.replace('\n', ' ')
            enhanced_description = ' '.join(enhanced_description.split())
            
            self.logger.info(f"LLM enrichment successful for: {subcategory_prefix[:30]}...")
            return enhanced_description
            
        except Exception as e:
            self.logger.warning(f"LLM enrichment failed: {e}. Using fallback.")
            # Fallback to simple concatenation
            return f"{subcategory_prefix} {subcategory_keywords} {subcategory2_prefix} {subcategory2_keywords}"
    
    def load_sentence_transformer(self, model_name: str):
        """Load a Sentence Transformer model"""
        try:
            self.logger.info(f"Loading Sentence Transformer: {model_name}")
            model = SentenceTransformer(model_name)
            return model
        except Exception as e:
            self.logger.error(f"Failed to load {model_name}: {e}")
            return None
    
    def generate_sentence_transformer_embeddings(self, texts: List[str], model) -> np.ndarray:
        """Generate embeddings using Sentence Transformers"""
        try:
            embeddings = model.encode(texts, show_progress_bar=True)
            return embeddings
        except Exception as e:
            self.logger.error(f"Sentence Transformer embedding generation failed: {e}")
            return None
    
    def load_openai_model(self, model_name: str):
        """Load OpenAI embedding model"""
        if not self.openai_client:
            self.logger.error("OpenAI client not initialized")
            return None
        return model_name
    
    def generate_openai_embeddings(self, texts: List[str], model_name: str) -> np.ndarray:
        """Generate embeddings using OpenAI API"""
        try:
            embeddings = []
            batch_size = 100
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                response = self.openai_client.embeddings.create(input=batch, model=model_name)
                batch_embeddings = [data.embedding for data in response.data]
                embeddings.extend(batch_embeddings)
            
            return np.array(embeddings, dtype=np.float32)
        except Exception as e:
            self.logger.error(f"OpenAI embedding generation failed: {e}")
            return None
    
    def create_faiss_index(self, embeddings: np.ndarray, model_name: str) -> faiss.Index:
        """Create FAISS index for fast similarity search"""
        try:
            dimension = embeddings.shape[1]
            
            # Use IndexFlatIP for inner product (cosine similarity with normalized vectors)
            index = faiss.IndexFlatIP(dimension)
            
            # Make a copy and normalize embeddings for cosine similarity
            normalized_embeddings = embeddings.copy().astype('float32')
            faiss.normalize_L2(normalized_embeddings)
            
            # Add normalized embeddings to index
            index.add(normalized_embeddings)
            
            self.logger.info(f"FAISS index created for {model_name}: {index.ntotal} vectors, dim={dimension}")
            return index
            
        except Exception as e:
            self.logger.error(f"FAISS index creation failed: {e}")
            return None
    
    def save_embeddings_and_index(self, embeddings: np.ndarray, model_name: str, metadata: dict):
        """Save embeddings, FAISS index, and metadata"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Clean model name for filename
        clean_model_name = model_name.replace("/", "_").replace("-", "_")
        
        # Save embeddings
        embeddings_path = self.results_dir / f"embeddings_{clean_model_name}_{timestamp}.npy"
        np.save(embeddings_path, embeddings)
        
        # Save metadata
        metadata_path = self.results_dir / f"embeddings_{clean_model_name}_{timestamp}_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        # Create and save FAISS index
        index = self.create_faiss_index(embeddings, model_name)
        if index:
            faiss_dir = self.results_dir / "faiss_indices"
            faiss_dir.mkdir(exist_ok=True)
            index_path = faiss_dir / f"faiss_index_{clean_model_name}_{timestamp}.index"
            faiss.write_index(index, str(index_path))
            
            self.faiss_indices[model_name] = {
                'index': index,
                'path': index_path,
                'embeddings_path': embeddings_path,
                'metadata_path': metadata_path
            }
        
        # Save LLM-enhanced category data if available
        if hasattr(self, 'category_data') and self.category_data:
            enhanced_data_path = self.results_dir / f"enhanced_categories_{clean_model_name}_{timestamp}.json"
            with open(enhanced_data_path, 'w', encoding='utf-8') as f:
                json.dump(self.category_data, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Saved LLM-enhanced category data to {enhanced_data_path}")
        
        self.logger.info(f"Saved embeddings and index for {model_name}")
        return {
            'embeddings_path': embeddings_path,
            'metadata_path': metadata_path,
            'index_path': index_path if index else None,
            'enhanced_data_path': enhanced_data_path if hasattr(self, 'category_data') and self.category_data else None
        }
    
    def find_existing_enhanced_data(self, model_name: str = None) -> Optional[str]:
        """Find the most recent enhanced category data file (model-independent)"""
        try:
            # Look for any enhanced categories file (not model-specific since descriptions are reusable)
            pattern = f"enhanced_categories_*.json"
            
            # Search for existing files
            matching_files = list(self.results_dir.glob(pattern))
            
            if matching_files:
                # Get the most recent file
                latest_file = max(matching_files, key=lambda f: f.stat().st_mtime)
                self.logger.info(f"Found existing enhanced data: {latest_file.name}")
                self.logger.info("Reusing LLM-generated descriptions (model-independent)")
                return str(latest_file)
            
            return None
        except Exception as e:
            self.logger.warning(f"Error finding existing enhanced data: {e}")
            return None
    
    def load_enhanced_data(self, file_path: str) -> Optional[List[Dict]]:
        """Load enhanced category data from JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.logger.info(f"Loaded {len(data)} enhanced categories from {Path(file_path).name}")
            return data
        except Exception as e:
            self.logger.error(f"Failed to load enhanced data from {file_path}: {e}")
            return None
    
    def process_categories(self, categories_df: pd.DataFrame, model_name: str = None) -> List[str]:
        """Process category data and create LLM-enhanced text representations for embedding"""
        texts = []
        processed_data = []
        
        # Check for existing enhanced data first
        if model_name:
            existing_data_path = self.find_existing_enhanced_data()
            if existing_data_path:
                # Load existing enhanced data
                self.category_data = self.load_enhanced_data(existing_data_path)
                if self.category_data:
                    # Extract the enhanced texts for embedding
                    texts = [item['enhanced_text'] for item in self.category_data]
                    self.logger.info(f"Using {len(texts)} existing enhanced texts for embedding")
                    return texts
        
        self.logger.info("Starting LLM enrichment for categories...")
        
        for idx, row in categories_df.iterrows():
            # Extract the 4 descriptive fields (avoiding label leakage)
            # Note: Column names have trailing spaces
            subcategory_prefix = str(row.get('SubCategory_Prefix ', '')).strip()
            subcategory_keywords = str(row.get('SubCategory_Keywords', '')).strip()
            subcategory2_prefix = str(row.get('SubCategory2_Prefix ', '')).strip()
            subcategory2_keywords = str(row.get('SubCategory2_Keywords', '')).strip()
            
            # Generate LLM-enhanced description
            enhanced_text = self.enrich_with_llm(
                subcategory_prefix, subcategory_keywords,
                subcategory2_prefix, subcategory2_keywords
            )
            
            texts.append(enhanced_text)
            
            processed_data.append({
                'index': idx,
                'service': row['Service'],
                'subcategory': row['SubCategory'],
                'subcategory2': row['SubCategory2'],
                'keywords': row['SubCategory2_Keywords'],
                'original_fields': {
                    'subcategory_prefix': subcategory_prefix,
                    'subcategory_keywords': subcategory_keywords,
                    'subcategory2_prefix': subcategory2_prefix,
                    'subcategory2_keywords': subcategory2_keywords
                },
                'enhanced_text': enhanced_text
            })
            
            # Progress indicator
            if (idx + 1) % 10 == 0:
                self.logger.info(f"Processed {idx + 1}/{len(categories_df)} categories")
        
        self.logger.info(f"LLM enrichment completed for {len(categories_df)} categories")
        self.category_data = processed_data
        return texts
    
    def search_similar_categories(self, query_text: str, model_name: str, top_k: int = 3) -> List[Dict]:
        """Search for similar categories using FAISS index"""
        if model_name not in self.faiss_indices:
            self.logger.error(f"No FAISS index found for {model_name}")
            return []
        
        try:
            # Generate embedding for query
            if 'text-embedding' in model_name:
                # OpenAI embedding
                query_embedding = self.generate_openai_embeddings([query_text], model_name)
            else:
                # Sentence Transformer
                model = self.models.get(model_name)
                if not model:
                    model = self.load_sentence_transformer(model_name)
                    self.models[model_name] = model
                query_embedding = self.generate_sentence_transformer_embeddings([query_text], model)
            
            if query_embedding is None:
                return []
            
            # Normalize query embedding for cosine similarity
            query_normalized = query_embedding.copy().astype('float32')
            faiss.normalize_L2(query_normalized)
            
            # Search using FAISS (returns cosine similarities)
            index = self.faiss_indices[model_name]['index']
            similarities, indices = index.search(query_normalized, top_k)
            
            # Format results
            results = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx < len(self.category_data):
                    category = self.category_data[idx]
                    # Ensure similarity is a valid float and clamp between 0 and 1
                    raw_similarity = float(similarity)
                    clamped_similarity = max(0.0, min(1.0, raw_similarity))
                    
                    results.append({
                        'rank': i + 1,
                        'subcategory': category['subcategory'],
                        'subcategory2': category['subcategory2'],
                        'service': category['service'],
                        'score': clamped_similarity,
                        'confidence': clamped_similarity * 100,
                        'embedding_index': int(idx)
                    })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []
    
    def classify_tickets(self, tickets_df: pd.DataFrame, model_name: str, limit: int = None) -> Dict:
        """Classify a set of tickets using the specified model"""
        if limit:
            tickets_df = tickets_df.head(limit)
        
        results = {
            'model_name': model_name,
            'test_type': 'real_user_tickets',
            'total_tickets_tested': len(tickets_df),
            'results': []
        }
        
        total_confidence = 0
        
        for idx, row in tickets_df.iterrows():
            ticket_description = row['Cleaned_Description']
            
            # Get top 3 classifications
            classifications = self.search_similar_categories(ticket_description, model_name, top_k=3)
            
            if classifications:
                best_match = classifications[0]
                total_confidence += best_match['confidence']
                
                ticket_result = {
                    'ticket_id': idx,
                    'ticket_description': ticket_description,
                    'original_description': ticket_description,
                    'classifications': classifications,
                    'best_match': best_match
                }
                results['results'].append(ticket_result)
        
        # Add analysis
        if results['results']:
            avg_confidence = total_confidence / len(results['results'])
            results['analysis'] = {
                'total_processed': len(results['results']),
                'average_confidence': avg_confidence,
                'classification_format': 'SubCategory → SubCategory2'
            }
        
        return results