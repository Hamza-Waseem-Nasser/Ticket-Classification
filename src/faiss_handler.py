"""
FAISS Handler for Vector Indexing and Similarity Search
Optimized for multiple embedding models and fast retrieval
"""

import numpy as np
import faiss
import pickle
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import yaml

class FAISSHandler:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.indices_dir = Path("results/faiss_indices")
        self.indices_dir.mkdir(exist_ok=True)
        
        self.index = None
        self.metadata = None
        self.model_name = None
        
    def create_index(self, embeddings: np.ndarray, metadata: List[Dict], 
                    model_name: str, index_type: str = "IndexFlatIP") -> None:
        """Create FAISS index for embeddings"""
        
        dimension = embeddings.shape[1]
        
        # Create index based on type
        if index_type == "IndexFlatIP":
            # Inner Product for cosine similarity (normalize embeddings first)
            normalized_embeddings = self._normalize_embeddings(embeddings)
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(normalized_embeddings.astype('float32'))
        elif index_type == "IndexFlatL2":
            # L2 distance
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings.astype('float32'))
        elif index_type == "IndexIVFFlat":
            # IVF with flat quantizer (for larger datasets)
            nlist = min(100, embeddings.shape[0] // 4)  # Number of clusters
            quantizer = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            normalized_embeddings = self._normalize_embeddings(embeddings)
            self.index.train(normalized_embeddings.astype('float32'))
            self.index.add(normalized_embeddings.astype('float32'))
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        self.metadata = metadata
        self.model_name = model_name
        
        # Save index and metadata
        self._save_index()
        
        print(f"Created FAISS index with {self.index.ntotal} vectors of dimension {dimension}")
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings for cosine similarity"""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return embeddings / norms
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[List[float], List[Dict]]:
        """Search for similar vectors"""
        if self.index is None:
            raise ValueError("No index loaded. Create or load an index first.")
        
        # Normalize query embedding if using IP index
        if isinstance(self.index, (faiss.IndexFlatIP, faiss.IndexIVFFlat)):
            query_embedding = self._normalize_embeddings(query_embedding.reshape(1, -1))
        else:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Get metadata for results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:  # Valid index
                result = self.metadata[idx].copy()
                result['similarity_score'] = float(scores[0][i])
                results.append(result)
        
        return scores[0].tolist(), results
    
    def batch_search(self, query_embeddings: np.ndarray, k: int = 5) -> List[Tuple[List[float], List[Dict]]]:
        """Batch search for multiple queries"""
        results = []
        for query_embedding in query_embeddings:
            scores, metadata_results = self.search(query_embedding, k)
            results.append((scores, metadata_results))
        return results
    
    def evaluate_retrieval_performance(self, query_embeddings: np.ndarray, 
                                     true_labels: List[str], k: int = 5) -> Dict[str, float]:
        """Evaluate retrieval performance"""
        correct_at_k = [0] * k
        total_queries = len(query_embeddings)
        
        for i, query_embedding in enumerate(query_embeddings):
            scores, results = self.search(query_embedding, k)
            true_label = true_labels[i]
            
            # Check if true label appears in top-k results
            for j in range(min(k, len(results))):
                if results[j]['primary_label'] == true_label:
                    # Mark as correct for this k and all higher k values
                    for l in range(j, k):
                        correct_at_k[l] += 1
                    break
        
        # Calculate recall@k for each k
        recall_at_k = {f'recall@{i+1}': correct_at_k[i] / total_queries for i in range(k)}
        
        return recall_at_k
    
    def get_similarity_distribution(self, query_embeddings: np.ndarray) -> Dict[str, Any]:
        """Analyze similarity score distribution"""
        all_scores = []
        
        for query_embedding in query_embeddings:
            scores, _ = self.search(query_embedding, k=10)
            all_scores.extend(scores)
        
        all_scores = np.array(all_scores)
        
        return {
            'mean_similarity': float(np.mean(all_scores)),
            'std_similarity': float(np.std(all_scores)),
            'min_similarity': float(np.min(all_scores)),
            'max_similarity': float(np.max(all_scores)),
            'percentiles': {
                '25th': float(np.percentile(all_scores, 25)),
                '50th': float(np.percentile(all_scores, 50)),
                '75th': float(np.percentile(all_scores, 75)),
                '90th': float(np.percentile(all_scores, 90)),
                '95th': float(np.percentile(all_scores, 95))
            }
        }
    
    def _save_index(self):
        """Save FAISS index and metadata"""
        if self.index is None or self.model_name is None:
            return
        
        model_name_safe = self.model_name.replace('/', '_').replace('-', '_')
        
        # Save FAISS index
        index_file = self.indices_dir / f"{model_name_safe}.faiss"
        faiss.write_index(self.index, str(index_file))
        
        # Save metadata
        metadata_file = self.indices_dir / f"{model_name_safe}_metadata.pkl"
        with open(metadata_file, 'wb') as f:
            pickle.dump({
                'metadata': self.metadata,
                'model_name': self.model_name,
                'index_config': {
                    'type': type(self.index).__name__,
                    'dimension': self.index.d,
                    'total_vectors': self.index.ntotal
                }
            }, f)
        
        print(f"Saved FAISS index and metadata for {self.model_name}")
    
    def load_index(self, model_name: str):
        """Load existing FAISS index and metadata"""
        model_name_safe = model_name.replace('/', '_').replace('-', '_')
        
        index_file = self.indices_dir / f"{model_name_safe}.faiss"
        metadata_file = self.indices_dir / f"{model_name_safe}_metadata.pkl"
        
        if not index_file.exists() or not metadata_file.exists():
            raise FileNotFoundError(f"Index files not found for model: {model_name}")
        
        # Load FAISS index
        self.index = faiss.read_index(str(index_file))
        
        # Load metadata
        with open(metadata_file, 'rb') as f:
            data = pickle.load(f)
            self.metadata = data['metadata']
            self.model_name = data['model_name']
        
        print(f"Loaded FAISS index for {self.model_name} with {self.index.ntotal} vectors")
    
    def list_available_indices(self) -> List[str]:
        """List all available FAISS indices"""
        faiss_files = list(self.indices_dir.glob("*.faiss"))
        model_names = []
        
        for faiss_file in faiss_files:
            model_name = faiss_file.stem.replace('_', '/')
            model_names.append(model_name)
        
        return model_names
    
    def get_index_info(self) -> Dict[str, Any]:
        """Get information about current index"""
        if self.index is None:
            return {"error": "No index loaded"}
        
        return {
            'model_name': self.model_name,
            'index_type': type(self.index).__name__,
            'dimension': self.index.d,
            'total_vectors': self.index.ntotal,
            'is_trained': getattr(self.index, 'is_trained', True)
        }
