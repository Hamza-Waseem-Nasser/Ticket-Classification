"""
Embedding Manager for Multiple Model Testing and Comparison
Handles various embedding models and provides evaluation framework
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
import pickle
from pathlib import Path
import yaml
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import time
import json
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class EmbeddingManager:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.results_dir = Path("results/embeddings")
        self.results_dir.mkdir(exist_ok=True)
        
        self.model_cache = {}
        self.embedding_cache = {}
        
        # Initialize OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            self.openai_client = OpenAI(api_key=api_key)
        else:
            self.openai_client = None
        
    def get_available_models(self) -> Dict[str, List[str]]:
        """Get all available embedding models from config"""
        return self.config['embedding_models']
    
    def load_model(self, model_name: str) -> Any:
        """Load embedding model (cached)"""
        if model_name in self.model_cache:
            return self.model_cache[model_name]
        
        print(f"Loading model: {model_name}")
        
        if "text-embedding" in model_name:  # OpenAI models
            # OpenAI models don't need to be "loaded" - just use API
            self.model_cache[model_name] = "openai"
            return "openai"
        else:  # Sentence Transformers
            model = SentenceTransformer(model_name)
            self.model_cache[model_name] = model
            return model
    
    def generate_embeddings(self, texts: List[str], model_name: str) -> np.ndarray:
        """Generate embeddings for texts using specified model"""
        cache_key = f"{model_name}_{hash(str(texts))}"
        
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        model = self.load_model(model_name)
        
        if model == "openai":
            embeddings = self._generate_openai_embeddings(texts, model_name)
        else:
            embeddings = model.encode(texts, show_progress_bar=True)
        
        self.embedding_cache[cache_key] = embeddings
        return embeddings
    
    def _generate_openai_embeddings(self, texts: List[str], model_name: str) -> np.ndarray:
        """Generate embeddings using OpenAI API"""
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized. Check OPENAI_API_KEY.")
            
        embeddings = []
        
        for text in texts:
            try:
                response = self.openai_client.embeddings.create(
                    model=model_name,
                    input=text
                )
                embeddings.append(response.data[0].embedding)
                time.sleep(0.1)  # Rate limiting
            except Exception as e:
                print(f"Error generating embedding for text: {e}")
                # Use zero vector as fallback
                embeddings.append([0.0] * 1536)  # ada-002 dimension
        
        return np.array(embeddings)
    
    def evaluate_model_performance(self, 
                                 train_texts: List[str], 
                                 train_labels: List[str],
                                 test_texts: List[str], 
                                 test_labels: List[str],
                                 model_name: str) -> Dict[str, Any]:
        """Evaluate embedding model performance for classification"""
        
        print(f"Evaluating model: {model_name}")
        
        # Generate embeddings
        start_time = time.time()
        train_embeddings = self.generate_embeddings(train_texts, model_name)
        test_embeddings = self.generate_embeddings(test_texts, model_name)
        embedding_time = time.time() - start_time
        
        # Evaluate using k-NN classification
        start_time = time.time()
        predictions = self._knn_classify(train_embeddings, train_labels, test_embeddings)
        classification_time = time.time() - start_time
        
        # Calculate metrics
        accuracy = accuracy_score(test_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_labels, predictions, average='weighted', zero_division=0
        )
        
        # Top-k accuracy
        top_k_accuracy = self._calculate_top_k_accuracy(
            train_embeddings, train_labels, test_embeddings, test_labels, k=3
        )
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'top_3_accuracy': top_k_accuracy,
            'embedding_time': embedding_time,
            'classification_time': classification_time,
            'total_time': embedding_time + classification_time,
            'embedding_dimension': train_embeddings.shape[1] if len(train_embeddings) > 0 else 0
        }
        
        # Save results
        self._save_model_results(results, train_embeddings, test_embeddings)
        
        return results
    
    def _knn_classify(self, train_embeddings: np.ndarray, train_labels: List[str], 
                     test_embeddings: np.ndarray, k: int = 3) -> List[str]:
        """Classify using k-NN with cosine similarity"""
        predictions = []
        
        for test_embedding in test_embeddings:
            # Calculate similarities
            similarities = cosine_similarity([test_embedding], train_embeddings)[0]
            
            # Get top-k neighbors
            top_k_indices = np.argsort(similarities)[-k:][::-1]
            top_k_labels = [train_labels[i] for i in top_k_indices]
            
            # Majority vote
            from collections import Counter
            prediction = Counter(top_k_labels).most_common(1)[0][0]
            predictions.append(prediction)
        
        return predictions
    
    def _calculate_top_k_accuracy(self, train_embeddings: np.ndarray, train_labels: List[str],
                                test_embeddings: np.ndarray, test_labels: List[str], k: int = 3) -> float:
        """Calculate top-k accuracy"""
        correct = 0
        
        for i, test_embedding in enumerate(test_embeddings):
            similarities = cosine_similarity([test_embedding], train_embeddings)[0]
            top_k_indices = np.argsort(similarities)[-k:][::-1]
            top_k_labels = [train_labels[j] for j in top_k_indices]
            
            if test_labels[i] in top_k_labels:
                correct += 1
        
        return correct / len(test_labels)
    
    def _save_model_results(self, results: Dict, train_embeddings: np.ndarray, test_embeddings: np.ndarray):
        """Save model evaluation results and embeddings"""
        model_name_safe = results['model_name'].replace('/', '_').replace('-', '_')
        
        # Save results
        results_file = self.results_dir / f"{model_name_safe}_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save embeddings
        embeddings_file = self.results_dir / f"{model_name_safe}_embeddings.pkl"
        with open(embeddings_file, 'wb') as f:
            pickle.dump({
                'train_embeddings': train_embeddings,
                'test_embeddings': test_embeddings,
                'model_name': results['model_name']
            }, f)
    
    def compare_all_models(self, train_texts: List[str], train_labels: List[str],
                          test_texts: List[str], test_labels: List[str]) -> pd.DataFrame:
        """Compare performance of all available models"""
        all_results = []
        
        for category, models in self.get_available_models().items():
            print(f"\n=== Testing {category} models ===")
            for model_name in models:
                try:
                    results = self.evaluate_model_performance(
                        train_texts, train_labels, test_texts, test_labels, model_name
                    )
                    results['category'] = category
                    all_results.append(results)
                except Exception as e:
                    print(f"Error evaluating {model_name}: {e}")
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(all_results)
        comparison_file = Path("results/model_comparisons/embedding_comparison.csv")
        comparison_df.to_csv(comparison_file, index=False)
        
        return comparison_df
    
    def get_best_model(self, metric: str = 'f1_score') -> str:
        """Get the best performing model based on specified metric"""
        comparison_file = Path("results/model_comparisons/embedding_comparison.csv")
        
        if not comparison_file.exists():
            raise FileNotFoundError("No comparison results found. Run compare_all_models first.")
        
        df = pd.read_csv(comparison_file)
        best_model = df.loc[df[metric].idxmax(), 'model_name']
        
        return best_model
