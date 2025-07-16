"""
Main script to run ticket classification with vector similarity search
This script processes 3 tickets using Saber Categories and cleaned tickets data
"""

import sys
import os
sys.path.append('src')
sys.path.append('utils')

import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path

from embedding_manager import EmbeddingManager
from data_processor import DataProcessor

def main():
    """Main function to run ticket classification"""
    
    # Configuration
    config = {
        'data': {
            'categories_file': 'Saber Categories-1.csv',
            'tickets_file': 'cleaned_tickets.csv'
        },
        'embedding': {
            'model': 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',  # Larger multilingual model
            'max_tickets': 3
        }
    }
    
    print("🚀 Starting Saber Ticket Classification")
    print("=" * 50)
    
    # Initialize processors
    data_processor = DataProcessor(config)
    embedding_manager = EmbeddingManager(config)
    
    # Load data
    print("📊 Loading data...")
    categories_df = data_processor.load_categories(config['data']['categories_file'])
    tickets_df = data_processor.load_tickets(config['data']['tickets_file'], 
                                           limit=config['embedding']['max_tickets'])
    
    if categories_df.empty or tickets_df.empty:
        print("❌ Failed to load data")
        return
    
    print(f"✅ Loaded {len(categories_df)} categories and {len(tickets_df)} tickets")
    
    # Preprocess data
    print("🧹 Preprocessing data...")
    categories_df = data_processor.preprocess_categories(categories_df)
    tickets_df = data_processor.preprocess_tickets(tickets_df)
    
    # Process categories for embedding
    print("🔤 Processing categories for embedding...")
    category_texts = embedding_manager.process_categories(categories_df)
    
    # Load embedding model
    model_name = config['embedding']['model']
    print(f"🤖 Loading embedding model: {model_name}")
    model = embedding_manager.load_sentence_transformer(model_name)
    
    if model is None:
        print("❌ Failed to load embedding model")
        return
    
    embedding_manager.models[model_name] = model
    
    # Generate embeddings
    print("🧠 Generating embeddings for categories...")
    embeddings = embedding_manager.generate_sentence_transformer_embeddings(category_texts, model)
    
    if embeddings is None:
        print("❌ Failed to generate embeddings")
        return
    
    print(f"✅ Generated embeddings: {embeddings.shape}")
    
    # Create metadata
    metadata = {
        'model_name': model_name,
        'timestamp': datetime.now().isoformat(),
        'num_categories': len(categories_df),
        'embedding_dimension': embeddings.shape[1],
        'categories_file': config['data']['categories_file'],
        'preprocessing_steps': [
            'text_cleaning', 
            'llm_enrichment_with_gemini',
            'four_field_extraction',
            'arabic_enhanced_descriptions'
        ],
        'fields_used': [
            'SubCategory_Prefix',
            'SubCategory_Keywords', 
            'SubCategory2_Prefix',
            'SubCategory2_Keywords'
        ],
        'enhancement_method': 'gemini_llm_arabic_enrichment'
    }
    
    # Save embeddings and create FAISS index
    print("💾 Saving embeddings and creating FAISS index...")
    paths = embedding_manager.save_embeddings_and_index(embeddings, model_name, metadata)
    
    # Classify tickets
    print("🎯 Classifying tickets...")
    results = embedding_manager.classify_tickets(tickets_df, model_name, 
                                               limit=config['embedding']['max_tickets'])
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_model_name = model_name.replace("/", "_").replace("-", "_")
    results_path = f"results/experiments/phase2_embeddings/real_ticket_classification_{clean_model_name}_{timestamp}.json"
    
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Results saved to: {results_path}")
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 CLASSIFICATION RESULTS SUMMARY")
    print("=" * 50)
    
    if 'analysis' in results:
        analysis = results['analysis']
        print(f"📈 Total tickets processed: {analysis['total_processed']}")
        print(f"📊 Average confidence: {analysis['average_confidence']:.2f}%")
        print(f"🎯 Classification format: {analysis['classification_format']}")
    
    print(f"\n🔍 Sample Results:")
    for i, result in enumerate(results['results'][:3]):
        ticket_id = result['ticket_id']
        description = result['ticket_description'][:100] + "..." if len(result['ticket_description']) > 100 else result['ticket_description']
        best_match = result['best_match']
        
        print(f"\n📋 Ticket {ticket_id + 1}:")
        print(f"   Description: {description}")
        print(f"   Best Match: {best_match['subcategory']} → {best_match['subcategory2']}")
        print(f"   Confidence: {best_match['confidence']:.1f}%")
    
    print("\n✅ Classification completed successfully!")
    return results_path

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
