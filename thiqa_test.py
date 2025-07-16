"""
Thiqa Testing Script for Saber Ticket Classification
Tests classification accuracy against Thiqa ground truth data
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

def get_confidence_color(confidence):
    """Get color coding based on confidence score"""
    if confidence >= 85:
        return "🟢 HIGH"
    elif confidence >= 65:
        return "🟡 MEDIUM" 
    else:
        return "🔴 LOW"

def calculate_accuracy(results_df):
    """Calculate accuracy metrics"""
    # Level 1 accuracy (Subcategory)
    level1_correct = (results_df['Predicted_Subcategory'] == results_df['Subcategory_Thiqah']).sum()
    level1_total = len(results_df)
    level1_accuracy = (level1_correct / level1_total) * 100
    
    # Level 2 accuracy (Subcategory2) - handle empty ground truth
    level2_mask = results_df['Subcategory2_Thiqah'].notna() & (results_df['Subcategory2_Thiqah'] != '')
    level2_filtered = results_df[level2_mask]
    
    if len(level2_filtered) > 0:
        level2_correct = (level2_filtered['Predicted_Subcategory2'] == level2_filtered['Subcategory2_Thiqah']).sum()
        level2_accuracy = (level2_correct / len(level2_filtered)) * 100
    else:
        level2_accuracy = 0
    
    # Empty ground truth count (automatically correct)
    empty_ground_truth = len(results_df) - len(level2_filtered)
    
    return {
        'level1_accuracy': level1_accuracy,
        'level2_accuracy': level2_accuracy,
        'level1_correct': level1_correct,
        'level1_total': level1_total,
        'level2_correct': level2_correct,
        'level2_total': len(level2_filtered),
        'empty_ground_truth': empty_ground_truth
    }

def main():
    """Main function to run Thiqa testing"""
    
    # Configuration
    config = {
        'data': {
            'categories_file': 'Saber Categories-1.csv',
            'thiqa_file': 'Thiqa-Cleaned-by-me.csv'
        },
        'embedding': {
            'model': 'text-embedding-3-small',  # OpenAI model
            'max_tickets': 100  # Test more tickets
        }
    }
    
    print("🧪 Starting Thiqa Classification Testing")
    print("=" * 50)
    
    # Initialize processors
    data_processor = DataProcessor(config)
    embedding_manager = EmbeddingManager(config)
    
    # Load Saber categories
    print("📊 Loading Saber categories...")
    categories_df = data_processor.load_categories(config['data']['categories_file'])
    if categories_df.empty:
        print("❌ Failed to load Saber categories")
        return
    
    # Load Thiqa test data
    print("📊 Loading Thiqa test data...")
    thiqa_df = pd.read_csv(config['data']['thiqa_file'])
    thiqa_df = thiqa_df.head(config['embedding']['max_tickets'])  # Subset
    print(f"✅ Loaded {len(categories_df)} Saber categories and {len(thiqa_df)} Thiqa tickets")
    
    # Preprocess Saber categories
    print("🧹 Preprocessing Saber categories...")
    categories_df = data_processor.preprocess_categories(categories_df)
    
    # Process categories for embedding (reuse existing descriptions)
    print("🔤 Processing categories for embedding...")
    model_name = config['embedding']['model']
    category_texts = embedding_manager.process_categories(categories_df, model_name)
    
    # Load embedding model
    print(f"🤖 Loading embedding model: {model_name}")
    if 'text-embedding' in model_name:
        model = embedding_manager.load_openai_model(model_name)
    else:
        model = embedding_manager.load_sentence_transformer(model_name)
    
    if model is None:
        print("❌ Failed to load embedding model")
        return
    
    # Generate embeddings for categories
    print("🧠 Generating embeddings for categories...")
    if 'text-embedding' in model_name:
        embeddings = embedding_manager.generate_openai_embeddings(category_texts, model_name)
    else:
        embeddings = embedding_manager.generate_sentence_transformer_embeddings(category_texts, model)
    
    if embeddings is None:
        print("❌ Failed to generate embeddings")
        return
    
    print(f"✅ Generated embeddings: {embeddings.shape}")
    
    # Create metadata
    metadata = {
        'model_name': model_name,
        'embedding_dimension': embeddings.shape[1],
        'num_categories': embeddings.shape[0],
        'timestamp': datetime.now().isoformat(),
        'test_type': 'thiqa_validation',
        'categories_file': config['data']['categories_file'],
        'thiqa_file': config['data']['thiqa_file'],
        'subset_size': len(thiqa_df)
    }
    
    # Save embeddings and create FAISS index
    print("💾 Saving embeddings and creating FAISS index...")
    paths = embedding_manager.save_embeddings_and_index(embeddings, model_name, metadata)
    
    # Classify Thiqa tickets
    print("🎯 Classifying Thiqa tickets...")
    results = []
    
    for idx, row in thiqa_df.iterrows():
        incident_id = row['Incident']
        description = row['Description']
        ground_truth_sub = row['Subcategory_Thiqah']
        ground_truth_sub2 = row['Subcategory2_Thiqah']
        
        # Get classification
        classifications = embedding_manager.search_similar_categories(description, model_name, top_k=1)
        
        if classifications:
            best_match = classifications[0]
            predicted_sub = best_match['subcategory']
            predicted_sub2 = best_match['subcategory2']
            confidence = best_match['confidence']
            confidence_color = get_confidence_color(confidence)
            
            # Check accuracy
            level1_correct = predicted_sub == ground_truth_sub
            
            # Handle empty ground truth for level 2
            if pd.isna(ground_truth_sub2) or ground_truth_sub2 == '':
                level2_correct = True  # Count as correct
                level2_note = "Empty ground truth"
            else:
                level2_correct = predicted_sub2 == ground_truth_sub2
                level2_note = "Compared"
            
        else:
            predicted_sub = "NO_MATCH"
            predicted_sub2 = "NO_MATCH"
            confidence = 0
            confidence_color = "🔴 LOW"
            level1_correct = False
            level2_correct = False
            level2_note = "No prediction"
        
        results.append({
            'Incident': incident_id,
            'Description': description,
            'Predicted_Subcategory': predicted_sub,
            'Predicted_Subcategory2': predicted_sub2,
            'Subcategory_Thiqah': ground_truth_sub,
            'Subcategory2_Thiqah': ground_truth_sub2,
            'Confidence': f"{confidence:.1f}%",
            'Confidence_Level': confidence_color,
            'Level1_Correct': level1_correct,
            'Level2_Correct': level2_correct,
            'Level2_Note': level2_note
        })
        
        print(f"✅ Classified ticket {idx+1}/{len(thiqa_df)}: {confidence:.1f}% confidence")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate accuracy
    accuracy_metrics = calculate_accuracy(results_df)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = f"results/thiqa_test_results_{timestamp}.csv"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    results_df.to_csv(results_path, index=False, encoding='utf-8')
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 THIQA TESTING RESULTS SUMMARY")
    print("=" * 50)
    print(f"📈 Total tickets tested: {len(thiqa_df)}")
    print(f"🎯 Level 1 Accuracy (Subcategory): {accuracy_metrics['level1_accuracy']:.1f}%")
    print(f"🎯 Level 2 Accuracy (Subcategory2): {accuracy_metrics['level2_accuracy']:.1f}%")
    print(f"📊 Level 1: {accuracy_metrics['level1_correct']}/{accuracy_metrics['level1_total']} correct")
    print(f"📊 Level 2: {accuracy_metrics['level2_correct']}/{accuracy_metrics['level2_total']} correct")
    print(f"📝 Empty ground truth (auto-correct): {accuracy_metrics['empty_ground_truth']}")
    print(f"💾 Results saved to: {results_path}")
    
    print("\n🎨 Confidence Color Legend:")
    print("🟢 HIGH (85-100%): Very confident predictions")
    print("🟡 MEDIUM (65-84%): Moderately confident predictions") 
    print("🔴 LOW (0-64%): Uncertain predictions")
    
    print("✅ Thiqa testing completed successfully!")

if __name__ == "__main__":
    main()
