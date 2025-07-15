"""
AI Agent Module for Description Generation
Handles both OpenAI and Ollama backends
"""

import os
from typing import List, Dict, Optional
from openai import OpenAI
import requests
import json
from pathlib import Path
import yaml

class AIAgent:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.provider = self.config['ai_agent']['provider']
        self.model = self.config['ai_agent']['model']
        
        if self.provider == "openai":
            self._setup_openai()
        elif self.provider == "ollama":
            self._setup_ollama()
    
    def _setup_openai(self):
        """Setup OpenAI client"""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.client = OpenAI(api_key=api_key)
    
    def _setup_ollama(self):
        """Setup Ollama client"""
        self.ollama_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    
    def generate_description(self, structured_text: str) -> str:
        """Generate rich description for a ticket category"""
        prompt = self._create_prompt(structured_text)
        
        if self.provider == "openai":
            return self._generate_openai(prompt)
        elif self.provider == "ollama":
            return self._generate_ollama(prompt)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    def _create_prompt(self, structured_text: str) -> str:
        """Create the prompt for description generation"""
        return f"""
Analyze this Saber support ticket category data (Arabic/English) and create a comprehensive, semantic-rich description:

{structured_text}

Create a detailed description that:
1. Explains what business process this category handles in the Saber platform
2. Describes typical user problems and symptoms they experience
3. Includes relevant technical terms and process keywords in both Arabic and English
4. Mentions common workflow scenarios and contexts
5. Provides semantic relationships between prefix, keywords, and category
6. Uses natural language that would help similarity matching for user queries

Focus on creating text that would help an AI system understand when new support tickets belong to this category.
Write the description in both Arabic and English, emphasizing the business context and user intent.

Important: The description should be rich enough to match user queries that might be simpler (e.g., user says "login problem" should match "تسجيل الدخول" category).

Semantic Description:"""
    
    def _generate_openai(self, prompt: str) -> str:
        """Generate description using OpenAI"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert in support ticket categorization and Arabic-English business processes."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config['ai_agent']['temperature'],
                max_tokens=self.config['ai_agent']['max_tokens']
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return "Error generating description"
    
    def _generate_ollama(self, prompt: str) -> str:
        """Generate description using Ollama"""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.config['ai_agent']['temperature'],
                        "num_predict": self.config['ai_agent']['max_tokens']
                    }
                }
            )
            response.raise_for_status()
            return response.json()["response"].strip()
        except Exception as e:
            print(f"Ollama API error: {e}")
            return "Error generating description"
    
    def batch_generate_descriptions(self, structured_texts: List[str]) -> List[str]:
        """Generate descriptions for multiple texts"""
        descriptions = []
        for text in structured_texts:
            description = self.generate_description(text)
            descriptions.append(description)
        return descriptions
