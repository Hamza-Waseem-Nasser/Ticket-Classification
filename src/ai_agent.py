"""
AI Agent Module for Description Generation
Handles OpenAI, Ollama, and Google Gemini backends
"""

import os
from typing import List, Dict, Optional
import requests
import json
from pathlib import Path
import yaml
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
try:
    import google.generativeai as genai
except ImportError:
    genai = None

class AIAgent:
    def __init__(self, config_path: str = "config/config.yaml"):
        if config_path:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        else:
            # Default config when no file provided
            self.config = {
                'ai_agent': {
                    'provider': 'gemini',
                    'model': 'gemini-2.5-flash',
                    'temperature': 0.7,
                    'max_tokens': 500
                }
            }
        
        self.provider = self.config['ai_agent']['provider']
        self.model = self.config['ai_agent']['model']
        self.system_prompt = None  # Can be set externally
        
        if self.provider == "openai":
            self._setup_openai()
        elif self.provider == "ollama":
            self._setup_ollama()
        elif self.provider == "gemini":
            self._setup_gemini()
    
    def _setup_openai(self):
        """Setup OpenAI client"""
        if OpenAI is None:
            raise ImportError("OpenAI package not installed. Install with: pip install openai")
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.client = OpenAI(api_key=api_key)
    
    def _setup_ollama(self):
        """Setup Ollama client"""
        self.ollama_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    
    def _setup_gemini(self):
        """Setup Google Gemini client"""
        if genai is None:
            raise ImportError("Google Generative AI package not installed. Install with: pip install google-generativeai")
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        genai.configure(api_key=api_key)
        
        # Add models/ prefix if not present
        model_name = self.model
        if not model_name.startswith('models/'):
            model_name = f"models/{model_name}"
        
        self.gemini_model = genai.GenerativeModel(model_name)
    
    def generate_description(self, structured_text: str) -> str:
        """Generate rich description for a ticket category"""
        prompt = self.system_prompt if self.system_prompt else self._create_prompt(structured_text)
        
        if self.provider == "openai":
            return self._generate_openai(prompt, structured_text)
        elif self.provider == "ollama":
            return self._generate_ollama(prompt, structured_text)
        elif self.provider == "gemini":
            return self._generate_gemini(prompt, structured_text)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    def _create_prompt(self, structured_text: str) -> str:
        """Create the prompt for description generation"""
        return f"""
You are an expert at creating semantic-rich descriptions for search and embedding systems.

Your task: Generate a comprehensive description for this Saber platform category that will maximize embedding similarity with real user queries.

Category Data:
{structured_text}

Generate a description that includes:

1. CORE PROBLEM SCENARIOS (Arabic & English):
   - How users typically describe this issue
   - Common symptoms and error messages
   - User frustration points and pain descriptions

2. SEMANTIC VARIATIONS:
   - Multiple ways to express the same problem
   - Synonyms and alternative phrasings
   - Both formal and informal language

3. CONTEXTUAL KEYWORDS:
   - Related processes and workflows
   - Platform-specific terminology
   - Business context and use cases

4. USER QUERY PATTERNS:
   - Short queries users might type
   - Longer problem descriptions
   - Question formats users ask

WRITING STYLE:
- Mix Arabic and English naturally (code-switching)
- Include both technical and casual language
- Use problem-focused, user-centric language
- 100-200 words for semantic richness

GOAL: Create text that will have HIGH EMBEDDING SIMILARITY with real user queries about this category.

Description:"""
    
    def _generate_openai(self, prompt: str, structured_text: str = "") -> str:
        """Generate description using OpenAI"""
        try:
            if self.system_prompt:
                # Use system prompt + structured text
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": structured_text}
                ]
            else:
                # Use old prompt format
                messages = [
                    {"role": "system", "content": "You are an expert in support ticket categorization and Arabic-English business processes."},
                    {"role": "user", "content": prompt}
                ]
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.config['ai_agent']['temperature'],
                max_tokens=self.config['ai_agent']['max_tokens']
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return "Error generating description"
    
    def _generate_ollama(self, prompt: str, structured_text: str = "") -> str:
        """Generate description using Ollama"""
        try:
            full_prompt = f"{prompt}\n\n{structured_text}" if self.system_prompt else prompt
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
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
    
    def _generate_gemini(self, prompt: str, structured_text: str = "") -> str:
        """Generate description using Google Gemini"""
        try:
            if self.system_prompt:
                # Use system prompt + structured text
                full_prompt = f"{self.system_prompt}\n\nGiven the category information below, generate a natural problem description that users searching for this category would actually write:\n\n{structured_text}"
            else:
                # Use old prompt format
                full_prompt = prompt
            
            response = self.gemini_model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.config['ai_agent']['temperature'],
                    max_output_tokens=self.config['ai_agent']['max_tokens']
                )
            )
            return response.text.strip()
        except Exception as e:
            print(f"Gemini API error: {e}")
            return f"Error generating description: {e}"
    
    def batch_generate_descriptions(self, structured_texts: List[str]) -> List[str]:
        """Generate descriptions for multiple texts"""
        descriptions = []
        for text in structured_texts:
            description = self.generate_description(text)
            descriptions.append(description)
        return descriptions
