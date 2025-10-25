import requests
import json
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

class OllamaClient:
    """Client for communicating with Ollama API"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.api_endpoint = f"{base_url}/api/generate"
        self.models_endpoint = f"{base_url}/api/tags"
    
    def get_available_models(self) -> List[str]:
        """Fetch available models from Ollama"""
        try:
            response = requests.get(self.models_endpoint, timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = [model['name'] for model in data.get('models', [])]
                return models
            return []
        except Exception as e:
            logger.error(f"Error fetching models: {e}")
            return []
    
    def generate_response(self, prompt: str, model: str, stream: bool = False) -> str:
        """Generate response using specified model"""
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": stream,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
            }
            
            response = requests.post(
                self.api_endpoint,
                json=payload,
                timeout=300
            )
            
            if response.status_code == 200:
                if stream:
                    result = ""
                    for line in response.iter_lines():
                        if line:
                            data = json.loads(line)
                            result += data.get('response', '')
                    return result
                else:
                    data = response.json()
                    return data.get('response', '')
            else:
                logger.error(f"Ollama API error: {response.status_code}")
                return ""
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return ""
    
    def check_connection(self) -> bool:
        """Check if Ollama is running and accessible"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Cannot connect to Ollama: {e}")
            return False
