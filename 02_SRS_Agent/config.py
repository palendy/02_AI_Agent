"""
Configuration settings for SRS Generation Agent
=============================================

Configuration management following 99_RAG_Note course patterns
for environment setup and agent customization.
"""

import os
from dataclasses import dataclass
from typing import Dict, Any, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class ModelConfig:
    """Model configuration settings"""
    name: str = "gpt-4o-mini"
    temperature: float = 0.1
    max_tokens: int = 4000
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0


@dataclass
class EmbeddingConfig:
    """Embedding model configuration"""
    model: str = "text-embedding-3-small"
    chunk_size: int = 1500
    chunk_overlap: int = 200
    separators: List[str] = None
    
    def __post_init__(self):
        if self.separators is None:
            self.separators = ["\n\n", "\n", ".", "!", "?", ",", " ", ""]


@dataclass
class VectorStoreConfig:
    """Vector store configuration"""
    type: str = "faiss"  # faiss, chroma, pinecone
    search_type: str = "mmr"  # similarity, mmr
    k: int = 8
    fetch_k: int = 20
    lambda_mult: float = 0.5


@dataclass
class WorkflowConfig:
    """Workflow and processing configuration"""
    max_document_size_mb: int = 50
    supported_formats: List[str] = None
    timeout_seconds: int = 300
    retry_attempts: int = 3
    
    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = ['.txt', '.pdf', '.docx', '.md']


@dataclass
class SRSConfig:
    """SRS document generation configuration"""
    include_toc: bool = True
    include_metadata: bool = True
    requirements_prefix_map: Dict[str, str] = None
    section_templates: Dict[str, str] = None
    
    def __post_init__(self):
        if self.requirements_prefix_map is None:
            self.requirements_prefix_map = {
                "functional": "FR",
                "non_functional": "NFR",
                "system_interfaces": "SI",
                "data": "DR",
                "performance": "PR"
            }
        
        if self.section_templates is None:
            self.section_templates = {
                "introduction": "1. Introduction",
                "overall_description": "2. Overall Description",
                "functional_requirements": "3. Functional Requirements",
                "non_functional_requirements": "4. Non-Functional Requirements",
                "system_interfaces": "5. System Interfaces",
                "data_requirements": "6. Data Requirements",
                "performance_requirements": "7. Performance Requirements"
            }


class AgentConfig:
    """Main configuration class for SRS Generation Agent"""
    
    def __init__(self, config_overrides: Dict[str, Any] = None):
        """Initialize configuration with optional overrides"""
        
        # Load base configurations
        self.model = ModelConfig()
        self.embedding = EmbeddingConfig()
        self.vectorstore = VectorStoreConfig()
        self.workflow = WorkflowConfig()
        self.srs = SRSConfig()
        
        # Environment variables
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_organization = os.getenv("OPENAI_ORGANIZATION")
        self.debug_mode = os.getenv("DEBUG", "false").lower() == "true"
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        
        # Apply overrides if provided
        if config_overrides:
            self._apply_overrides(config_overrides)
        
        # Validate configuration
        self._validate_config()
    
    def _apply_overrides(self, overrides: Dict[str, Any]):
        """Apply configuration overrides"""
        for key, value in overrides.items():
            if hasattr(self, key):
                if isinstance(getattr(self, key), (ModelConfig, EmbeddingConfig, 
                                                 VectorStoreConfig, WorkflowConfig, SRSConfig)):
                    # Handle nested config objects
                    config_obj = getattr(self, key)
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            if hasattr(config_obj, sub_key):
                                setattr(config_obj, sub_key, sub_value)
                else:
                    setattr(self, key, value)
    
    def _validate_config(self):
        """Validate configuration settings"""
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        if self.embedding.chunk_size <= 0:
            raise ValueError("Chunk size must be positive")
        
        if self.vectorstore.k <= 0:
            raise ValueError("Vector store k parameter must be positive")
        
        if self.workflow.max_document_size_mb <= 0:
            raise ValueError("Maximum document size must be positive")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "model": self.model.__dict__,
            "embedding": self.embedding.__dict__,
            "vectorstore": self.vectorstore.__dict__,
            "workflow": self.workflow.__dict__,
            "srs": self.srs.__dict__,
            "environment": {
                "openai_api_key": "***" if self.openai_api_key else None,
                "openai_organization": self.openai_organization,
                "debug_mode": self.debug_mode,
                "log_level": self.log_level
            }
        }
    
    @classmethod
    def from_file(cls, config_file: str) -> 'AgentConfig':
        """Load configuration from JSON file"""
        import json
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            return cls(config_data)
        except FileNotFoundError:
            raise ValueError(f"Configuration file not found: {config_file}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}")
    
    def save_to_file(self, config_file: str):
        """Save configuration to JSON file"""
        import json
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


# Predefined configuration profiles
class ConfigProfiles:
    """Predefined configuration profiles for different use cases"""
    
    @staticmethod
    def development() -> Dict[str, Any]:
        """Development configuration with faster, cheaper model"""
        return {
            "model": {
                "name": "gpt-4o-mini",
                "temperature": 0.2,
                "max_tokens": 2000
            },
            "vectorstore": {
                "k": 5,
                "fetch_k": 10
            },
            "debug_mode": True,
            "log_level": "DEBUG"
        }
    
    @staticmethod
    def production() -> Dict[str, Any]:
        """Production configuration with higher quality model"""
        return {
            "model": {
                "name": "gpt-4o",
                "temperature": 0.1,
                "max_tokens": 4000
            },
            "vectorstore": {
                "k": 8,
                "fetch_k": 20
            },
            "workflow": {
                "timeout_seconds": 600,
                "retry_attempts": 3
            },
            "debug_mode": False,
            "log_level": "INFO"
        }
    
    @staticmethod
    def high_quality() -> Dict[str, Any]:
        """High quality configuration for complex documents"""
        return {
            "model": {
                "name": "gpt-4o",
                "temperature": 0.05,
                "max_tokens": 8000
            },
            "embedding": {
                "chunk_size": 1200,
                "chunk_overlap": 300
            },
            "vectorstore": {
                "k": 12,
                "fetch_k": 30,
                "lambda_mult": 0.3
            },
            "debug_mode": False,
            "log_level": "INFO"
        }
    
    @staticmethod
    def fast_processing() -> Dict[str, Any]:
        """Fast processing configuration for quick results"""
        return {
            "model": {
                "name": "gpt-4o-mini",
                "temperature": 0.3,
                "max_tokens": 1500
            },
            "embedding": {
                "chunk_size": 2000,
                "chunk_overlap": 100
            },
            "vectorstore": {
                "k": 4,
                "fetch_k": 8
            },
            "workflow": {
                "timeout_seconds": 120
            },
            "debug_mode": False,
            "log_level": "WARNING"
        }


# Default configuration instance
DEFAULT_CONFIG = AgentConfig()


if __name__ == "__main__":
    # Test configuration loading and profiles
    print("SRS Generation Agent Configuration")
    print("=" * 50)
    
    # Test default configuration
    print("\n1. Default Configuration:")
    config = AgentConfig()
    print(f"   Model: {config.model.name}")
    print(f"   Temperature: {config.model.temperature}")
    print(f"   Chunk Size: {config.embedding.chunk_size}")
    print(f"   Vector K: {config.vectorstore.k}")
    
    # Test development profile
    print("\n2. Development Profile:")
    dev_config = AgentConfig(ConfigProfiles.development())
    print(f"   Model: {dev_config.model.name}")
    print(f"   Temperature: {dev_config.model.temperature}")
    print(f"   Debug Mode: {dev_config.debug_mode}")
    
    # Test production profile
    print("\n3. Production Profile:")
    prod_config = AgentConfig(ConfigProfiles.production())
    print(f"   Model: {prod_config.model.name}")
    print(f"   Temperature: {prod_config.model.temperature}")
    print(f"   Timeout: {prod_config.workflow.timeout_seconds}s")
    
    # Save sample configuration
    config_file = "sample_config.json"
    config.save_to_file(config_file)
    print(f"\n4. Sample configuration saved to: {config_file}")
    
    print("\nConfiguration test completed successfully!")