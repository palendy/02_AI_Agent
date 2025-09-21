"""
Configuration management for SRS Generation Agent.
Handles environment variables, model settings, and application configuration.
"""

import os
from typing import Optional, Dict, Any
from pathlib import Path
from pydantic import Field, validator
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class ModelConfig(BaseSettings):
    """Configuration for AI models and providers."""
    
    # OpenAI Configuration
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_organization: Optional[str] = Field(default=None, env="OPENAI_ORGANIZATION")
    openai_model: str = Field(default="gpt-4", env="OPENAI_MODEL")
    
    # Anthropic Configuration
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(default="claude-3-5-sonnet-20241022", env="ANTHROPIC_MODEL")
    
    # Default model provider
    default_provider: str = Field(default="openai", env="DEFAULT_PROVIDER")
    
    # Model parameters
    temperature: float = Field(default=0.1, env="MODEL_TEMPERATURE")
    max_tokens: int = Field(default=4000, env="MODEL_MAX_TOKENS")
    
    @validator("default_provider")
    def validate_provider(cls, v):
        if v not in ["openai", "anthropic"]:
            raise ValueError("Provider must be 'openai' or 'anthropic'")
        return v


class ProcessingConfig(BaseSettings):
    """Configuration for document processing and generation."""
    
    # File processing
    max_file_size_mb: int = Field(default=50, env="MAX_FILE_SIZE_MB")
    supported_formats: list = Field(default=["pdf", "docx", "txt", "md"])
    chunk_size: int = Field(default=2000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="CHUNK_OVERLAP")
    
    # Vector store configuration
    vector_store_type: str = Field(default="faiss", env="VECTOR_STORE_TYPE")
    embedding_model: str = Field(default="text-embedding-ada-002", env="EMBEDDING_MODEL")
    
    # SRS generation
    min_requirements_per_section: int = Field(default=3, env="MIN_REQUIREMENTS_PER_SECTION")
    max_requirements_per_section: int = Field(default=15, env="MAX_REQUIREMENTS_PER_SECTION")
    
    @validator("vector_store_type")
    def validate_vector_store(cls, v):
        if v not in ["faiss", "chroma"]:
            raise ValueError("Vector store type must be 'faiss' or 'chroma'")
        return v


class LoggingConfig(BaseSettings):
    """Configuration for logging and debugging."""
    
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: Optional[str] = Field(default=None, env="LOG_FILE")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT"
    )
    
    @validator("log_level")
    def validate_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()


class PathConfig(BaseSettings):
    """Configuration for file paths and directories."""
    
    # Base directories
    base_dir: Path = Field(default_factory=lambda: Path(__file__).parent)
    output_dir: Path = Field(default_factory=lambda: Path(__file__).parent / "output")
    templates_dir: Path = Field(default_factory=lambda: Path(__file__).parent / "templates")
    temp_dir: Path = Field(default_factory=lambda: Path(__file__).parent / "temp")
    
    # Specific files
    srs_template_file: str = Field(default="srs_template.json", env="SRS_TEMPLATE_FILE")
    validation_rules_file: str = Field(default="validation_rules.json", env="VALIDATION_RULES_FILE")
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        for dir_path in [self.output_dir, self.templates_dir, self.temp_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)


class SemiconductorConfig(BaseSettings):
    """Configuration specific to semiconductor firmware development."""
    
    # Industry standards
    compliance_standards: list = Field(default=[
        "ISO 26262", "IEC 61508", "DO-178C", "MISRA C", "AUTOSAR"
    ])
    
    # Semiconductor-specific requirements
    target_architectures: list = Field(default=[
        "ARM Cortex-M", "RISC-V", "x86", "DSP", "FPGA"
    ])
    
    # Performance requirements
    real_time_constraints: bool = Field(default=True, env="REAL_TIME_CONSTRAINTS")
    safety_critical: bool = Field(default=True, env="SAFETY_CRITICAL")
    
    # Testing requirements
    code_coverage_threshold: float = Field(default=95.0, env="CODE_COVERAGE_THRESHOLD")
    static_analysis_required: bool = Field(default=True, env="STATIC_ANALYSIS_REQUIRED")


class AppConfig:
    """Main application configuration class that combines all config sections."""
    
    def __init__(self):
        self.model = ModelConfig()
        self.processing = ProcessingConfig()
        self.logging = LoggingConfig()
        self.paths = PathConfig()
        self.semiconductor = SemiconductorConfig()
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate the overall configuration."""
        # Check if at least one AI provider is configured
        if not self.model.openai_api_key and not self.model.anthropic_api_key:
            raise ValueError(
                "At least one AI provider API key must be configured. "
                "Please set OPENAI_API_KEY or ANTHROPIC_API_KEY in your .env file."
            )
        
        # Validate selected provider has API key
        if self.model.default_provider == "openai" and not self.model.openai_api_key:
            raise ValueError("OpenAI API key is required when using OpenAI as default provider")
        
        if self.model.default_provider == "anthropic" and not self.model.anthropic_api_key:
            raise ValueError("Anthropic API key is required when using Anthropic as default provider")
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration for the selected provider."""
        if self.model.default_provider == "openai":
            return {
                "provider": "openai",
                "api_key": self.model.openai_api_key,
                "model": self.model.openai_model,
                "temperature": self.model.temperature,
                "max_tokens": self.model.max_tokens,
                "organization": self.model.openai_organization
            }
        else:
            return {
                "provider": "anthropic",
                "api_key": self.model.anthropic_api_key,
                "model": self.model.anthropic_model,
                "temperature": self.model.temperature,
                "max_tokens": self.model.max_tokens
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format."""
        return {
            "model": self.model.dict(),
            "processing": self.processing.dict(),
            "logging": self.logging.dict(),
            "paths": {
                "base_dir": str(self.paths.base_dir),
                "output_dir": str(self.paths.output_dir),
                "templates_dir": str(self.paths.templates_dir),
                "temp_dir": str(self.paths.temp_dir),
                "srs_template_file": self.paths.srs_template_file,
                "validation_rules_file": self.paths.validation_rules_file
            },
            "semiconductor": self.semiconductor.dict()
        }


# Global configuration instance
config = AppConfig()


def get_config() -> AppConfig:
    """Get the global configuration instance."""
    return config


def reload_config():
    """Reload configuration from environment variables."""
    global config
    load_dotenv(override=True)
    config = AppConfig()
    return config


# Configuration validation utilities
def validate_file_path(file_path: str) -> bool:
    """Validate if a file path exists and is accessible."""
    try:
        path = Path(file_path)
        return path.exists() and path.is_file()
    except Exception:
        return False


def validate_directory_path(dir_path: str) -> bool:
    """Validate if a directory path exists and is accessible."""
    try:
        path = Path(dir_path)
        return path.exists() and path.is_dir()
    except Exception:
        return False


def get_supported_file_extensions() -> list:
    """Get list of supported file extensions."""
    return [f".{fmt}" for fmt in config.processing.supported_formats]


def is_supported_file(file_path: str) -> bool:
    """Check if a file is in a supported format."""
    path = Path(file_path)
    return path.suffix.lower().lstrip('.') in config.processing.supported_formats