# SRS Generation Agent

A comprehensive LangGraph-based agent that automatically generates System Requirements Specification (SRS) documents from specification materials. Built following the 99_RAG_Note course patterns for LangChain and LangGraph implementation.

## Overview

This agent analyzes specification documents and generates professional SRS documents that include:
- Introduction and scope
- Overall system description
- Functional requirements
- Non-functional requirements
- System interfaces
- Data requirements
- Performance requirements

## Features

### Core Capabilities
- **Multi-document Processing**: Supports text files, PDFs, and other document formats
- **RAG-based Analysis**: Uses retrieval-augmented generation for accurate requirement extraction
- **Structured Workflow**: 11-step LangGraph workflow for comprehensive processing
- **Industry Standards**: Generates SRS documents following IEEE and industry best practices
- **Flexible Configuration**: Customizable models, parameters, and processing options

### Technical Architecture
- **LangGraph State Machine**: Orchestrates the multi-step workflow
- **Vector Store Integration**: FAISS-based document retrieval with MMR search
- **OpenAI Integration**: GPT-4 and GPT-4o models for high-quality analysis
- **Memory Management**: Session-based state persistence
- **Error Handling**: Comprehensive error tracking and recovery

## Installation

### Prerequisites
- Python 3.8+
- OpenAI API key
- Git (optional, for cloning)

### Setup

1. **Clone or download the project files:**
   ```bash
   # If using git
   git clone <repository-url>
   cd srs-generation-agent
   
   # Or download the files directly
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   ```bash
   # Create .env file
   echo "OPENAI_API_KEY=your-api-key-here" > .env
   
   # Or export directly
   export OPENAI_API_KEY="your-api-key-here"
   ```

## Quick Start

### Basic Usage

```python
from srs_generation_agent import SRSGenerationAgent

# Initialize the agent
agent = SRSGenerationAgent(model_name="gpt-4o-mini")

# Generate SRS from specification files
spec_files = ["path/to/spec1.txt", "path/to/spec2.pdf"]
result = agent.generate_srs(spec_files)

if result["success"]:
    # Save the generated SRS document
    agent.save_srs_document(result["srs_document"], "generated_srs.md")
    print(f"Generated {len(result['functional_requirements'])} functional requirements")
else:
    print(f"Generation failed: {result['error']}")
```

### Running the Example

```bash
# Run the comprehensive example and test
python srs_example_usage.py
```

This will:
1. Create sample specification documents
2. Run the SRS generation agent
3. Display results and statistics
4. Save the generated SRS document

## Configuration

### Using Configuration Profiles

```python
from config import AgentConfig, ConfigProfiles

# Use a predefined profile
config = AgentConfig(ConfigProfiles.production())
agent = SRSGenerationAgent(
    model_name=config.model.name,
    temperature=config.model.temperature
)
```

### Available Profiles

- **Development**: Fast processing with GPT-4o-mini
- **Production**: Balanced quality and performance with GPT-4o
- **High Quality**: Maximum quality for complex documents
- **Fast Processing**: Quick results for simple documents

### Custom Configuration

```python
custom_config = {
    "model": {
        "name": "gpt-4o",
        "temperature": 0.1
    },
    "vectorstore": {
        "k": 10,
        "search_type": "mmr"
    }
}

config = AgentConfig(custom_config)
```

## Workflow Steps

The agent processes documents through these steps:

1. **Document Loading**: Load and validate input documents
2. **Document Processing**: Split and preprocess text content
3. **Vector Store Creation**: Create searchable document embeddings
4. **Requirements Analysis**: Perform high-level analysis
5. **Functional Requirements**: Extract functional requirements
6. **Non-Functional Requirements**: Extract quality attributes
7. **System Interfaces**: Identify integration requirements
8. **Data Requirements**: Extract data management needs
9. **Performance Requirements**: Identify performance criteria
10. **Section Generation**: Create structured SRS sections
11. **Document Compilation**: Assemble final SRS document

## Output Format

The generated SRS document includes:

```markdown
# System Requirements Specification (SRS)

## 1. Introduction
- Purpose and scope
- Definitions and references

## 2. Overall Description  
- Product perspective
- Product functions
- User characteristics

## 3. Functional Requirements
- FR-001: User authentication functionality
- FR-002: Data processing capabilities
- ...

## 4. Non-Functional Requirements
- NFR-001: Performance requirements
- NFR-002: Security requirements
- ...

## 5. System Interfaces
- SI-001: External API integrations
- SI-002: Database interfaces
- ...

## 6. Data Requirements
- DR-001: Data storage requirements
- DR-002: Data quality standards
- ...

## 7. Performance Requirements
- PR-001: Response time requirements
- PR-002: Throughput specifications
- ...
```

## API Reference

### SRSGenerationAgent

```python
class SRSGenerationAgent:
    def __init__(self, model_name="gpt-4o", temperature=0.1):
        """Initialize the SRS generation agent"""
        
    def generate_srs(self, spec_files: List[str], thread_id: str = "srs_generation"):
        """Generate SRS document from specification files"""
        
    def save_srs_document(self, srs_document: str, output_path: str) -> bool:
        """Save the generated SRS document to a file"""
```

### Configuration Classes

```python
class AgentConfig:
    """Main configuration class"""
    
    def __init__(self, config_overrides: Dict[str, Any] = None):
        """Initialize with optional overrides"""
        
    @classmethod
    def from_file(cls, config_file: str) -> 'AgentConfig':
        """Load configuration from JSON file"""
        
    def save_to_file(self, config_file: str):
        """Save configuration to JSON file"""
```

## Supported File Formats

- **Text Files**: `.txt`, `.md`
- **PDF Documents**: `.pdf`
- **Word Documents**: `.docx` (with additional setup)

## Error Handling

The agent includes comprehensive error handling:

- **File Validation**: Checks file existence and format
- **Processing Errors**: Handles document loading and processing issues
- **API Errors**: Manages OpenAI API rate limits and errors
- **Memory Management**: Prevents memory issues with large documents

## Performance Considerations

### Document Size Limits
- Maximum file size: 50MB per document
- Recommended chunk size: 1500 characters
- Optimal document count: 1-10 documents per session

### Model Selection
- **GPT-4o**: Best quality, slower processing
- **GPT-4o-mini**: Faster processing, good quality
- **GPT-3.5-turbo**: Fastest, basic quality

### Memory Usage
- Vector store: ~1-2MB per 100 document chunks
- State management: Minimal memory overhead
- Session persistence: Optional for long-running processes

## Troubleshooting

### Common Issues

1. **OpenAI API Key Error**
   ```
   Error: OPENAI_API_KEY environment variable is required
   ```
   **Solution**: Set your OpenAI API key in environment variables

2. **Import Errors**
   ```
   ImportError: No module named 'langchain'
   ```
   **Solution**: Install dependencies with `pip install -r requirements.txt`

3. **Document Loading Errors**
   ```
   Error loading documents: File not found
   ```
   **Solution**: Check file paths and permissions

4. **Memory Issues**
   ```
   Error: Out of memory during processing
   ```
   **Solution**: Reduce document size or chunk size in configuration

### Debugging

Enable debug mode for detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

config = AgentConfig({"debug_mode": True})
```

## Examples

### Enterprise Software Requirements

```python
# Generate SRS for enterprise software
spec_files = [
    "business_requirements.txt",
    "technical_specifications.pdf",
    "user_stories.md"
]

agent = SRSGenerationAgent(model_name="gpt-4o")
result = agent.generate_srs(spec_files, thread_id="enterprise_srs_001")
```

### Healthcare System Requirements

```python
# Generate SRS for healthcare system
config = AgentConfig(ConfigProfiles.high_quality())
agent = SRSGenerationAgent(
    model_name=config.model.name,
    temperature=config.model.temperature
)

result = agent.generate_srs(["healthcare_specs.pdf"])
```

### E-commerce Platform Requirements

```python
# Generate SRS for e-commerce platform
agent = SRSGenerationAgent(model_name="gpt-4o-mini", temperature=0.1)
result = agent.generate_srs([
    "ecommerce_business_requirements.txt",
    "technical_architecture.pdf"
])
```

## Contributing

This agent is based on the 99_RAG_Note course materials and follows the established patterns for LangChain and LangGraph development.

### Development Setup

1. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   pip install pytest jupyter
   ```

2. Run tests:
   ```bash
   python srs_example_usage.py
   ```

3. Code style follows the course materials' patterns

## License

This project follows the licensing terms of the 99_RAG_Note course materials.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the configuration options
3. Enable debug logging for detailed error information
4. Refer to the 99_RAG_Note course materials for LangGraph patterns

---

Built with LangGraph, LangChain, and OpenAI following the 99_RAG_Note course patterns for production-ready AI agents.