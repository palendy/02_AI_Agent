# SRS Generation Agent v2.0

A comprehensive LangChain-based AI agent for generating Software Requirement Specification (SRS) documents specifically tailored for semiconductor firmware development.

## 🚀 Features

### Multi-Format Document Processing
- **PDF Documents**: Extract text and metadata from PDF files
- **DOCX Documents**: Parse Word documents including tables and formatting
- **Text Files**: Support for plain text and markdown files
- **Markdown Files**: Full markdown processing with structure preservation

### Semiconductor Firmware Focus
- **Industry Standards Compliance**: ISO 26262, IEC 61508, DO-178C, MISRA C, AUTOSAR
- **Target Architectures**: ARM Cortex-M, RISC-V, x86, DSP, FPGA support
- **Real-time Constraints**: Hard/soft real-time requirements handling
- **Safety Integrity Levels**: SIL-1 through SIL-4 support
- **Power Management**: Low-power modes and energy efficiency requirements

### AI-Powered Generation
- **LangChain Framework**: Advanced chain and agent architecture
- **Multiple LLM Support**: OpenAI GPT-4/GPT-3.5, Anthropic Claude
- **Vector Search**: FAISS/ChromaDB for intelligent content retrieval
- **Structured Output**: Pydantic models for consistent results

### Validation & Quality Assurance
- **Multi-level Validation**: Basic, Standard, Comprehensive, Certification-ready
- **Quality Metrics**: Completeness, consistency, testability, traceability
- **Compliance Checking**: Automated standard compliance verification
- **Detailed Reports**: Comprehensive validation reports with recommendations

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Dependencies
```bash
# Install required packages
pip install -r requirements.txt
```

### API Keys Setup
Create a `.env` file in the project directory:

```env
# OpenAI Configuration (required for OpenAI models)
OPENAI_API_KEY=your_openai_api_key_here
# OPENAI_ORGANIZATION=your_openai_organization_id_here

# Anthropic Configuration (required for Claude models)
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Configuration
DEBUG=false
LOG_LEVEL=INFO
DEFAULT_PROVIDER=openai
```

## 🎯 Quick Start

### 1. System Test
```bash
# Run system health check
python run.py --test

# Show system information
python run.py --system-info
```

### 2. Create Example Project
```bash
# Generate example project with sample files
python run.py --create-example
```

### 3. Generate SRS Document
```bash
# Basic generation
python run.py --project "Motor Control System" \
              --input requirements.pdf specification.docx \
              --output motor_control_srs.md

# With specific parameters
python run.py --project "Sensor Hub" \
              --input sensor_spec.md hardware_req.pdf \
              --architecture "ARM Cortex-M4" \
              --safety-level "SIL-3" \
              --validation-level comprehensive
```

### 4. Interactive Mode
```bash
# Interactive generation with guided setup
python run.py --interactive
```

### 5. CLI Interface
```bash
# Use rich CLI interface
python run.py --cli

# Or directly
python cli.py generate "My Project" input1.pdf input2.docx
```

## 📋 Usage Examples

### Command Line Interface

#### Generate SRS
```bash
# Simple generation
python cli.py generate "Automotive ECU" requirements.pdf --output ecu_srs.md

# With custom settings
python cli.py generate "IoT Gateway" spec1.md spec2.pdf \
  --architecture "ARM Cortex-M7" \
  --safety-level "SIL-2" \
  --validation-level comprehensive \
  --interactive
```

#### Validate Existing SRS
```bash
# Validate document
python cli.py validate existing_srs.md --level comprehensive --show-details

# Save validation report
python cli.py validate my_srs.md --output validation_report.md
```

#### Analyze Input Files
```bash
# Check file compatibility
python cli.py analyze *.pdf *.docx *.md
```

### Programmatic Usage

```python
from srs_generation_agent import SRSGenerationAgent
from validation_engine import ValidationEngine

# Initialize agent
agent = SRSGenerationAgent()

# Generate SRS
result = agent.generate_srs(
    project_name="Motor Control System",
    file_paths=["requirements.pdf", "specification.docx"],
    target_architecture="ARM Cortex-M4",
    safety_level="SIL-3"
)

# Validate result
validator = ValidationEngine()
validation_result = validator.validate_document(
    result["srs_document"], 
    validation_level="comprehensive"
)

# Save document
agent.save_srs_document(result, "output_srs.md")
```

## 🏗️ Architecture

### Core Components

```
SRS Generation Agent
├── Document Parsers (PDF, DOCX, TXT, MD)
├── LangChain Agent (Chains, Tools, Memory)
├── Template Manager (Semiconductor-specific)
├── Validation Engine (Quality & Compliance)
├── Configuration Management
├── Logging & Error Handling
└── CLI Interface
```

### Key Classes

- **`SRSGenerationAgent`**: Main orchestrator using LangChain
- **`DocumentParserFactory`**: Multi-format document processing
- **`SRSTemplateManager`**: Semiconductor SRS templates
- **`ValidationEngine`**: Quality assurance and compliance
- **`SemiconductorSRSGenerator`**: Domain-specific requirements

### LangChain Integration

- **Chains**: Requirement extraction, section generation, validation
- **Tools**: Document search, template access, requirement generation
- **Agents**: ZERO_SHOT_REACT_DESCRIPTION with custom tools
- **Memory**: Conversation buffer for context retention
- **Vector Stores**: FAISS for semantic document search

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required for OpenAI |
| `ANTHROPIC_API_KEY` | Anthropic API key | Required for Claude |
| `DEFAULT_PROVIDER` | AI provider (`openai`/`anthropic`) | `openai` |
| `MODEL_TEMPERATURE` | Model temperature (0-1) | `0.1` |
| `CHUNK_SIZE` | Document chunk size | `2000` |
| `VECTOR_STORE_TYPE` | Vector store (`faiss`/`chroma`) | `faiss` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `DEBUG` | Debug mode | `false` |

### Configuration Files

- **`.env`**: Environment variables
- **`templates/srs_template.json`**: SRS document structure
- **`templates/validation_rules.json`**: Quality validation rules

## 📊 Validation Levels

### Basic (60% threshold)
- Format validation
- Completeness check
- Basic consistency

### Standard (75% threshold)
- All basic checks
- Testability verification
- Industry standards compliance

### Comprehensive (85% threshold)
- All standard checks
- Traceability analysis
- Cross-reference validation
- Dependency analysis

### Certification Ready (95% threshold)
- All comprehensive checks
- Full compliance verification
- Audit trail completeness
- Formal verification readiness

## 📈 Quality Metrics

The system evaluates SRS documents across five dimensions:

1. **Completeness (25%)**: All required sections and attributes
2. **Consistency (20%)**: Terminology and formatting consistency
3. **Testability (20%)**: Verifiable and measurable requirements
4. **Traceability (15%)**: Source and design traceability
5. **Compliance (20%)**: Industry standards adherence

## 🛠️ Development

### Project Structure
```
SRS_Agent_V2/
├── config.py                    # Configuration management
├── document_parsers.py          # Multi-format document parsing
├── srs_generation_agent.py      # Main LangChain agent
├── srs_template_manager.py      # Template system
├── validation_engine.py         # Quality validation
├── logger.py                    # Logging and error handling
├── cli.py                       # Command line interface
├── run.py                       # Main runner script
├── requirements.txt             # Python dependencies
├── .env                         # Environment configuration
└── templates/
    ├── srs_template.json        # SRS document template
    └── validation_rules.json    # Validation rules
```

### Adding Custom Templates
```python
from srs_template_manager import SRSTemplateManager

# Load custom template
manager = SRSTemplateManager()
manager.load_template("path/to/custom_template.json")

# Generate with custom template
structure = manager.generate_complete_document_structure()
```

### Custom Validation Rules
```json
{
  "validation_rules": {
    "custom_rule": {
      "pattern": "^Custom requirement pattern$",
      "severity": "warning",
      "description": "Custom validation rule"
    }
  }
}
```

## 🧪 Testing

### Unit Tests
```bash
# Run test suite (when implemented)
python -m pytest tests/

# Run specific test
python -m pytest tests/test_document_parsers.py
```

### System Tests
```bash
# Quick system test
python run.py --test

# Manual integration test
python run.py --project "Test" --input example_project/requirements.md
```

## 🔍 Troubleshooting

### Common Issues

#### API Key Errors
```bash
# Check configuration
python cli.py config --show-all

# Verify API keys in .env file
cat .env
```

#### Document Parsing Failures
```bash
# Analyze input files
python cli.py analyze problematic_file.pdf

# Check supported formats
python -c "from document_parsers import DocumentParserFactory; print(DocumentParserFactory.get_supported_extensions())"
```

#### Memory Issues
```bash
# Reduce chunk size
export CHUNK_SIZE=1000

# Use smaller model
export OPENAI_MODEL=gpt-3.5-turbo
```

#### Validation Failures
```bash
# Use lower validation level
python cli.py validate document.md --level basic

# Check validation details
python cli.py validate document.md --show-details
```

## 📚 Documentation

### API Reference
- All public methods are documented with docstrings
- Type hints provided for better IDE support
- Examples included in method documentation

### Logging
- Structured logging with rich formatting
- Multiple log levels and outputs
- Performance monitoring included
- Error tracking and reporting

## 🤝 Contributing

### Development Setup
1. Clone repository
2. Install dependencies: `pip install -r requirements.txt`
3. Configure API keys in `.env`
4. Run tests: `python run.py --test`

### Code Style
- Follow PEP 8 guidelines
- Use type hints
- Document all public methods
- Add error handling

## 📄 License

This project is provided as-is for educational and development purposes.

## 🆘 Support

For issues and questions:
1. Check troubleshooting section
2. Run system diagnostics: `python run.py --system-info`
3. Enable debug mode: `python run.py --debug`
4. Check logs for detailed error information

---

**SRS Generation Agent v2.0** - Transforming requirements into comprehensive Software Requirements Specifications for semiconductor firmware development.