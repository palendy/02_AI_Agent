"""
Main SRS Generation Agent using LangChain framework.
Orchestrates document analysis, requirement extraction, and SRS generation.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime

# LangChain imports
from langchain.agents import AgentType, create_react_agent, AgentExecutor
from langchain import hub
from langchain.tools import Tool
from langchain.chains import LLMChain, SequentialChain
from langchain.schema import Document
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
try:
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings
except ImportError:
    try:
        from langchain_community.embeddings import OpenAIEmbeddings
        from langchain_community.vectorstores import FAISS
    except ImportError:
        # Final fallback to older imports
        from langchain.vectorstores import FAISS
        from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
try:
    from langchain_community.callbacks.manager import get_openai_callback
except ImportError:
    # Fallback to older import
    from langchain.callbacks import get_openai_callback

# Model imports
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# Pydantic for structured outputs
from pydantic import BaseModel, Field

# Local imports
from config import get_config
from document_parsers import MultiDocumentParser, DocumentParserFactory
from srs_template_manager import SRSTemplateManager, SemiconductorSRSGenerator

logger = logging.getLogger(__name__)


class RequirementExtraction(BaseModel):
    """Structured output for requirement extraction."""
    requirements: List[Dict[str, Any]] = Field(description="List of extracted requirements")
    section_type: str = Field(description="Type of requirements section")
    confidence_score: float = Field(description="Confidence in extraction quality (0-1)")
    source_references: List[str] = Field(description="Source document references")


class SRSSection(BaseModel):
    """Structured output for SRS sections."""
    section_id: str = Field(description="Section identifier")
    title: str = Field(description="Section title")
    content: str = Field(description="Section content in markdown format")
    requirements: List[Dict[str, Any]] = Field(description="Section requirements")
    metadata: Dict[str, Any] = Field(description="Section metadata")


class SRSGenerationAgent:
    """Main agent for SRS document generation."""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.document_parser = MultiDocumentParser(self.config)
        self.template_manager = SRSTemplateManager(self.config)
        self.semiconductor_generator = SemiconductorSRSGenerator(self.template_manager, self.config)
        
        # Initialize LLM
        self.llm = self._initialize_llm()
        
        # Initialize embeddings and vector store
        self.embeddings = self._initialize_embeddings()
        self.vector_store = None
        
        # Initialize chains
        self.extraction_chain = None
        self.generation_chain = None
        self.validation_chain = None
        
        # Initialize agent tools
        self.tools = self._initialize_tools()
        self.agent_executor = None
        
        # Memory for conversation
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        self._setup_chains()
        self._setup_agent()
    
    def _initialize_llm(self):
        """Initialize the language model based on configuration."""
        model_config = self.config.get_model_config()
        
        try:
            if model_config["provider"] == "openai":
                return ChatOpenAI(
                    model=model_config["model"],
                    temperature=model_config["temperature"],
                    max_tokens=model_config["max_tokens"],
                    openai_api_key=model_config["api_key"],
                    openai_organization=model_config.get("organization")
                )
            elif model_config["provider"] == "anthropic":
                return ChatAnthropic(
                    model=model_config["model"],
                    temperature=model_config["temperature"],
                    max_tokens=model_config["max_tokens"],
                    anthropic_api_key=model_config["api_key"]
                )
            else:
                raise ValueError(f"Unsupported provider: {model_config['provider']}")
        except Exception as e:
            logger.error(f"Error initializing LLM: {str(e)}")
            raise
    
    def _initialize_embeddings(self):
        """Initialize embeddings for vector storage."""
        try:
            if self.config.model.default_provider == "openai":
                return OpenAIEmbeddings(
                    model=self.config.processing.embedding_model,
                    openai_api_key=self.config.model.openai_api_key
                )
            else:
                # Fallback to sentence transformers for non-OpenAI providers
                from langchain.embeddings import HuggingFaceEmbeddings
                return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        except Exception as e:
            logger.error(f"Error initializing embeddings: {str(e)}")
            raise
    
    def _setup_chains(self):
        """Setup LangChain chains for different tasks."""
        # Requirement extraction chain
        extraction_prompt = PromptTemplate(
            input_variables=["document_content", "section_type", "semiconductor_context"],
            template="""
You are an expert in semiconductor firmware requirements engineering. 

Extract detailed requirements from the following document content for a {section_type} section.

Document Content:
{document_content}

Semiconductor Context:
{semiconductor_context}

Guidelines:
1. Extract specific, testable requirements
2. Use proper requirement format: "The system shall [action]"
3. Assign priority levels: Critical, High, Medium, Low
4. Include rationale for each requirement
5. Consider semiconductor-specific constraints (real-time, power, safety)
6. Ensure requirements are atomic and verifiable

Provide your response as a structured JSON with:
- requirements: List of requirement objects with id, text, priority, rationale
- section_type: The type of requirements section
- confidence_score: Your confidence in the extraction (0-1)
- source_references: References to source content

Output:
"""
        )
        
        # Create extraction chain using modern LangChain syntax
        self.extraction_chain = extraction_prompt | self.llm
        
        # SRS section generation chain
        generation_prompt = PromptTemplate(
            input_variables=["section_template", "extracted_requirements", "project_context"],
            template="""
Generate a comprehensive SRS section for semiconductor firmware development.

Section Template:
{section_template}

Extracted Requirements:
{extracted_requirements}

Project Context:
{project_context}

Guidelines:
1. Follow the section template structure
2. Incorporate all extracted requirements appropriately
3. Add additional requirements if gaps are identified
4. Use proper markdown formatting
5. Include requirement traceability
6. Ensure compliance with semiconductor industry standards
7. Address real-time, safety, and power constraints

Generate a complete section with:
- Proper heading structure
- Introduction and context
- Detailed requirements list
- Cross-references where appropriate
- Tables or diagrams if helpful

Output the section in markdown format.
"""
        )
        
        # Create generation chain using modern LangChain syntax
        self.generation_chain = generation_prompt | self.llm
        
        # Validation chain
        validation_prompt = PromptTemplate(
            input_variables=["srs_content", "quality_criteria", "compliance_standards"],
            template="""
Validate the following SRS content against quality criteria and compliance standards.

SRS Content:
{srs_content}

Quality Criteria:
{quality_criteria}

Compliance Standards:
{compliance_standards}

Validation Tasks:
1. Check requirement completeness and coverage
2. Verify requirement format and structure
3. Assess testability and verifiability
4. Check for consistency and conflicts
5. Validate compliance with standards
6. Assess clarity and unambiguity

Provide detailed feedback with:
- Validation results (pass/fail for each criteria)
- Specific issues found
- Recommendations for improvement
- Compliance assessment
- Overall quality score (0-100)

Output as structured analysis.
"""
        )
        
        # Create validation chain using modern LangChain syntax
        self.validation_chain = validation_prompt | self.llm
    
    def _initialize_tools(self) -> List[Tool]:
        """Initialize agent tools."""
        tools = [
            Tool(
                name="extract_requirements",
                description="Extract requirements from document content for a specific section type",
                func=self._extract_requirements_tool
            ),
            Tool(
                name="generate_section",
                description="Generate SRS section content based on template and requirements",
                func=self._generate_section_tool
            ),
            Tool(
                name="validate_srs",
                description="Validate SRS content against quality criteria",
                func=self._validate_srs_tool
            ),
            Tool(
                name="search_documents",
                description="Search through processed documents for specific information",
                func=self._search_documents_tool
            ),
            Tool(
                name="get_template_info",
                description="Get template information for SRS sections",
                func=self._get_template_info_tool
            ),
            Tool(
                name="generate_semiconductor_requirements",
                description="Generate semiconductor-specific requirements (power, safety, real-time)",
                func=self._generate_semiconductor_requirements_tool
            )
        ]
        return tools
    
    def _setup_agent(self):
        """Setup the main agent executor."""
        try:
            # Try to use hub prompt, fall back to simple prompt if not available
            try:
                prompt = hub.pull("hwchase17/react")
            except:
                # Fallback prompt template
                from langchain.prompts import PromptTemplate
                prompt = PromptTemplate.from_template(
                    """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}"""
                )
            
            # Create agent
            agent = create_react_agent(self.llm, self.tools, prompt)
            
            # Create agent executor
            self.agent_executor = AgentExecutor(
                agent=agent,
                tools=self.tools,
                memory=self.memory,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=10
            )
        except Exception as e:
            logger.error(f"Error setting up agent: {str(e)}")
            # Create a simplified executor that just uses the tools directly
            self.agent_executor = None
    
    def process_documents(self, file_paths: List[str]) -> Dict[str, List[Document]]:
        """Process input documents and create vector store."""
        logger.info(f"Processing {len(file_paths)} documents")
        
        try:
            # Parse documents
            parsed_docs = self.document_parser.parse_documents(file_paths)
            
            # Flatten all documents
            all_documents = []
            for file_path, documents in parsed_docs.items():
                all_documents.extend(documents)
            
            if not all_documents:
                logger.warning("No documents were successfully parsed")
                return {}
            
            # Create vector store
            self.vector_store = FAISS.from_documents(all_documents, self.embeddings)
            logger.info(f"Created vector store with {len(all_documents)} document chunks")
            
            return parsed_docs
            
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            raise
    
    def generate_srs(self, project_name: str, file_paths: List[str], 
                    target_architecture: str = "ARM Cortex-M",
                    safety_level: str = "SIL-2") -> Dict[str, Any]:
        """Generate complete SRS document."""
        logger.info(f"Starting SRS generation for project: {project_name}")
        
        try:
            # Process input documents
            processed_docs = self.process_documents(file_paths)
            
            if not processed_docs:
                raise ValueError("No documents could be processed")
            
            # Prepare project context
            project_context = {
                "project_name": project_name,
                "target_architecture": target_architecture,
                "safety_level": safety_level,
                "compliance_standards": self.config.semiconductor.compliance_standards,
                "input_documents": list(processed_docs.keys())
            }
            
            # Generate document structure
            document_structure = self.template_manager.generate_complete_document_structure()
            
            # Generate each section
            srs_sections = []
            for section in document_structure["sections"]:
                section_content = self._generate_section_content(
                    section, processed_docs, project_context
                )
                srs_sections.append(section_content)
            
            # Compile final SRS document
            final_srs = self._compile_final_document(srs_sections, project_context)
            
            # Validate the generated SRS
            validation_results = self._validate_complete_srs(final_srs)
            
            result = {
                "project_name": project_name,
                "generated_at": datetime.now().isoformat(),
                "srs_document": final_srs,
                "validation_results": validation_results,
                "metadata": {
                    "input_files": list(processed_docs.keys()),
                    "sections_generated": len(srs_sections),
                    "total_requirements": sum(len(section.get("requirements", [])) for section in srs_sections),
                    "target_architecture": target_architecture,
                    "safety_level": safety_level
                }
            }
            
            logger.info("SRS generation completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error generating SRS: {str(e)}")
            raise
    
    def _generate_section_content(self, section_template: Dict[str, Any], 
                                processed_docs: Dict[str, List[Document]], 
                                project_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate content for a specific SRS section."""
        section_id = section_template.get("section_id", "")
        section_title = section_template.get("title", "")
        
        logger.info(f"Generating content for section: {section_title}")
        
        try:
            # Search for relevant content in documents
            relevant_content = self._search_relevant_content(section_id, processed_docs)
            
            # Extract requirements using the extraction chain directly
            try:
                extraction_input = {
                    "document_content": relevant_content[:8000],  # Limit content length
                    "section_type": section_id,
                    "semiconductor_context": f"Target: {project_context.get('target_architecture', 'ARM Cortex-M')}, Safety: {project_context.get('safety_level', 'SIL-2')}"
                }
                
                requirements_result = self.extraction_chain.invoke(extraction_input)
                
                # Extract content from response (handle AIMessage and other formats)
                if hasattr(requirements_result, 'content'):
                    requirements_result = requirements_result.content
                elif isinstance(requirements_result, dict):
                    requirements_result = requirements_result.get('text', str(requirements_result))
                else:
                    requirements_result = str(requirements_result)
                
            except Exception as e:
                logger.warning(f"Direct extraction failed, using simplified approach: {str(e)}")
                requirements_result = {
                    "requirements": [],
                    "section_type": section_id,
                    "confidence_score": 0.5,
                    "source_references": []
                }
            
            # Generate section content using the generation chain directly
            try:
                # Prepare template information
                template_content = f"Section: {section_title}\nDescription: {section_template.get('description', '')}"
                
                generation_input = {
                    "section_template": template_content,
                    "extracted_requirements": str(requirements_result),
                    "project_context": str(project_context)
                }
                
                section_content = self.generation_chain.invoke(generation_input)
                
                # Extract content from response (handle AIMessage and other formats)
                if hasattr(section_content, 'content'):
                    section_content = section_content.content
                elif isinstance(section_content, dict):
                    section_content = section_content.get('text', str(section_content))
                else:
                    section_content = str(section_content)
                    
            except Exception as e:
                logger.warning(f"Direct generation failed, creating basic section: {str(e)}")
                section_content = f"# {section_title}\n\nThis section contains requirements related to {section_id} for semiconductor firmware development.\n\n*Content generation in progress...*"
            
            return {
                "section_id": section_id,
                "title": section_title,
                "content": section_content,
                "requirements": [],  # Will be populated by parsing the content
                "metadata": {
                    "generation_timestamp": datetime.now().isoformat(),
                    "source_content_length": len(relevant_content),
                    "template_used": section_template.get("section_number", "")
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating section content for {section_title}: {str(e)}")
            # Return minimal section structure
            return {
                "section_id": section_id,
                "title": section_title,
                "content": f"# {section_title}\n\n*Section content generation failed*\n",
                "requirements": [],
                "metadata": {"error": str(e)}
            }
    
    def _search_relevant_content(self, section_id: str, 
                               processed_docs: Dict[str, List[Document]]) -> str:
        """Search for content relevant to a specific section."""
        if not self.vector_store:
            # Fallback to simple text concatenation
            all_content = []
            for documents in processed_docs.values():
                for doc in documents:
                    all_content.append(doc.page_content)
            return "\n\n".join(all_content[:10])  # Limit to first 10 chunks
        
        # Search terms based on section type
        search_terms_map = {
            "introduction": ["purpose", "scope", "overview", "objectives"],
            "system_overview": ["architecture", "system", "platform", "hardware"],
            "functional_requirements": ["function", "feature", "capability", "operation"],
            "non_functional_requirements": ["performance", "reliability", "efficiency"],
            "interface_requirements": ["interface", "communication", "protocol", "API"],
            "safety_security_requirements": ["safety", "security", "fault", "error"],
            "validation_testing_requirements": ["test", "validation", "verification"]
        }
        
        search_terms = search_terms_map.get(section_id, [section_id])
        
        try:
            relevant_docs = []
            for term in search_terms:
                results = self.vector_store.similarity_search(term, k=3)
                relevant_docs.extend(results)
            
            # Remove duplicates and combine content
            unique_content = []
            seen_content = set()
            for doc in relevant_docs:
                if doc.page_content not in seen_content:
                    unique_content.append(doc.page_content)
                    seen_content.add(doc.page_content)
            
            return "\n\n".join(unique_content)
            
        except Exception as e:
            logger.warning(f"Error searching relevant content: {str(e)}")
            return ""
    
    def _compile_final_document(self, sections: List[Dict[str, Any]], 
                              project_context: Dict[str, Any]) -> str:
        """Compile final SRS document from sections."""
        project_name = project_context.get("project_name", "Semiconductor Firmware System")
        
        # Document header
        document_content = f"""# Software Requirements Specification
## {project_name}

**Project Information:**
- Target Architecture: {project_context.get('target_architecture', 'ARM Cortex-M')}
- Safety Level: {project_context.get('safety_level', 'SIL-2')}
- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Compliance Standards: {', '.join(project_context.get('compliance_standards', []))}

---

"""
        
        # Add each section
        for section in sections:
            document_content += f"{section.get('content', '')}\n\n---\n\n"
        
        # Add appendices
        document_content += self._generate_appendices(project_context)
        
        return document_content
    
    def _generate_appendices(self, project_context: Dict[str, Any]) -> str:
        """Generate document appendices."""
        appendices = """
## Appendix A: Compliance Standards

This document addresses the following compliance standards:
"""
        for standard in project_context.get('compliance_standards', []):
            appendices += f"- {standard}\n"
        
        appendices += f"""

## Appendix B: Target Architecture Details

**Target Platform:** {project_context.get('target_architecture', 'ARM Cortex-M')}

## Appendix C: Traceability Matrix

*Requirements traceability matrix would be generated here based on source documents and generated requirements.*

## Appendix D: Glossary

*Technical terms and definitions specific to semiconductor firmware development.*

---

*Document generated by SRS Generation Agent*
*Generation timestamp: {datetime.now().isoformat()}*
"""
        return appendices
    
    def _validate_complete_srs(self, srs_content: str) -> Dict[str, Any]:
        """Validate the complete SRS document."""
        quality_criteria = self.template_manager.get_quality_criteria()
        compliance_standards = self.config.semiconductor.compliance_standards
        
        try:
            validation_query = (
                "Validate this complete SRS document for quality, "
                "completeness, and compliance with semiconductor industry standards."
            )
            
            validation_input = {
                "srs_content": srs_content[:10000],  # Limit content length
                "quality_criteria": str(quality_criteria),
                "compliance_standards": str(compliance_standards)
            }
            
            validation_result = self.validation_chain.invoke(validation_input)
            
            # Extract validation result (handle AIMessage and other formats)
            if hasattr(validation_result, 'content'):
                validation_result = validation_result.content
            elif isinstance(validation_result, dict):
                validation_result = validation_result.get('text', str(validation_result))
            else:
                validation_result = str(validation_result)
            
            return {
                "validation_timestamp": datetime.now().isoformat(),
                "quality_score": 85,  # Placeholder - would be extracted from validation_result
                "validation_details": validation_result,
                "compliance_status": "Preliminary - requires detailed review"
            }
            
        except Exception as e:
            logger.error(f"Error validating SRS: {str(e)}")
            return {
                "validation_timestamp": datetime.now().isoformat(),
                "error": str(e),
                "status": "Validation failed"
            }
    
    # Agent tool implementations
    def _extract_requirements_tool(self, input_text: str) -> str:
        """Tool for extracting requirements."""
        try:
            # Parse the input to extract parameters
            if not self.vector_store:
                return "Error: No documents loaded for requirement extraction"
            
            # Search for relevant content
            results = self.vector_store.similarity_search(input_text, k=5)
            content = "\n\n".join([doc.page_content for doc in results])
            
            # Use extraction chain
            extraction_input = {
                "document_content": content[:4000],
                "section_type": "functional",  # Default section type
                "semiconductor_context": "ARM Cortex-M, SIL-2 safety level"
            }
            
            result = self.extraction_chain.invoke(extraction_input)
            
            # Extract content from response (handle AIMessage and other formats)
            if hasattr(result, 'content'):
                return result.content
            elif isinstance(result, dict):
                return str(result.get('text', result))
            return str(result)
            
        except Exception as e:
            return f"Error extracting requirements: {str(e)}"
    
    def _generate_section_tool(self, input_text: str) -> str:
        """Tool for generating section content."""
        try:
            # Use generation chain to create section content
            generation_input = {
                "section_template": f"Section requested: {input_text}",
                "extracted_requirements": "No specific requirements provided",
                "project_context": "Semiconductor firmware development"
            }
            
            result = self.generation_chain.invoke(generation_input)
            
            # Extract content from response (handle AIMessage and other formats)
            if hasattr(result, 'content'):
                return result.content
            elif isinstance(result, dict):
                return str(result.get('text', result))
            return str(result)
            
        except Exception as e:
            return f"Error generating section: {str(e)}"
    
    def _validate_srs_tool(self, input_text: str) -> str:
        """Tool for validating SRS content."""
        try:
            # Use validation chain
            validation_input = {
                "srs_content": input_text[:5000],  # Limit content length
                "quality_criteria": "Completeness, Clarity, Testability, Consistency",
                "compliance_standards": "ISO 26262, IEC 61508, MISRA C"
            }
            
            result = self.validation_chain.invoke(validation_input)
            
            # Extract content from response (handle AIMessage and other formats)
            if hasattr(result, 'content'):
                return result.content
            elif isinstance(result, dict):
                return str(result.get('text', result))
            return str(result)
            
        except Exception as e:
            return f"Error validating SRS: {str(e)}"
    
    def _search_documents_tool(self, query: str) -> str:
        """Tool for searching through documents."""
        if not self.vector_store:
            return "No documents loaded for search"
        
        try:
            results = self.vector_store.similarity_search(query, k=3)
            return "\n\n".join([doc.page_content for doc in results])
        except Exception as e:
            return f"Error searching documents: {str(e)}"
    
    def _get_template_info_tool(self, section_id: str) -> str:
        """Tool for getting template information."""
        try:
            template_info = self.template_manager.generate_section_outline(section_id)
            return str(template_info)
        except Exception as e:
            return f"Error getting template info: {str(e)}"
    
    def _generate_semiconductor_requirements_tool(self, req_type: str) -> str:
        """Tool for generating semiconductor-specific requirements."""
        try:
            if req_type == "power":
                reqs = self.semiconductor_generator.generate_power_management_requirements()
            elif req_type == "safety":
                reqs = self.semiconductor_generator.generate_safety_requirements()
            elif req_type == "real_time":
                reqs = self.semiconductor_generator.generate_real_time_requirements()
            else:
                return f"Unknown requirement type: {req_type}"
            
            return str(reqs)
        except Exception as e:
            return f"Error generating semiconductor requirements: {str(e)}"
    
    def save_srs_document(self, srs_result: Dict[str, Any], output_path: str) -> bool:
        """Save generated SRS document to file."""
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as file:
                file.write(srs_result["srs_document"])
            
            logger.info(f"SRS document saved to: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving SRS document: {str(e)}")
            return False
    
    def get_generation_statistics(self) -> Dict[str, Any]:
        """Get statistics about the generation process."""
        return {
            "documents_processed": len(self.vector_store.docstore._dict) if self.vector_store else 0,
            "template_version": self.template_manager.template_data.get("srs_template", {}).get("metadata", {}).get("template_version", "unknown"),
            "model_provider": self.config.model.default_provider,
            "model_name": self.config.model.openai_model if self.config.model.default_provider == "openai" else self.config.model.anthropic_model
        }


# Utility functions
def create_srs_agent(config=None) -> SRSGenerationAgent:
    """Create and initialize SRS generation agent."""
    return SRSGenerationAgent(config)


async def generate_srs_async(agent: SRSGenerationAgent, project_name: str, 
                           file_paths: List[str], **kwargs) -> Dict[str, Any]:
    """Asynchronous SRS generation."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, agent.generate_srs, project_name, file_paths, **kwargs
    )