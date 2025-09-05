"""
SRS (System Requirements Specification) Generation Agent using LangGraph
==================================================================

This agent analyzes specification documents and generates comprehensive SRS documents
following industry standards. It uses RAG patterns for spec analysis and multi-step
workflow orchestration with LangGraph.

Based on 99_RAG_Note course patterns for LangChain and LangGraph implementation.
"""

import os
import logging
from typing import TypedDict, List, Dict, Any, Annotated
from operator import add

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import BaseRetriever

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SRSState(TypedDict):
    """State definition for SRS generation workflow"""
    # Input documents and analysis
    spec_documents: List[str]  # Input specification file paths
    raw_documents: List[Document]  # Loaded documents
    processed_documents: List[Document]  # Split and processed documents
    
    # RAG components
    vectorstore: Any  # Vector store for document retrieval
    retriever: BaseRetriever  # Document retriever
    
    # Analysis results
    requirements_analysis: Dict[str, Any]  # Extracted requirements analysis
    functional_requirements: List[str]  # Functional requirements
    non_functional_requirements: List[str]  # Non-functional requirements
    system_interfaces: List[str]  # System interface requirements
    data_requirements: List[str]  # Data requirements
    performance_requirements: List[str]  # Performance requirements
    
    # SRS document generation
    srs_sections: Dict[str, str]  # Generated SRS sections
    final_srs_document: str  # Complete SRS document
    
    # Workflow control
    current_step: str  # Current processing step
    errors: Annotated[List[str], add]  # Error messages
    metadata: Dict[str, Any]  # Additional metadata


class SRSGenerationAgent:
    """LangGraph agent for generating SRS documents from specification materials"""
    
    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.1):
        """Initialize the SRS generation agent"""
        self.model_name = model_name
        self.temperature = temperature
        
        # Initialize LLM based on model type
        if "claude" in model_name.lower():
            self.llm = ChatAnthropic(model_name=model_name, temperature=temperature)
        else:
            self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
            
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        # Initialize workflow
        self.workflow = self._create_workflow()
        self.memory = MemorySaver()
        self.app = self.workflow.compile(checkpointer=self.memory)
        
        logger.info(f"SRS Generation Agent initialized with model: {model_name}")
    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow for SRS generation"""
        workflow = StateGraph(SRSState)
        
        # Add nodes
        workflow.add_node("load_documents", self._load_documents)
        workflow.add_node("process_documents", self._process_documents)
        workflow.add_node("create_vectorstore", self._create_vectorstore)
        workflow.add_node("analyze_requirements", self._analyze_requirements)
        workflow.add_node("extract_functional_requirements", self._extract_functional_requirements)
        workflow.add_node("extract_non_functional_requirements", self._extract_non_functional_requirements)
        workflow.add_node("extract_system_interfaces", self._extract_system_interfaces)
        workflow.add_node("extract_data_requirements", self._extract_data_requirements)
        workflow.add_node("extract_performance_requirements", self._extract_performance_requirements)
        workflow.add_node("generate_srs_sections", self._generate_srs_sections)
        workflow.add_node("compile_final_srs", self._compile_final_srs)
        
        # Define edges
        workflow.set_entry_point("load_documents")
        workflow.add_edge("load_documents", "process_documents")
        workflow.add_edge("process_documents", "create_vectorstore")
        workflow.add_edge("create_vectorstore", "analyze_requirements")
        workflow.add_edge("analyze_requirements", "extract_functional_requirements")
        workflow.add_edge("extract_functional_requirements", "extract_non_functional_requirements")
        workflow.add_edge("extract_non_functional_requirements", "extract_system_interfaces")
        workflow.add_edge("extract_system_interfaces", "extract_data_requirements")
        workflow.add_edge("extract_data_requirements", "extract_performance_requirements")
        workflow.add_edge("extract_performance_requirements", "generate_srs_sections")
        workflow.add_edge("generate_srs_sections", "compile_final_srs")
        workflow.add_edge("compile_final_srs", END)
        
        return workflow
    
    def _load_documents(self, state: SRSState) -> SRSState:
        """Load specification documents from file paths"""
        logger.info("Loading specification documents...")
        
        try:
            documents = []
            for file_path in state["spec_documents"]:
                if not os.path.exists(file_path):
                    error_msg = f"File not found: {file_path}"
                    logger.error(error_msg)
                    state["errors"].append(error_msg)
                    continue
                
                # Determine loader based on file extension
                if file_path.lower().endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                else:
                    loader = TextLoader(file_path, encoding='utf-8')
                
                docs = loader.load()
                documents.extend(docs)
                logger.info(f"Loaded {len(docs)} documents from {file_path}")
            
            state["raw_documents"] = documents
            state["current_step"] = "documents_loaded"
            logger.info(f"Successfully loaded {len(documents)} total documents")
            
        except Exception as e:
            error_msg = f"Error loading documents: {str(e)}"
            logger.error(error_msg)
            state["errors"].append(error_msg)
        
        return state
    
    def _process_documents(self, state: SRSState) -> SRSState:
        """Process and split documents for better analysis"""
        logger.info("Processing and splitting documents...")
        
        try:
            if not state["raw_documents"]:
                error_msg = "No documents to process"
                logger.error(error_msg)
                state["errors"].append(error_msg)
                return state
            
            # Split documents
            split_docs = self.text_splitter.split_documents(state["raw_documents"])
            
            # Clean and preprocess text
            processed_docs = []
            for doc in split_docs:
                # Clean text content
                cleaned_content = doc.page_content.strip()
                if len(cleaned_content) > 50:  # Filter out very short chunks
                    doc.page_content = cleaned_content
                    processed_docs.append(doc)
            
            state["processed_documents"] = processed_docs
            state["current_step"] = "documents_processed"
            logger.info(f"Processed {len(processed_docs)} document chunks")
            
        except Exception as e:
            error_msg = f"Error processing documents: {str(e)}"
            logger.error(error_msg)
            state["errors"].append(error_msg)
        
        return state
    
    def _create_vectorstore(self, state: SRSState) -> SRSState:
        """Create vector store and retriever for RAG"""
        logger.info("Creating vector store and retriever...")
        
        try:
            if not state["processed_documents"]:
                error_msg = "No processed documents available for vectorstore"
                logger.error(error_msg)
                state["errors"].append(error_msg)
                return state
            
            # Create vector store
            vectorstore = FAISS.from_documents(
                documents=state["processed_documents"],
                embedding=self.embeddings
            )
            
            # Create retriever with MMR search
            retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 8,
                    "fetch_k": 20,
                    "lambda_mult": 0.5
                }
            )
            
            state["vectorstore"] = vectorstore
            state["retriever"] = retriever
            state["current_step"] = "vectorstore_created"
            logger.info("Vector store and retriever created successfully")
            
        except Exception as e:
            error_msg = f"Error creating vectorstore: {str(e)}"
            logger.error(error_msg)
            state["errors"].append(error_msg)
        
        return state
    
    def _analyze_requirements(self, state: SRSState) -> SRSState:
        """Perform high-level requirements analysis"""
        logger.info("Performing requirements analysis...")
        
        try:
            # Create analysis prompt
            analysis_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert systems analyst specializing in requirements engineering.
                Analyze the provided specification documents and extract key information for SRS generation.
                
                Focus on identifying:
                1. Project scope and objectives
                2. Stakeholders and users
                3. System overview and architecture
                4. Key business processes
                5. Integration requirements
                6. Constraints and assumptions
                
                Provide a structured analysis in JSON format."""),
                ("user", """Analyze these specification documents and provide a comprehensive requirements analysis:
                
                {context}
                
                Return your analysis as a JSON object with the following structure:
                {{
                    "project_scope": "Description of project scope",
                    "objectives": ["List of main objectives"],
                    "stakeholders": ["List of key stakeholders"],
                    "system_overview": "High-level system description",
                    "business_processes": ["List of key business processes"],
                    "integration_needs": ["List of integration requirements"],
                    "constraints": ["List of constraints"],
                    "assumptions": ["List of assumptions"]
                }}""")
            ])
            
            # Get relevant context using retriever
            context_docs = state["retriever"].get_relevant_documents(
                "system requirements specification project scope objectives stakeholders"
            )
            context = "\n\n".join([doc.page_content for doc in context_docs])
            
            # Generate analysis
            chain = analysis_prompt | self.llm | JsonOutputParser()
            analysis = chain.invoke({"context": context})
            
            state["requirements_analysis"] = analysis
            state["current_step"] = "requirements_analyzed"
            logger.info("Requirements analysis completed")
            
        except Exception as e:
            error_msg = f"Error in requirements analysis: {str(e)}"
            logger.error(error_msg)
            state["errors"].append(error_msg)
        
        return state
    
    def _extract_functional_requirements(self, state: SRSState) -> SRSState:
        """Extract functional requirements from specifications"""
        logger.info("Extracting functional requirements...")
        
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert business analyst specializing in functional requirements extraction.
                Extract detailed functional requirements from the specification documents.
                
                Functional requirements describe what the system must do - specific functions, features, and capabilities.
                Each requirement should be:
                - Clear and unambiguous
                - Testable and verifiable
                - Traceable to business needs
                - Properly formatted with unique identifiers"""),
                ("user", """Extract functional requirements from these documents:
                
                {context}
                
                Return a list of functional requirements, each formatted as:
                "FR-XXX: [Clear requirement statement]"
                
                Focus on:
                - User interface functions
                - Data processing capabilities
                - Business logic operations
                - System behaviors
                - Integration functions""")
            ])
            
            # Get relevant context
            context_docs = state["retriever"].get_relevant_documents(
                "functional requirements features capabilities user interface business logic operations"
            )
            context = "\n\n".join([doc.page_content for doc in context_docs])
            
            # Generate requirements
            chain = prompt | self.llm | StrOutputParser()
            response = chain.invoke({"context": context})
            
            # Parse requirements into list
            requirements = [req.strip() for req in response.split('\n') if req.strip() and 'FR-' in req]
            
            state["functional_requirements"] = requirements
            state["current_step"] = "functional_requirements_extracted"
            logger.info(f"Extracted {len(requirements)} functional requirements")
            
        except Exception as e:
            error_msg = f"Error extracting functional requirements: {str(e)}"
            logger.error(error_msg)
            state["errors"].append(error_msg)
        
        return state
    
    def _extract_non_functional_requirements(self, state: SRSState) -> SRSState:
        """Extract non-functional requirements"""
        logger.info("Extracting non-functional requirements...")
        
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert systems architect specializing in non-functional requirements.
                Extract non-functional requirements that define system quality attributes and constraints.
                
                Non-functional requirements include:
                - Performance (speed, throughput, response time)
                - Scalability and capacity
                - Security and privacy
                - Reliability and availability
                - Usability and accessibility
                - Maintainability and supportability
                - Compliance and regulatory requirements"""),
                ("user", """Extract non-functional requirements from these documents:
                
                {context}
                
                Return a list of non-functional requirements, each formatted as:
                "NFR-XXX: [Clear requirement statement with measurable criteria where possible]"
                
                Include specific metrics and acceptance criteria when available.""")
            ])
            
            # Get relevant context
            context_docs = state["retriever"].get_relevant_documents(
                "non-functional requirements performance scalability security reliability usability compliance"
            )
            context = "\n\n".join([doc.page_content for doc in context_docs])
            
            # Generate requirements
            chain = prompt | self.llm | StrOutputParser()
            response = chain.invoke({"context": context})
            
            # Parse requirements into list
            requirements = [req.strip() for req in response.split('\n') if req.strip() and 'NFR-' in req]
            
            state["non_functional_requirements"] = requirements
            state["current_step"] = "non_functional_requirements_extracted"
            logger.info(f"Extracted {len(requirements)} non-functional requirements")
            
        except Exception as e:
            error_msg = f"Error extracting non-functional requirements: {str(e)}"
            logger.error(error_msg)
            state["errors"].append(error_msg)
        
        return state
    
    def _extract_system_interfaces(self, state: SRSState) -> SRSState:
        """Extract system interface requirements"""
        logger.info("Extracting system interface requirements...")
        
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert integration architect specializing in system interfaces.
                Extract system interface requirements that define how the system interacts with:
                - External systems and APIs
                - User interfaces and presentation layers
                - Hardware interfaces
                - Database and data storage interfaces
                - Network and communication interfaces"""),
                ("user", """Extract system interface requirements from these documents:
                
                {context}
                
                Return a list of interface requirements, each formatted as:
                "SI-XXX: [Clear interface requirement with protocols, formats, and specifications]"
                
                Include technical details about data formats, protocols, and integration methods.""")
            ])
            
            # Get relevant context
            context_docs = state["retriever"].get_relevant_documents(
                "system interfaces API integration external systems database network communication protocols"
            )
            context = "\n\n".join([doc.page_content for doc in context_docs])
            
            # Generate requirements
            chain = prompt | self.llm | StrOutputParser()
            response = chain.invoke({"context": context})
            
            # Parse requirements into list
            requirements = [req.strip() for req in response.split('\n') if req.strip() and 'SI-' in req]
            
            state["system_interfaces"] = requirements
            state["current_step"] = "system_interfaces_extracted"
            logger.info(f"Extracted {len(requirements)} system interface requirements")
            
        except Exception as e:
            error_msg = f"Error extracting system interfaces: {str(e)}"
            logger.error(error_msg)
            state["errors"].append(error_msg)
        
        return state
    
    def _extract_data_requirements(self, state: SRSState) -> SRSState:
        """Extract data requirements"""
        logger.info("Extracting data requirements...")
        
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert data architect specializing in data requirements.
                Extract data requirements that define:
                - Data entities and their relationships
                - Data storage and retention requirements
                - Data quality and validation rules
                - Data security and privacy requirements
                - Data migration and conversion needs
                - Backup and recovery requirements"""),
                ("user", """Extract data requirements from these documents:
                
                {context}
                
                Return a list of data requirements, each formatted as:
                "DR-XXX: [Clear data requirement with specifications and constraints]"
                
                Include details about data types, formats, volumes, and quality criteria.""")
            ])
            
            # Get relevant context
            context_docs = state["retriever"].get_relevant_documents(
                "data requirements database storage entities relationships data quality security privacy backup"
            )
            context = "\n\n".join([doc.page_content for doc in context_docs])
            
            # Generate requirements
            chain = prompt | self.llm | StrOutputParser()
            response = chain.invoke({"context": context})
            
            # Parse requirements into list
            requirements = [req.strip() for req in response.split('\n') if req.strip() and 'DR-' in req]
            
            state["data_requirements"] = requirements
            state["current_step"] = "data_requirements_extracted"
            logger.info(f"Extracted {len(requirements)} data requirements")
            
        except Exception as e:
            error_msg = f"Error extracting data requirements: {str(e)}"
            logger.error(error_msg)
            state["errors"].append(error_msg)
        
        return state
    
    def _extract_performance_requirements(self, state: SRSState) -> SRSState:
        """Extract performance requirements"""
        logger.info("Extracting performance requirements...")
        
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert performance engineer specializing in system performance requirements.
                Extract performance requirements that define:
                - Response time and latency requirements
                - Throughput and transaction volume requirements
                - Concurrent user and load requirements
                - Resource utilization limits
                - Scalability requirements
                - Availability and uptime requirements"""),
                ("user", """Extract performance requirements from these documents:
                
                {context}
                
                Return a list of performance requirements, each formatted as:
                "PR-XXX: [Clear performance requirement with measurable metrics and acceptance criteria]"
                
                Include specific numeric targets, measurement methods, and performance conditions.""")
            ])
            
            # Get relevant context
            context_docs = state["retriever"].get_relevant_documents(
                "performance requirements response time throughput scalability load users availability uptime"
            )
            context = "\n\n".join([doc.page_content for doc in context_docs])
            
            # Generate requirements
            chain = prompt | self.llm | StrOutputParser()
            response = chain.invoke({"context": context})
            
            # Parse requirements into list
            requirements = [req.strip() for req in response.split('\n') if req.strip() and 'PR-' in req]
            
            state["performance_requirements"] = requirements
            state["current_step"] = "performance_requirements_extracted"
            logger.info(f"Extracted {len(requirements)} performance requirements")
            
        except Exception as e:
            error_msg = f"Error extracting performance requirements: {str(e)}"
            logger.error(error_msg)
            state["errors"].append(error_msg)
        
        return state
    
    def _generate_srs_sections(self, state: SRSState) -> SRSState:
        """Generate individual SRS document sections"""
        logger.info("Generating SRS document sections...")
        
        try:
            sections = {}
            analysis = state.get("requirements_analysis", {})
            
            # 1. Introduction Section
            intro_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an expert technical writer specializing in SRS documents."),
                ("user", """Create a comprehensive Introduction section for an SRS document based on this analysis:
                
                Project Scope: {project_scope}
                Objectives: {objectives}
                
                The Introduction should include:
                1.1 Purpose
                1.2 Scope
                1.3 Definitions, Acronyms, and Abbreviations
                1.4 References
                1.5 Overview
                
                Write in professional technical documentation style.""")
            ])
            
            chain = intro_prompt | self.llm | StrOutputParser()
            sections["introduction"] = chain.invoke({
                "project_scope": analysis.get("project_scope", ""),
                "objectives": ", ".join(analysis.get("objectives", []))
            })
            
            # 2. Overall Description Section
            desc_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an expert technical writer specializing in SRS documents."),
                ("user", """Create a comprehensive Overall Description section based on this analysis:
                
                System Overview: {system_overview}
                Stakeholders: {stakeholders}
                Business Processes: {business_processes}
                Constraints: {constraints}
                Assumptions: {assumptions}
                
                The Overall Description should include:
                2.1 Product Perspective
                2.2 Product Functions
                2.3 User Classes and Characteristics
                2.4 Operating Environment
                2.5 Design and Implementation Constraints
                2.6 Assumptions and Dependencies""")
            ])
            
            desc_chain = desc_prompt | self.llm | StrOutputParser()
            sections["overall_description"] = desc_chain.invoke({
                "system_overview": analysis.get("system_overview", ""),
                "stakeholders": ", ".join(analysis.get("stakeholders", [])),
                "business_processes": ", ".join(analysis.get("business_processes", [])),
                "constraints": ", ".join(analysis.get("constraints", [])),
                "assumptions": ", ".join(analysis.get("assumptions", []))
            })
            
            # 3. Functional Requirements Section
            func_req_content = "3. Functional Requirements\n\n"
            for req in state.get("functional_requirements", []):
                func_req_content += f"{req}\n"
            sections["functional_requirements"] = func_req_content
            
            # 4. Non-Functional Requirements Section
            nonfunc_req_content = "4. Non-Functional Requirements\n\n"
            for req in state.get("non_functional_requirements", []):
                nonfunc_req_content += f"{req}\n"
            sections["non_functional_requirements"] = nonfunc_req_content
            
            # 5. System Interfaces Section
            interface_content = "5. System Interfaces\n\n"
            for req in state.get("system_interfaces", []):
                interface_content += f"{req}\n"
            sections["system_interfaces"] = interface_content
            
            # 6. Data Requirements Section
            data_content = "6. Data Requirements\n\n"
            for req in state.get("data_requirements", []):
                data_content += f"{req}\n"
            sections["data_requirements"] = data_content
            
            # 7. Performance Requirements Section
            perf_content = "7. Performance Requirements\n\n"
            for req in state.get("performance_requirements", []):
                perf_content += f"{req}\n"
            sections["performance_requirements"] = perf_content
            
            state["srs_sections"] = sections
            state["current_step"] = "srs_sections_generated"
            logger.info("SRS sections generated successfully")
            
        except Exception as e:
            error_msg = f"Error generating SRS sections: {str(e)}"
            logger.error(error_msg)
            state["errors"].append(error_msg)
        
        return state
    
    def _compile_final_srs(self, state: SRSState) -> SRSState:
        """Compile the final SRS document"""
        logger.info("Compiling final SRS document...")
        
        try:
            sections = state.get("srs_sections", {})
            analysis = state.get("requirements_analysis", {})
            
            # Create document header
            srs_document = f"""
# System Requirements Specification (SRS)

**Project:** {analysis.get('project_scope', 'System Requirements Specification')}
**Version:** 1.0
**Date:** {state.get('metadata', {}).get('generation_date', 'Generated by SRS Agent')}

---

## Table of Contents

1. Introduction
2. Overall Description
3. Functional Requirements
4. Non-Functional Requirements
5. System Interfaces
6. Data Requirements
7. Performance Requirements

---

## 1. {sections.get('introduction', 'Introduction section not available')}

---

## 2. {sections.get('overall_description', 'Overall description section not available')}

---

## {sections.get('functional_requirements', '3. Functional Requirements section not available')}

---

## {sections.get('non_functional_requirements', '4. Non-Functional Requirements section not available')}

---

## {sections.get('system_interfaces', '5. System Interfaces section not available')}

---

## {sections.get('data_requirements', '6. Data Requirements section not available')}

---

## {sections.get('performance_requirements', '7. Performance Requirements section not available')}

---

## Document Information

- **Generated by:** SRS Generation Agent
- **Source Documents:** {', '.join(state.get('spec_documents', []))}
- **Total Requirements:** {len(state.get('functional_requirements', [])) + len(state.get('non_functional_requirements', [])) + len(state.get('system_interfaces', [])) + len(state.get('data_requirements', [])) + len(state.get('performance_requirements', []))}

"""
            
            state["final_srs_document"] = srs_document
            state["current_step"] = "srs_completed"
            logger.info("Final SRS document compiled successfully")
            
        except Exception as e:
            error_msg = f"Error compiling final SRS: {str(e)}"
            logger.error(error_msg)
            state["errors"].append(error_msg)
        
        return state
    
    def generate_srs(self, spec_files: List[str], thread_id: str = "srs_generation") -> Dict[str, Any]:
        """
        Generate SRS document from specification files
        
        Args:
            spec_files: List of specification file paths
            thread_id: Unique identifier for this generation session
            
        Returns:
            Dictionary containing the generated SRS and metadata
        """
        logger.info(f"Starting SRS generation for files: {spec_files}")
        
        # Initialize state
        initial_state = {
            "spec_documents": spec_files,
            "raw_documents": [],
            "processed_documents": [],
            "vectorstore": None,
            "retriever": None,
            "requirements_analysis": {},
            "functional_requirements": [],
            "non_functional_requirements": [],
            "system_interfaces": [],
            "data_requirements": [],
            "performance_requirements": [],
            "srs_sections": {},
            "final_srs_document": "",
            "current_step": "initialized",
            "errors": [],
            "metadata": {
                "generation_date": "2025-01-09",
                "thread_id": thread_id,
                "model_name": self.model_name
            }
        }
        
        # Execute workflow
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            result = self.app.invoke(initial_state, config)
            
            logger.info("SRS generation completed successfully")
            return {
                "success": True,
                "srs_document": result["final_srs_document"],
                "functional_requirements": result["functional_requirements"],
                "non_functional_requirements": result["non_functional_requirements"],
                "system_interfaces": result["system_interfaces"],
                "data_requirements": result["data_requirements"],
                "performance_requirements": result["performance_requirements"],
                "requirements_analysis": result["requirements_analysis"],
                "errors": result["errors"],
                "metadata": result["metadata"]
            }
            
        except Exception as e:
            error_msg = f"Error in SRS generation workflow: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "srs_document": "",
                "functional_requirements": [],
                "non_functional_requirements": [],
                "system_interfaces": [],
                "data_requirements": [],
                "performance_requirements": [],
                "requirements_analysis": {},
                "errors": [error_msg],
                "metadata": initial_state["metadata"]
            }
    
    def save_srs_document(self, srs_document: str, output_path: str) -> bool:
        """Save the generated SRS document to a file"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(srs_document)
            logger.info(f"SRS document saved to: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving SRS document: {str(e)}")
            return False


if __name__ == "__main__":
    # Example usage
    agent = SRSGenerationAgent(model_name="gpt-4o-mini")
    
    # Example with sample specification files
    spec_files = [
        # Add your specification file paths here
        # "path/to/spec1.txt",
        # "path/to/spec2.pdf"
    ]
    
    if spec_files:
        result = agent.generate_srs(spec_files)
        
        if result["success"]:
            print("SRS Generation Successful!")
            print(f"Generated {len(result['functional_requirements'])} functional requirements")
            print(f"Generated {len(result['non_functional_requirements'])} non-functional requirements")
            
            # Save to file
            output_path = "generated_srs_document.md"
            agent.save_srs_document(result["srs_document"], output_path)
            print(f"SRS document saved to: {output_path}")
        else:
            print(f"SRS Generation Failed: {result['error']}")
    else:
        print("Please provide specification file paths to generate SRS document")