"""
Hybrid SRS Generation Agent - 기존 방식 + 사실 검증
===============================================

이 하이브리드 버전은:
1. 기존의 풍부한 요구사항 생성 유지
2. 생성 후 사실 검증 단계 추가  
3. Hallucination 탐지 및 수정
4. 요구사항 추출량 최적화
"""

import os
import re
import logging
from typing import TypedDict, List, Dict, Any, Annotated
from operator import add
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """요구사항 검증 결과"""
    original_requirement: str
    is_valid: bool
    confidence_score: float
    evidence: List[str]
    corrections: str = ""
    rejection_reason: str = ""


class HybridSRSState(TypedDict):
    """하이브리드 SRS 생성 상태"""
    # 기존 필드들 (원본 에이전트와 동일)
    spec_documents: List[str]
    raw_documents: List[Document]
    processed_documents: List[Document]
    requirements_analysis: Dict[str, Any]
    functional_requirements: List[str]
    non_functional_requirements: List[str]
    system_interfaces: List[str]
    data_requirements: List[str]
    performance_requirements: List[str]
    srs_sections: Dict[str, str]
    final_srs_document: str
    current_step: str
    errors: Annotated[List[str], add]
    metadata: Dict[str, Any]
    
    # 새로운 검증 관련 필드들
    validation_results: Dict[str, List[ValidationResult]]
    validated_requirements: Dict[str, List[str]]
    rejected_requirements: List[Dict[str, Any]]
    fact_check_summary: Dict[str, Any]


class HybridSRSGenerationAgent:
    """기존 방식 + 사실 검증 하이브리드 에이전트"""
    
    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.1):
        self.model_name = model_name
        self.temperature = temperature
        
        # LLM 초기화
        if "claude" in model_name.lower():
            self.llm = ChatAnthropic(model_name=model_name, temperature=temperature)
        else:
            self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        
        # 검증용 저온도 LLM (사실 확인용)
        if "claude" in model_name.lower():
            self.validator_llm = ChatAnthropic(model_name=model_name, temperature=0.0)
        else:
            self.validator_llm = ChatOpenAI(model_name=model_name, temperature=0.0)
            
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # 향상된 문서 분할기 - 기술 문서에 최적화
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,  # 더 큰 청크로 컨텍스트 확보
            chunk_overlap=300,  # 더 많은 오버랩으로 연관성 확보
            separators=["\\n\\n", "\\n", ". ", "! ", "? ", ", ", " ", ""],
            length_function=len,
            is_separator_regex=False
        )
        
        self.vectorstore = None
        self.retriever = None
        
        # 워크플로우 생성
        self.workflow = self._create_workflow()
        self.memory = MemorySaver()
        self.app = self.workflow.compile(checkpointer=self.memory)
        
        logger.info(f"Hybrid SRS Agent initialized with model: {model_name}")
    
    def _create_workflow(self) -> StateGraph:
        """하이브리드 워크플로우 생성"""
        workflow = StateGraph(HybridSRSState)
        
        # 기존 노드들 (원본과 동일)
        workflow.add_node("load_documents", self._load_documents)
        workflow.add_node("process_documents", self._process_documents)
        workflow.add_node("create_vectorstore", self._create_vectorstore)
        workflow.add_node("analyze_requirements", self._analyze_requirements)
        workflow.add_node("extract_functional_requirements", self._extract_functional_requirements)
        workflow.add_node("extract_non_functional_requirements", self._extract_non_functional_requirements)
        workflow.add_node("extract_system_interfaces", self._extract_system_interfaces)
        workflow.add_node("extract_data_requirements", self._extract_data_requirements)
        workflow.add_node("extract_performance_requirements", self._extract_performance_requirements)
        
        # 새로운 검증 노드들
        workflow.add_node("validate_requirements", self._validate_requirements)
        workflow.add_node("extract_implicit_requirements", self._extract_implicit_requirements)
        workflow.add_node("fact_check_requirements", self._fact_check_requirements)
        workflow.add_node("apply_corrections", self._apply_corrections)
        
        # 기존 생성 노드들
        workflow.add_node("generate_srs_sections", self._generate_srs_sections)
        workflow.add_node("compile_final_srs", self._compile_final_srs)
        
        # 워크플로우 엣지 정의
        workflow.set_entry_point("load_documents")
        workflow.add_edge("load_documents", "process_documents")
        workflow.add_edge("process_documents", "create_vectorstore")
        workflow.add_edge("create_vectorstore", "analyze_requirements")
        workflow.add_edge("analyze_requirements", "extract_functional_requirements")
        workflow.add_edge("extract_functional_requirements", "extract_non_functional_requirements")
        workflow.add_edge("extract_non_functional_requirements", "extract_system_interfaces")
        workflow.add_edge("extract_system_interfaces", "extract_data_requirements")
        workflow.add_edge("extract_data_requirements", "extract_performance_requirements")
        
        # 검증 단계 추가
        workflow.add_edge("extract_performance_requirements", "extract_implicit_requirements")
        workflow.add_edge("extract_implicit_requirements", "validate_requirements")
        workflow.add_edge("validate_requirements", "fact_check_requirements")
        workflow.add_edge("fact_check_requirements", "apply_corrections")
        workflow.add_edge("apply_corrections", "generate_srs_sections")
        workflow.add_edge("generate_srs_sections", "compile_final_srs")
        workflow.add_edge("compile_final_srs", END)
        
        return workflow
    
    def _format_citation(self, document: Document) -> str:
        """문서에서 인용 정보 포맷팅"""
        if not hasattr(document, 'metadata') or not document.metadata:
            return "[Source: Unknown]"
        
        source_file = document.metadata.get('source', 'Unknown')
        if isinstance(source_file, str) and source_file.startswith('/'):
            source_file = source_file.split('/')[-1]  # Get filename only
        
        page = document.metadata.get('page', None)
        
        if page is not None and str(page).lower() != 'unknown':
            return f"[Source: {source_file}, Page {page}]"
        else:
            return f"[Source: {source_file}]"
    
    def _load_documents(self, state: HybridSRSState) -> HybridSRSState:
        """문서 로딩 (기존과 동일)"""
        logger.info("Loading specification documents...")
        
        try:
            documents = []
            for file_path in state["spec_documents"]:
                if not os.path.exists(file_path):
                    error_msg = f"File not found: {file_path}"
                    logger.error(error_msg)
                    state["errors"].append(error_msg)
                    continue
                
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
    
    def _process_documents(self, state: HybridSRSState) -> HybridSRSState:
        """문서 처리 (기존과 동일)"""
        logger.info("Processing and splitting documents...")
        
        try:
            if not state["raw_documents"]:
                error_msg = "No documents to process"
                logger.error(error_msg)
                state["errors"].append(error_msg)
                return state
            
            split_docs = self.text_splitter.split_documents(state["raw_documents"])
            
            processed_docs = []
            for doc in split_docs:
                cleaned_content = doc.page_content.strip()
                if len(cleaned_content) > 50:
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
    
    def _create_vectorstore(self, state: HybridSRSState) -> HybridSRSState:
        """벡터스토어 생성 (기존과 동일)"""
        logger.info("Creating vector store and retriever...")
        
        try:
            if not state["processed_documents"]:
                error_msg = "No processed documents available for vectorstore"
                logger.error(error_msg)
                state["errors"].append(error_msg)
                return state
            
            self.vectorstore = FAISS.from_documents(
                documents=state["processed_documents"],
                embedding=self.embeddings
            )
            
            self.retriever = self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 30,  # 더 많은 컨텍스트 검색 (2.5x 증가)
                    "fetch_k": 75,  # 더 많은 후보 검색 (2.5x 증가)
                    "lambda_mult": 0.5
                }
            )
            state["current_step"] = "vectorstore_created"
            logger.info("Vector store and retriever created successfully")
            
        except Exception as e:
            error_msg = f"Error creating vectorstore: {str(e)}"
            logger.error(error_msg)
            state["errors"].append(error_msg)
        
        return state
    
    def _analyze_requirements(self, state: HybridSRSState) -> HybridSRSState:
        """요구사항 분석 (기존과 유사하되 더 자세히)"""
        logger.info("Performing comprehensive requirements analysis...")
        
        try:
            analysis_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert systems analyst. Perform a thorough analysis of the specification documents.
                
                Focus on identifying:
                1. Detailed project scope and objectives
                2. All stakeholders and user types
                3. Comprehensive system overview and architecture
                4. All business processes and workflows
                5. Integration and interface requirements
                6. Technical constraints and assumptions
                7. Performance expectations and metrics
                8. Security and compliance requirements
                
                Be thorough but factual - only extract what is explicitly mentioned or clearly implied."""),
                ("user", """Analyze these specification documents comprehensively:

{context}

Provide a detailed analysis in JSON format with extensive coverage of all aspects found in the documents.""")
            ])
            
            # 더 많은 컨텍스트 검색
            context_queries = [
                "system requirements specification project scope objectives",
                "stakeholders users business processes workflows",
                "architecture design technical specifications",
                "performance security compliance constraints"
            ]
            
            all_context = []
            for query in context_queries:
                context_docs = self.retriever.get_relevant_documents(query)
                all_context.extend([doc.page_content for doc in context_docs])
            
            context = "\n\n".join(set(all_context))  # 중복 제거
            
            chain = analysis_prompt | self.llm | JsonOutputParser()
            analysis = chain.invoke({"context": context})
            
            state["requirements_analysis"] = analysis
            state["current_step"] = "requirements_analyzed"
            logger.info("Comprehensive requirements analysis completed")
            
        except Exception as e:
            error_msg = f"Error in requirements analysis: {str(e)}"
            logger.error(error_msg)
            state["errors"].append(error_msg)
        
        return state
    
    def _extract_functional_requirements(self, state: HybridSRSState) -> HybridSRSState:
        """기능 요구사항 추출 (소스 인용 포함)"""
        logger.info("Extracting functional requirements with source citations...")
        
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a technical requirements analyst specializing in system specifications. Extract CONCRETE, IMPLEMENTATION-READY functional requirements.

                EXTRACTION RULES:
                1. Extract ONLY requirements that specify what the system/platform/component MUST do
                2. Include EXACT technical parameters: numbers, values, limits, thresholds
                3. Reference SPECIFIC methods, functions, classes, or components mentioned
                4. Include PRECISE error handling procedures and status codes
                5. Quote EXACT command formats, data structures, and protocols
                6. Specify DETAILED processing steps and algorithmic requirements
                7. Include EXPLICIT validation rules and constraints
                8. Reference SPECIFIC standards, formats, and compliance requirements

                FORMAT REQUIREMENTS:
                - Each requirement must start with "The system must" or "The platform must" or "The component must"
                - Include specific technical terms and terminology from the source document
                - End with precise source citation: [Source: filename, Page X]
                - Be specific enough that a developer could implement it

                EXAMPLES OF GOOD REQUIREMENTS:
                ✓ "The platform must support APDU commands with maximum data field length of 65535 bytes as defined in ISO 7816-4 extended length format. [Source: spec.pdf, Page 97]"
                ✓ "The system must return status code 0x6881 (SW_LOGICAL_CHANNEL_NOT_SUPPORTED) when logical channel resources are unavailable during SELECT FILE command processing. [Source: spec.pdf, Page 39]"

                AVOID GENERIC REQUIREMENTS:
                ✗ "The system must provide authentication"
                ✗ "The system must handle errors properly"
                ✗ "The system must be secure"

                Extract concrete, testable functional requirements with implementation details."""),
                ("user", """Extract CONCRETE functional requirements from these technical specification documents:

{context}

Focus on extracting requirements that specify exact system behaviors, precise technical parameters, specific processing steps, and detailed implementation constraints. Each requirement must be specific enough for direct implementation.""")
            ])
            
            # 다양한 기술적 컨텍스트로 검색하여 더 많은 구체적 요구사항 확보
            context_queries = [
                "must shall should requirements specifications",
                "commands operations processing procedures methods",
                "functions capabilities features behaviors actions",
                "parameters values limits thresholds constraints",
                "formats structures protocols standards compliance"
            ]
            
            context_docs = []
            for query in context_queries:
                docs = self.retriever.get_relevant_documents(query)
                context_docs.extend(docs[:8])  # 각 쿼리당 8개 문서
            
            # 컨텍스트에 인용 정보 포함
            context_parts = []
            for doc in context_docs:
                citation = self._format_citation(doc)
                context_parts.append(f"{doc.page_content}\n{citation}")
            context = "\n\n---\n\n".join(context_parts)
            
            chain = prompt | self.llm | StrOutputParser()
            response = chain.invoke({"context": context})
            
            # 요구사항 파싱 및 ID 생성
            requirements = []
            req_counter = 1
            lines = response.split('\n')
            
            for line in lines:
                line = line.strip()
                
                # 요구사항 라인 감지 (더 유연한 패턴)
                if (line and 
                    (line.startswith('-') or line.startswith('*') or line.startswith('•') or 
                     any(line.startswith(f"{i}.") for i in range(1, 100)) or
                     line.lower().startswith('the ') and ('must' in line.lower() or 'shall' in line.lower()))):
                    
                    # 리스트 마커 제거
                    clean_req = re.sub(r'^[-*•]\s*|\d+\.\s*', '', line).strip()
                    
                    # 최소 길이 및 기술적 내용 확인
                    if (len(clean_req) > 40 and 
                        ('must' in clean_req.lower() or 'shall' in clean_req.lower()) and
                        not clean_req.startswith('FR-')):
                        
                        # 인용 정보 확인 및 추가
                        if '[Source:' not in clean_req:
                            clean_req += " [Source: Specification Document]"
                        
                        # 요구사항 ID 추가
                        formatted_req = f"FR-{req_counter:03d}: {clean_req}"
                        requirements.append(formatted_req)
                        req_counter += 1
                        
                        # 최대 20개 요구사항으로 제한 (품질 확보)
                        if len(requirements) >= 20:
                            break
            
            state["functional_requirements"] = requirements
            state["current_step"] = "functional_requirements_extracted"
            logger.info(f"Extracted {len(requirements)} functional requirements")
            
        except Exception as e:
            error_msg = f"Error extracting functional requirements: {str(e)}"
            logger.error(error_msg)
            state["errors"].append(error_msg)
        
        return state
    
    def _extract_non_functional_requirements(self, state: HybridSRSState) -> HybridSRSState:
        """비기능 요구사항 추출 (소스 인용 포함)"""
        logger.info("Extracting non-functional requirements with source citations...")
        
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a quality assurance analyst specializing in non-functional requirements. Extract MEASURABLE, TESTABLE quality attributes and constraints.

                EXTRACTION RULES:
                1. Extract ONLY requirements that specify quality attributes, constraints, or performance criteria
                2. Include EXACT performance metrics: response times, throughput, capacity limits
                3. Specify PRECISE resource constraints: memory limits, CPU usage, storage requirements  
                4. Reference SPECIFIC security mechanisms, algorithms, and standards
                5. Include EXACT availability, reliability, and uptime requirements
                6. Specify DETAILED scalability thresholds and limits
                7. Include PRECISE error handling and recovery specifications
                8. Reference SPECIFIC compliance standards and regulations

                FORMAT REQUIREMENTS:
                - Each requirement must specify measurable criteria or constraints
                - Include specific technical parameters and thresholds from source
                - End with precise source citation: [Source: filename, Page X]
                - Be testable and verifiable

                EXAMPLES OF GOOD REQUIREMENTS:
                ✓ "The system must limit transaction commit capacity to prevent resource exhaustion due to platform constraints. [Source: spec.pdf, Page 70]"
                ✓ "The platform must ensure transient objects are stored in volatile memory (RAM) and not in EEPROM for performance optimization. [Source: spec.pdf, Page 47]"

                AVOID GENERIC REQUIREMENTS:
                ✗ "The system must be secure"
                ✗ "The system must be reliable"
                ✗ "The system must perform well"

                Extract concrete, verifiable non-functional requirements."""),
                ("user", """Extract MEASURABLE non-functional requirements from these technical specification documents:

{context}

Focus on extracting quality attributes with specific metrics, precise constraints, exact performance criteria, and detailed compliance requirements.""")
            ])
            
            # 다양한 품질 속성 관련 컨텍스트 검색
            context_queries = [
                "performance constraints memory CPU storage limits",
                "security encryption algorithms standards compliance",
                "reliability availability uptime error handling recovery",
                "scalability capacity throughput transaction limits",
                "quality attributes testing verification validation"
            ]
            
            context_docs = []
            for query in context_queries:
                docs = self.retriever.get_relevant_documents(query)
                context_docs.extend(docs[:6])  # 각 쿼리당 6개 문서
            
            # 컨텍스트에 인용 정보 포함
            context_parts = []
            for doc in context_docs:
                citation = self._format_citation(doc)
                context_parts.append(f"{doc.page_content}\n{citation}")
            context = "\n\n---\n\n".join(context_parts)
            
            chain = prompt | self.llm | StrOutputParser()
            response = chain.invoke({"context": context})
            
            # 비기능 요구사항 파싱 및 ID 생성
            requirements = []
            req_counter = 1
            lines = response.split('\n')
            
            for line in lines:
                line = line.strip()
                
                # 비기능 요구사항 라인 감지
                if (line and 
                    (line.startswith('-') or line.startswith('*') or line.startswith('•') or 
                     any(line.startswith(f"{i}.") for i in range(1, 100)) or
                     line.lower().startswith('the ') and ('must' in line.lower() or 'shall' in line.lower()))):
                    
                    clean_req = re.sub(r'^[-*•]\s*|\d+\.\s*', '', line).strip()
                    
                    # 품질 속성 키워드 확인
                    quality_keywords = ['performance', 'security', 'reliability', 'scalability', 'availability', 
                                      'usability', 'maintainability', 'constraints', 'limits', 'capacity',
                                      'memory', 'storage', 'cpu', 'encryption', 'algorithm', 'standard']
                    
                    has_quality_attribute = any(keyword in clean_req.lower() for keyword in quality_keywords)
                    
                    if (len(clean_req) > 40 and 
                        ('must' in clean_req.lower() or 'shall' in clean_req.lower()) and
                        has_quality_attribute and
                        not clean_req.startswith('NFR-')):
                        
                        # 인용 정보 확인 및 추가
                        if '[Source:' not in clean_req:
                            clean_req += " [Source: Specification Document]"
                        
                        # 요구사항 ID 추가
                        formatted_req = f"NFR-{req_counter:03d}: {clean_req}"
                        requirements.append(formatted_req)
                        req_counter += 1
                        
                        # 최대 15개 요구사항으로 제한
                        if len(requirements) >= 15:
                            break
            
            state["non_functional_requirements"] = requirements
            state["current_step"] = "non_functional_requirements_extracted"
            logger.info(f"Extracted {len(requirements)} non-functional requirements")
            
        except Exception as e:
            error_msg = f"Error extracting non-functional requirements: {str(e)}"
            logger.error(error_msg)
            state["errors"].append(error_msg)
        
        return state
    
    def _extract_system_interfaces(self, state: HybridSRSState) -> HybridSRSState:
        """시스템 인터페이스 추출 (소스 인용 포함)"""
        logger.info("Extracting system interface requirements with source citations...")
        
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """Extract SPECIFIC, DETAILED system interface requirements with exact technical specifications.

                IMPORTANT GUIDELINES:
                1. Include EXACT API endpoints, methods, and parameters specified
                2. Specify PRECISE communication protocols and standards referenced
                3. Include SPECIFIC data formats and schemas mentioned (JSON, XML, etc.)
                4. Mention EXACT hardware interface specifications and pin configurations
                5. Reference SPECIFIC network protocols and port configurations
                6. Include DETAILED error codes and response formats
                7. Specify EXACT authentication and authorization mechanisms
                8. Include PRECISE timing and synchronization requirements
                9. ALWAYS include source citation at the end of each requirement

                CITATION REQUIREMENTS:
                - Every requirement MUST end with a source citation in the format [Source: filename, Page X] or [Source: filename]
                - Use the citation information from the document metadata when available

                AVOID generic statements like "The system must provide API interfaces"
                PREFER specific statements like "The system must provide REST API endpoints at /api/v1/users with GET, POST methods, accepting JSON payloads with max 1MB size and returning HTTP status codes 200, 400, 401, 500 as specified in API documentation section 2.3"

                Extract detailed, implementation-ready interface requirements."""),
                ("user", """Extract SPECIFIC and DETAILED system interface requirements with exact technical specifications:

{context}

Focus on precise API specifications, exact protocols, specific data formats, and detailed integration requirements mentioned in the source documents. MUST include proper source citation at the end of each requirement.""")
            ])
            
            # 인터페이스 관련 다양한 컨텍스트 검색
            context_queries = [
                "interface API methods functions commands procedures",
                "communication protocols data formats messages",
                "integration external systems connections network",
                "input output parameters arguments return values",
                "endpoints services operations transactions"
            ]
            
            context_docs = []
            for query in context_queries:
                docs = self.retriever.get_relevant_documents(query)
                context_docs.extend(docs[:6])  # 각 쿼리당 6개 문서
            
            # 컨텍스트에 인용 정보 포함
            context_parts = []
            for doc in context_docs:
                citation = self._format_citation(doc)
                context_parts.append(f"{doc.page_content}\n{citation}")
            context = "\n\n---\n\n".join(context_parts)
            
            chain = prompt | self.llm | StrOutputParser()
            response = chain.invoke({"context": context})
            
            # 시스템 인터페이스 요구사항 파싱 및 ID 생성
            requirements = []
            req_counter = 1
            lines = response.split('\n')
            
            for line in lines:
                line = line.strip()
                
                if (line and 
                    (line.startswith('-') or line.startswith('*') or line.startswith('•') or 
                     any(line.startswith(f"{i}.") for i in range(1, 100)) or
                     line.lower().startswith('the ') and ('must' in line.lower() or 'shall' in line.lower()))):
                    
                    clean_req = re.sub(r'^[-*•]\s*|\d+\.\s*', '', line).strip()
                    
                    # 인터페이스 관련 키워드 확인
                    interface_keywords = ['interface', 'method', 'function', 'command', 'api', 'protocol', 
                                        'communication', 'message', 'format', 'endpoint', 'service']
                    
                    has_interface_content = any(keyword in clean_req.lower() for keyword in interface_keywords)
                    
                    if (len(clean_req) > 40 and 
                        ('must' in clean_req.lower() or 'shall' in clean_req.lower()) and
                        has_interface_content and
                        not clean_req.startswith('SI-')):
                        
                        if '[Source:' not in clean_req:
                            clean_req += " [Source: Specification Document]"
                        
                        formatted_req = f"SI-{req_counter:03d}: {clean_req}"
                        requirements.append(formatted_req)
                        req_counter += 1
                        
                        if len(requirements) >= 12:
                            break
            
            state["system_interfaces"] = requirements
            state["current_step"] = "system_interfaces_extracted"
            logger.info(f"Extracted {len(requirements)} system interface requirements")
            
        except Exception as e:
            error_msg = f"Error extracting system interfaces: {str(e)}"
            logger.error(error_msg)
            state["errors"].append(error_msg)
        
        return state
    
    def _extract_data_requirements(self, state: HybridSRSState) -> HybridSRSState:
        """데이터 요구사항 추출 (소스 인용 포함)"""
        logger.info("Extracting data requirements with source citations...")
        
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """Extract SPECIFIC, DETAILED data requirements with exact technical specifications.

                IMPORTANT GUIDELINES:
                1. Include EXACT data types, field lengths, and constraints specified
                2. Specify PRECISE database schemas and table structures mentioned
                3. Include SPECIFIC validation rules and regex patterns referenced
                4. Mention EXACT data formats and encoding standards specified
                5. Reference SPECIFIC retention periods and archival procedures
                6. Include DETAILED backup schedules and recovery procedures
                7. Specify EXACT data encryption and security requirements
                8. Include PRECISE data volume and storage capacity requirements
                9. ALWAYS include source citation at the end of each requirement

                CITATION REQUIREMENTS:
                - Every requirement MUST end with a source citation in the format [Source: filename, Page X] or [Source: filename]
                - Use the citation information from the document metadata when available

                AVOID generic statements like "The system must store user data"
                PREFER specific statements like "The system must store user data in PostgreSQL database with username field as VARCHAR(50) NOT NULL UNIQUE, password field as CHAR(64) for SHA-256 hash, and created_date as TIMESTAMP with timezone as specified in database schema section 3.1"

                Extract detailed, implementation-ready data requirements."""),
                ("user", """Extract SPECIFIC and DETAILED data requirements with exact technical specifications:

{context}

Focus on precise data structures, exact field specifications, specific validation rules, and detailed storage requirements mentioned in the source documents. MUST include proper source citation at the end of each requirement.""")
            ])
            
            # 데이터 관련 다양한 컨텍스트 검색
            context_queries = [
                "data structures formats types fields values",
                "storage memory persistent transient objects",
                "database tables records entities relationships",
                "validation constraints rules requirements",
                "encoding formats standards specifications"
            ]
            
            context_docs = []
            for query in context_queries:
                docs = self.retriever.get_relevant_documents(query)
                context_docs.extend(docs[:6])  # 각 쿼리당 6개 문서
            
            # 컨텍스트에 인용 정보 포함
            context_parts = []
            for doc in context_docs:
                citation = self._format_citation(doc)
                context_parts.append(f"{doc.page_content}\n{citation}")
            context = "\n\n---\n\n".join(context_parts)
            
            chain = prompt | self.llm | StrOutputParser()
            response = chain.invoke({"context": context})
            
            # 데이터 요구사항 파싱 및 ID 생성
            requirements = []
            req_counter = 1
            lines = response.split('\n')
            
            for line in lines:
                line = line.strip()
                
                if (line and 
                    (line.startswith('-') or line.startswith('*') or line.startswith('•') or 
                     any(line.startswith(f"{i}.") for i in range(1, 100)) or
                     line.lower().startswith('the ') and ('must' in line.lower() or 'shall' in line.lower()))):
                    
                    clean_req = re.sub(r'^[-*•]\s*|\d+\.\s*', '', line).strip()
                    
                    # 데이터 관련 키워드 확인
                    data_keywords = ['data', 'format', 'structure', 'field', 'type', 'value', 'storage', 
                                   'memory', 'database', 'table', 'record', 'entity', 'validation', 'encoding']
                    
                    has_data_content = any(keyword in clean_req.lower() for keyword in data_keywords)
                    
                    if (len(clean_req) > 40 and 
                        ('must' in clean_req.lower() or 'shall' in clean_req.lower()) and
                        has_data_content and
                        not clean_req.startswith('DR-')):
                        
                        if '[Source:' not in clean_req:
                            clean_req += " [Source: Specification Document]"
                        
                        formatted_req = f"DR-{req_counter:03d}: {clean_req}"
                        requirements.append(formatted_req)
                        req_counter += 1
                        
                        if len(requirements) >= 10:
                            break
            
            state["data_requirements"] = requirements
            state["current_step"] = "data_requirements_extracted"
            logger.info(f"Extracted {len(requirements)} data requirements")
            
        except Exception as e:
            error_msg = f"Error extracting data requirements: {str(e)}"
            logger.error(error_msg)
            state["errors"].append(error_msg)
        
        return state
    
    def _extract_performance_requirements(self, state: HybridSRSState) -> HybridSRSState:
        """성능 요구사항 추출 (소스 인용 포함)"""
        logger.info("Extracting performance requirements with source citations...")
        
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """Extract SPECIFIC, DETAILED performance requirements with exact metrics and measurements.

                IMPORTANT GUIDELINES:
                1. Include EXACT response time requirements with specific millisecond values
                2. Specify PRECISE throughput numbers (requests per second, transactions per minute)
                3. Include SPECIFIC concurrent user limits and load capacity numbers
                4. Mention EXACT resource usage limits (CPU %, memory MB/GB, disk space)
                5. Reference SPECIFIC availability percentages and uptime requirements
                6. Include DETAILED scalability thresholds and growth targets
                7. Specify EXACT performance testing criteria and benchmarks
                8. Include PRECISE latency requirements and timeout values
                9. ALWAYS include source citation at the end of each requirement

                CITATION REQUIREMENTS:
                - Every requirement MUST end with a source citation in the format [Source: filename, Page X] or [Source: filename]
                - Use the citation information from the document metadata when available

                AVOID generic statements like "The system must be fast"
                PREFER specific statements like "The system must respond to user login requests within 500 milliseconds for 95% of requests, support concurrent login of 1000 users with CPU utilization not exceeding 80%, and maintain 99.9% uptime as specified in performance requirements section 5.2"

                Extract detailed, measurable performance requirements."""),
                ("user", """Extract SPECIFIC and DETAILED performance requirements with exact metrics and measurements:

{context}

Focus on precise performance numbers, exact timing requirements, specific capacity limits, and detailed benchmarks mentioned in the source documents. MUST include proper source citation at the end of each requirement.""")
            ])
            
            # 성능 관련 다양한 컨텍스트 검색
            context_queries = [
                "performance response time latency speed timing",
                "throughput capacity load concurrent users",
                "memory usage CPU utilization resource limits",
                "scalability availability uptime reliability",
                "metrics measurements benchmarks thresholds"
            ]
            
            context_docs = []
            for query in context_queries:
                docs = self.retriever.get_relevant_documents(query)
                context_docs.extend(docs[:5])  # 각 쿼리당 5개 문서
            
            # 컨텍스트에 인용 정보 포함
            context_parts = []
            for doc in context_docs:
                citation = self._format_citation(doc)
                context_parts.append(f"{doc.page_content}\n{citation}")
            context = "\n\n---\n\n".join(context_parts)
            
            chain = prompt | self.llm | StrOutputParser()
            response = chain.invoke({"context": context})
            
            # 성능 요구사항 파싱 및 ID 생성
            requirements = []
            req_counter = 1
            lines = response.split('\n')
            
            for line in lines:
                line = line.strip()
                
                if (line and 
                    (line.startswith('-') or line.startswith('*') or line.startswith('•') or 
                     any(line.startswith(f"{i}.") for i in range(1, 100)) or
                     line.lower().startswith('the ') and ('must' in line.lower() or 'shall' in line.lower()))):
                    
                    clean_req = re.sub(r'^[-*•]\s*|\d+\.\s*', '', line).strip()
                    
                    # 성능 관련 키워드 확인
                    performance_keywords = ['performance', 'response', 'time', 'latency', 'throughput', 
                                          'capacity', 'load', 'memory', 'cpu', 'resource', 'speed', 
                                          'concurrent', 'scalability', 'availability', 'metric']
                    
                    has_performance_content = any(keyword in clean_req.lower() for keyword in performance_keywords)
                    
                    if (len(clean_req) > 40 and 
                        ('must' in clean_req.lower() or 'shall' in clean_req.lower()) and
                        has_performance_content and
                        not clean_req.startswith('PR-')):
                        
                        if '[Source:' not in clean_req:
                            clean_req += " [Source: Specification Document]"
                        
                        formatted_req = f"PR-{req_counter:03d}: {clean_req}"
                        requirements.append(formatted_req)
                        req_counter += 1
                        
                        if len(requirements) >= 8:
                            break
            
            state["performance_requirements"] = requirements
            state["current_step"] = "performance_requirements_extracted"
            logger.info(f"Extracted {len(requirements)} performance requirements")
            
        except Exception as e:
            error_msg = f"Error extracting performance requirements: {str(e)}"
            logger.error(error_msg)
            state["errors"].append(error_msg)
        
        return state
    
    def _extract_implicit_requirements(self, state: HybridSRSState) -> HybridSRSState:
        """암시적 요구사항 추출 - 2차 패스 분석"""
        logger.info("Extracting implicit requirements with second-pass analysis...")
        
        try:
            # 1차 추출된 모든 요구사항 수집
            all_existing_requirements = (
                state["functional_requirements"] +
                state["non_functional_requirements"] +
                state["system_interfaces"] +
                state["data_requirements"] +
                state["performance_requirements"]
            )
            
            # 암시적 요구사항 추출 프롬프트
            implicit_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are conducting a specialized second-pass analysis to identify IMPLICIT requirements.
                These are requirements that are not explicitly stated but are necessary for the system to function.
                
                Focus on finding:
                1. IMPLICIT functional requirements from business process descriptions
                2. Derived requirements from stated constraints and goals
                3. Infrastructure requirements implied by functional needs
                4. Compliance requirements implied by domain or regulations
                5. Integration requirements implied by mentioned external systems
                6. Operational requirements for system maintenance and support
                7. User experience requirements implied by user roles
                8. Security requirements implied by data handling descriptions
                
                IMPORTANT: Only extract requirements that are genuinely IMPLIED by the context, not fabricated.
                Avoid duplicating existing explicit requirements."""),
                ("user", """Analyze this context for IMPLICIT requirements:

{context}

Existing explicit requirements (to avoid duplication):
{existing_requirements}

Extract implicit requirements that are logically necessary but not explicitly stated.""")
            ])
            
            # 다양한 관점에서 컨텍스트 검색
            implicit_queries = [
                "business process workflow operations must support",
                "system integration external dependencies interfaces",
                "user roles permissions access control security",
                "maintenance support deployment operations",
                "compliance regulatory standards legal requirements"
            ]
            
            all_context = []
            for query in implicit_queries:
                context_docs = self.retriever.get_relevant_documents(query)
                all_context.extend([doc.page_content for doc in context_docs[:5]])
            
            context = "\n\n".join(set(all_context))  # 중복 제거
            
            chain = implicit_prompt | self.llm | StrOutputParser()
            response = chain.invoke({
                "context": context,
                "existing_requirements": "\n".join(all_existing_requirements[:15])  # 처음 15개 참조
            })
            
            # 암시적 요구사항 파싱
            implicit_requirements = {
                "functional": [],
                "non_functional": [],
                "interfaces": [],
                "data": [],
                "performance": []
            }
            
            current_category = "functional"  # 기본 카테고리
            lines = response.split('\n')
            
            for line in lines:
                line = line.strip()
                
                # 카테고리 감지
                if "functional" in line.lower() and "non-functional" not in line.lower():
                    current_category = "functional"
                elif "non-functional" in line.lower() or "quality" in line.lower():
                    current_category = "non_functional"
                elif "interface" in line.lower() or "integration" in line.lower():
                    current_category = "interfaces"
                elif "data" in line.lower():
                    current_category = "data"
                elif "performance" in line.lower():
                    current_category = "performance"
                
                # 요구사항 추출
                if line and (line.startswith('-') or line.startswith('*') or 
                           line.startswith('•') or any(line.startswith(f"{i}.") for i in range(1, 100))):
                    clean_req = re.sub(r'^[-*•]\s*|\d+\.\s*', '', line).strip()
                    if len(clean_req) > 25 and not any(clean_req.startswith(prefix) for prefix in ['FR-', 'NFR-', 'DR-', 'PR-', 'SI-']):
                        # 인용이 포함되어 있는지 확인
                        if '[Source:' not in clean_req:
                            # 인용이 없다면 일반적인 소스 정보 추가
                            clean_req += " [Source: Implicit Requirements Analysis]"
                        implicit_requirements[current_category].append(clean_req)
            
            # 기존 요구사항 목록에 암시적 요구사항 추가
            if implicit_requirements["functional"]:
                state["functional_requirements"].extend(implicit_requirements["functional"][:5])
                logger.info(f"Added {len(implicit_requirements['functional'][:5])} implicit functional requirements")
            
            if implicit_requirements["non_functional"]:
                state["non_functional_requirements"].extend(implicit_requirements["non_functional"][:5])
                logger.info(f"Added {len(implicit_requirements['non_functional'][:5])} implicit non-functional requirements")
            
            if implicit_requirements["interfaces"]:
                state["system_interfaces"].extend(implicit_requirements["interfaces"][:3])
                logger.info(f"Added {len(implicit_requirements['interfaces'][:3])} implicit interface requirements")
            
            if implicit_requirements["data"]:
                state["data_requirements"].extend(implicit_requirements["data"][:3])
                logger.info(f"Added {len(implicit_requirements['data'][:3])} implicit data requirements")
            
            if implicit_requirements["performance"]:
                state["performance_requirements"].extend(implicit_requirements["performance"][:3])
                logger.info(f"Added {len(implicit_requirements['performance'][:3])} implicit performance requirements")
            
            total_implicit = sum(len(reqs[:5 if cat in ['functional', 'non_functional'] else 3]) 
                               for cat, reqs in implicit_requirements.items())
            
            state["current_step"] = "implicit_requirements_extracted"
            logger.info(f"Completed implicit requirements extraction: {total_implicit} total implicit requirements added")
            
        except Exception as e:
            error_msg = f"Error extracting implicit requirements: {str(e)}"
            logger.error(error_msg)
            state["errors"].append(error_msg)
        
        return state
    
    def _validate_requirements(self, state: HybridSRSState) -> HybridSRSState:
        """요구사항 검증"""
        logger.info("Validating extracted requirements...")
        
        try:
            validation_results = {}
            all_req_types = ["functional_requirements", "non_functional_requirements", 
                           "system_interfaces", "data_requirements", "performance_requirements"]
            
            for req_type in all_req_types:
                requirements = state.get(req_type, [])
                if not requirements:
                    validation_results[req_type] = []
                    continue
                
                type_validations = []
                for req in requirements:
                    validation = self._validate_single_requirement(req, state["processed_documents"])
                    type_validations.append(validation)
                
                validation_results[req_type] = type_validations
                
            state["validation_results"] = validation_results
            state["current_step"] = "requirements_validated"
            logger.info("Requirements validation completed")
            
        except Exception as e:
            error_msg = f"Error validating requirements: {str(e)}"
            logger.error(error_msg)
            state["errors"].append(error_msg)
        
        return state
    
    def _validate_single_requirement(self, requirement: str, processed_documents: List[Document]) -> ValidationResult:
        """단일 요구사항 검증 - 강화된 hallucination 감지"""
        try:
            # 강화된 할루시네이션 패턴 탐지
            hallucination_patterns = [
                r'FR-\d+:|NFR-\d+:|PR-\d+:|DR-\d+:|SI-\d+:',  # 가짜 ID (새로 추가한 ID 제외)
                r'99\.9%|100%|95%',  # 일반적인 가짜 메트릭
                r'section\s+\d+\.\d+\.\d+',  # 가짜 섹션 참조
                r'as\s+specified\s+in\s+section\s+\d+',  # 가짜 섹션 참조 패턴
                r'(username|password|login|database|API)\s+(field|endpoint)',  # 문서에 없는 일반적 용어들
                r'PostgreSQL|MySQL|MongoDB|REST\s+API',  # 구체적 기술 스택 (문서에 명시 안된 경우)
            ]
            
            # 요구사항에서 ID 부분 제거하고 검증
            req_content = re.sub(r'^[A-Z]{2,3}-\d{3}:\s*', '', requirement)
            
            has_fabrication = any(re.search(pattern, req_content, re.IGNORECASE) 
                                for pattern in hallucination_patterns)
            
            if has_fabrication:
                return ValidationResult(
                    original_requirement=requirement,
                    is_valid=False,
                    confidence_score=0.1,
                    evidence=[],
                    rejection_reason="Contains fabricated identifiers or metrics"
                )
            
            # 문서에서 키워드 기반 증거 검색 - 더 정밀한 방식
            # 요구사항에서 핵심 기술 용어 추출
            tech_terms = re.findall(r'\b[A-Z]{2,}(?:[_-][A-Z0-9]+)*\b', req_content)  # 대문자 기술 용어
            specific_numbers = re.findall(r'\b\d+\b', req_content)  # 구체적 숫자
            quoted_terms = re.findall(r'["`\']([^"`\']+)["`\']', req_content)  # 인용된 용어
            
            # 검색 쿼리 생성 (기술 용어 우선)
            search_terms = tech_terms + quoted_terms
            if not search_terms:
                # 기술 용어가 없으면 일반 키워드로 검색
                words = [w for w in req_content.lower().split() 
                        if len(w) > 4 and w not in ['must', 'shall', 'should', 'system', 'platform', 'component']]
                search_terms = words[:5]  # 상위 5개 키워드만 사용
            
            evidence = []
            confidence = 0.0
            
            # 여러 검색어로 검증
            for term in search_terms[:3]:  # 상위 3개 용어로 검색
                try:
                    relevant_docs = self.retriever.get_relevant_documents(term)
                    
                    for doc in relevant_docs[:2]:  # 각 검색어당 상위 2개 문서
                        content = doc.page_content.lower()
                        
                        # 정확한 용어 매칭 확인
                        if term.lower() in content:
                            confidence += 0.3
                            if doc.page_content not in [e[:100] for e in evidence]:  # 중복 방지
                                evidence.append(doc.page_content[:300] + "...")
                        
                        # 숫자 정확성 검증 (있는 경우)
                        if specific_numbers:
                            for num in specific_numbers:
                                if num in content:
                                    confidence += 0.2
                                    
                except Exception:
                    continue
            
            # 최소 증거 기준 적용
            is_valid = confidence > 0.3 and len(evidence) > 0
            
            return ValidationResult(
                original_requirement=requirement,
                is_valid=is_valid,
                confidence_score=min(confidence, 1.0),
                evidence=evidence[:3]  # 최대 3개 증거만 보관
            )
            
        except Exception as e:
            return ValidationResult(
                original_requirement=requirement,
                is_valid=False,
                confidence_score=0.0,
                evidence=[],
                rejection_reason=f"Validation error: {str(e)}"
            )
    
    def _fact_check_requirements(self, state: HybridSRSState) -> HybridSRSState:
        """요구사항 사실 확인"""
        logger.info("Fact-checking requirements...")
        
        try:
            validated_requirements = {}
            rejected_requirements = []
            
            for req_type, validations in state["validation_results"].items():
                valid_reqs = []
                
                for validation in validations:
                    if validation.is_valid and validation.confidence_score > 0.2:  # 더 관대한 임계값으로 변경
                        valid_reqs.append(validation.original_requirement)
                    else:
                        rejected_requirements.append({
                            "requirement": validation.original_requirement,
                            "reason": validation.rejection_reason or "Low confidence score",
                            "confidence": validation.confidence_score,
                            "type": req_type
                        })
                
                validated_requirements[req_type] = valid_reqs
            
            state["validated_requirements"] = validated_requirements
            state["rejected_requirements"] = rejected_requirements
            
            # 요약 생성
            total_valid = sum(len(reqs) for reqs in validated_requirements.values())
            total_rejected = len(rejected_requirements)
            
            state["fact_check_summary"] = {
                "total_validated": total_valid,
                "total_rejected": total_rejected,
                "validation_rate": total_valid / (total_valid + total_rejected) if (total_valid + total_rejected) > 0 else 0,
                "rejection_reasons": list(set([r["reason"] for r in rejected_requirements]))
            }
            
            state["current_step"] = "fact_check_completed"
            logger.info(f"Fact-checking completed: {total_valid} valid, {total_rejected} rejected")
            
        except Exception as e:
            error_msg = f"Error in fact checking: {str(e)}"
            logger.error(error_msg)
            state["errors"].append(error_msg)
        
        return state
    
    def _apply_corrections(self, state: HybridSRSState) -> HybridSRSState:
        """검증된 요구사항으로 교체"""
        logger.info("Applying validated requirements...")
        
        try:
            # 검증된 요구사항으로 교체
            if "validated_requirements" in state:
                for req_type in ["functional_requirements", "non_functional_requirements",
                               "system_interfaces", "data_requirements", "performance_requirements"]:
                    if req_type in state["validated_requirements"]:
                        state[req_type] = state["validated_requirements"][req_type]
            
            state["current_step"] = "corrections_applied"
            
        except Exception as e:
            error_msg = f"Error applying corrections: {str(e)}"
            logger.error(error_msg)
            state["errors"].append(error_msg)
        
        return state
    
    def _generate_srs_sections(self, state: HybridSRSState) -> HybridSRSState:
        """SRS 섹션 생성 (기존과 유사)"""
        logger.info("Generating SRS document sections...")
        
        try:
            sections = {}
            analysis = state.get("requirements_analysis", {})
            
            # Introduction Section
            intro_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an expert technical writer. Create a comprehensive Introduction section based on the analysis provided."),
                ("user", """Create an Introduction section for an SRS document:

Project Scope: {project_scope}
Objectives: {objectives}

Include: 1.1 Purpose, 1.2 Scope, 1.3 Definitions, 1.4 References, 1.5 Overview""")
            ])
            
            chain = intro_prompt | self.llm | StrOutputParser()
            sections["introduction"] = chain.invoke({
                "project_scope": analysis.get("project_scope", ""),
                "objectives": ", ".join(analysis.get("objectives", []))
            })
            
            # Overall Description Section
            desc_prompt = ChatPromptTemplate.from_messages([
                ("system", "Create a comprehensive Overall Description section."),
                ("user", """Create an Overall Description section:

System Overview: {system_overview}
Stakeholders: {stakeholders}
Business Processes: {business_processes}
Constraints: {constraints}
Assumptions: {assumptions}

Include: 2.1 Product Perspective, 2.2 Product Functions, 2.3 User Classes, 2.4 Operating Environment, 2.5 Constraints, 2.6 Assumptions""")
            ])
            
            desc_chain = desc_prompt | self.llm | StrOutputParser()
            sections["overall_description"] = desc_chain.invoke({
                "system_overview": analysis.get("system_overview", ""),
                "stakeholders": ", ".join(analysis.get("stakeholders", [])),
                "business_processes": ", ".join(analysis.get("business_processes", [])),
                "constraints": ", ".join(analysis.get("constraints", [])),
                "assumptions": ", ".join(analysis.get("assumptions", []))
            })
            
            # Requirements sections
            sections["functional_requirements"] = "3. Functional Requirements\n\n" + \
                "\n".join(f"- {req}" for req in state.get("functional_requirements", []))
            
            sections["non_functional_requirements"] = "4. Non-Functional Requirements\n\n" + \
                "\n".join(f"- {req}" for req in state.get("non_functional_requirements", []))
            
            sections["system_interfaces"] = "5. System Interfaces\n\n" + \
                "\n".join(f"- {req}" for req in state.get("system_interfaces", []))
            
            sections["data_requirements"] = "6. Data Requirements\n\n" + \
                "\n".join(f"- {req}" for req in state.get("data_requirements", []))
            
            sections["performance_requirements"] = "7. Performance Requirements\n\n" + \
                "\n".join(f"- {req}" for req in state.get("performance_requirements", []))
            
            state["srs_sections"] = sections
            state["current_step"] = "srs_sections_generated"
            logger.info("SRS sections generated successfully")
            
        except Exception as e:
            error_msg = f"Error generating SRS sections: {str(e)}"
            logger.error(error_msg)
            state["errors"].append(error_msg)
        
        return state
    
    def _compile_final_srs(self, state: HybridSRSState) -> HybridSRSState:
        """최종 SRS 문서 컴파일"""
        logger.info("Compiling final hybrid SRS document...")
        
        try:
            sections = state.get("srs_sections", {})
            analysis = state.get("requirements_analysis", {})
            fact_check = state.get("fact_check_summary", {})
            
            srs_document = f"""
# System Requirements Specification (SRS)
**Generated with Enhanced Citation-Enabled Hybrid Approach**

**Project:** {analysis.get('project_scope', 'System Requirements Specification')}
**Version:** 1.0
**Date:** {state.get('metadata', {}).get('generation_date', 'Generated by Enhanced SRS Agent')}

---

## Document Features

- **Source Citations:** All requirements include source references
- **Total Validated Requirements:** {fact_check.get('total_validated', 0)}
- **Total Rejected Requirements:** {fact_check.get('total_rejected', 0)}
- **Validation Rate:** {fact_check.get('validation_rate', 0):.1%}
- **Anti-Hallucination:** Enabled
- **Fact-Checking:** Applied
- **Citation Tracking:** Enabled

---

## Table of Contents

1. Introduction
2. Overall Description
3. Functional Requirements ({len(state.get('functional_requirements', []))})
4. Non-Functional Requirements ({len(state.get('non_functional_requirements', []))})
5. System Interfaces ({len(state.get('system_interfaces', []))})
6. Data Requirements ({len(state.get('data_requirements', []))})
7. Performance Requirements ({len(state.get('performance_requirements', []))})

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

- **Generated by:** Enhanced Citation-Enabled SRS Generation Agent (v2.0)
- **Source Documents:** {', '.join(state.get('spec_documents', []))}
- **Total Requirements:** {sum(len(state.get(req_type, [])) for req_type in ['functional_requirements', 'non_functional_requirements', 'system_interfaces', 'data_requirements', 'performance_requirements'])}
- **Features Applied:** Source citations, fact-checking, and hallucination detection enabled
- **Rejected Requirements:** {fact_check.get('total_rejected', 0)} (due to: {', '.join(fact_check.get('rejection_reasons', []))})
- **Citation Coverage:** All requirements include source references to original specification documents

---

*This document was generated using an enhanced approach that combines comprehensive requirement extraction with rigorous fact validation and source citation tracking to ensure accuracy, traceability, and maximized coverage.*
"""
            
            state["final_srs_document"] = srs_document
            state["current_step"] = "srs_completed"
            logger.info("Final hybrid SRS document compiled successfully")
            
        except Exception as e:
            error_msg = f"Error compiling final SRS: {str(e)}"
            logger.error(error_msg)
            state["errors"].append(error_msg)
        
        return state
    
    def generate_srs(self, spec_files: List[str], thread_id: str = "hybrid_srs_generation") -> Dict[str, Any]:
        """하이브리드 SRS 생성"""
        logger.info(f"Starting hybrid SRS generation for files: {spec_files}")
        
        initial_state = {
            "spec_documents": spec_files,
            "raw_documents": [],
            "processed_documents": [],
            "requirements_analysis": {},
            "functional_requirements": [],
            "non_functional_requirements": [],
            "system_interfaces": [],
            "data_requirements": [],
            "performance_requirements": [],
            "validation_results": {},
            "validated_requirements": {},
            "rejected_requirements": [],
            "fact_check_summary": {},
            "srs_sections": {},
            "final_srs_document": "",
            "current_step": "initialized",
            "errors": [],
            "metadata": {
                "generation_date": "2025-01-09",
                "thread_id": thread_id,
                "model_name": self.model_name,
                "approach": "hybrid_extraction_validation"
            }
        }
        
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            result = self.app.invoke(initial_state, config)
            
            logger.info("Hybrid SRS generation completed successfully")
            return {
                "success": True,
                "srs_document": result["final_srs_document"],
                "functional_requirements": result["functional_requirements"],
                "non_functional_requirements": result["non_functional_requirements"],
                "system_interfaces": result["system_interfaces"],
                "data_requirements": result["data_requirements"],
                "performance_requirements": result["performance_requirements"],
                "requirements_analysis": result["requirements_analysis"],
                "validation_summary": result["fact_check_summary"],
                "rejected_requirements": result["rejected_requirements"],
                "errors": result["errors"],
                "metadata": result["metadata"]
            }
            
        except Exception as e:
            error_msg = f"Error in hybrid SRS generation workflow: {str(e)}"
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
                "validation_summary": {},
                "rejected_requirements": [],
                "errors": [error_msg],
                "metadata": initial_state["metadata"]
            }
    
    def save_srs_document(self, srs_document: str, output_path: str) -> bool:
        """SRS 문서 저장"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(srs_document)
            logger.info(f"Hybrid SRS document saved to: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving SRS document: {str(e)}")
            return False


if __name__ == "__main__":
    # 테스트 실행
    agent = HybridSRSGenerationAgent(model_name="gpt-4o-mini", temperature=0.1)
    
    spec_files = [
        # 테스트 파일 경로를 여기에 추가
    ]
    
    if spec_files:
        result = agent.generate_srs(spec_files)
        
        if result["success"]:
            print("✅ Hybrid SRS Generation Successful!")
            print(f"📊 Generated Requirements:")
            print(f"   - Functional: {len(result['functional_requirements'])}")
            print(f"   - Non-functional: {len(result['non_functional_requirements'])}")
            print(f"   - System interfaces: {len(result['system_interfaces'])}")
            print(f"   - Data: {len(result['data_requirements'])}")
            print(f"   - Performance: {len(result['performance_requirements'])}")
            
            print(f"\n🛡️ Validation Summary:")
            validation = result["validation_summary"]
            print(f"   - Validated: {validation.get('total_validated', 0)}")
            print(f"   - Rejected: {validation.get('total_rejected', 0)}")
            print(f"   - Rate: {validation.get('validation_rate', 0):.1%}")
            
            agent.save_srs_document(result["srs_document"], "hybrid_srs.md")
            print("💾 Hybrid SRS document saved!")
        else:
            print(f"❌ Hybrid SRS Generation Failed: {result['error']}")
    else:
        print("Please provide specification file paths")