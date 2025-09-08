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
from typing import TypedDict, List, Dict, Any, Annotated, Tuple
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
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
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
        workflow.add_node("enhance_extraction", self._enhance_extraction)
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
        workflow.add_edge("extract_performance_requirements", "enhance_extraction")
        workflow.add_edge("enhance_extraction", "validate_requirements")
        workflow.add_edge("validate_requirements", "fact_check_requirements")
        workflow.add_edge("fact_check_requirements", "apply_corrections")
        workflow.add_edge("apply_corrections", "generate_srs_sections")
        workflow.add_edge("generate_srs_sections", "compile_final_srs")
        workflow.add_edge("compile_final_srs", END)
        
        return workflow
    
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
                    "k": 12,  # 더 많은 컨텍스트 검색
                    "fetch_k": 30,
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
        """기능 요구사항 추출 (더 많이 추출)"""
        logger.info("Extracting functional requirements comprehensively...")
        
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert business analyst. Extract ALL functional requirements from the documents.
                
                Functional requirements describe what the system must do. Extract:
                - All system functions and capabilities
                - User interface requirements
                - Data processing functions
                - Business logic operations
                - System behaviors and workflows
                - Integration functions
                - API and interface functions
                
                Be comprehensive - extract as many legitimate functional requirements as possible.
                Format each as a clear requirement statement without made-up identifiers."""),
                ("user", """Extract comprehensive functional requirements from these documents:

{context}

Return a detailed list of functional requirements. Focus on completeness and accuracy.""")
            ])
            
            # 더 많은 기능 관련 컨텍스트 검색
            context_docs = self.retriever.get_relevant_documents(
                "functional requirements features capabilities functions operations business logic interface"
            )
            context = "\n\n".join([doc.page_content for doc in context_docs])
            
            chain = prompt | self.llm | StrOutputParser()
            response = chain.invoke({"context": context})
            
            # 요구사항 파싱 (더 유연하게)
            requirements = []
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                if line and (line.startswith('-') or line.startswith('*') or 
                           line.startswith('•') or any(line.startswith(f"{i}.") for i in range(1, 100))):
                    # 리스트 마커 제거
                    clean_req = re.sub(r'^[-*•]\s*|\d+\.\s*', '', line).strip()
                    if len(clean_req) > 20 and not clean_req.startswith('FR-'):  # 가짜 ID 제거
                        requirements.append(clean_req)
            
            state["functional_requirements"] = requirements
            state["current_step"] = "functional_requirements_extracted"
            logger.info(f"Extracted {len(requirements)} functional requirements")
            
        except Exception as e:
            error_msg = f"Error extracting functional requirements: {str(e)}"
            logger.error(error_msg)
            state["errors"].append(error_msg)
        
        return state
    
    def _extract_non_functional_requirements(self, state: HybridSRSState) -> HybridSRSState:
        """비기능 요구사항 추출 (더 많이 추출)"""
        logger.info("Extracting non-functional requirements comprehensively...")
        
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """Extract ALL non-functional requirements that define system quality attributes.
                
                Include:
                - Performance requirements (response time, throughput, capacity)
                - Security and privacy requirements  
                - Reliability and availability requirements
                - Scalability requirements
                - Usability and accessibility requirements
                - Maintainability and supportability requirements
                - Compliance and regulatory requirements
                - Resource constraints and limitations
                
                Be thorough and extract everything mentioned in the documents."""),
                ("user", """Extract comprehensive non-functional requirements:

{context}

Focus on quality attributes, constraints, and performance expectations mentioned in the documents.""")
            ])
            
            context_docs = self.retriever.get_relevant_documents(
                "performance security reliability scalability usability compliance constraints quality"
            )
            context = "\n\n".join([doc.page_content for doc in context_docs])
            
            chain = prompt | self.llm | StrOutputParser()
            response = chain.invoke({"context": context})
            
            requirements = []
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                if line and (line.startswith('-') or line.startswith('*') or 
                           line.startswith('•') or any(line.startswith(f"{i}.") for i in range(1, 100))):
                    clean_req = re.sub(r'^[-*•]\s*|\d+\.\s*', '', line).strip()
                    if len(clean_req) > 20 and not clean_req.startswith('NFR-'):
                        requirements.append(clean_req)
            
            state["non_functional_requirements"] = requirements
            state["current_step"] = "non_functional_requirements_extracted"
            logger.info(f"Extracted {len(requirements)} non-functional requirements")
            
        except Exception as e:
            error_msg = f"Error extracting non-functional requirements: {str(e)}"
            logger.error(error_msg)
            state["errors"].append(error_msg)
        
        return state
    
    def _extract_system_interfaces(self, state: HybridSRSState) -> HybridSRSState:
        """시스템 인터페이스 추출 (더 많이 추출)"""
        logger.info("Extracting system interface requirements comprehensively...")
        
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """Extract ALL system interface requirements that define how the system interacts with:
                - External systems and APIs
                - User interfaces and presentation layers  
                - Hardware interfaces and devices
                - Database and data storage interfaces
                - Network and communication interfaces
                - File system and data format interfaces
                
                Be comprehensive in extracting interface specifications."""),
                ("user", """Extract system interface requirements:

{context}

Focus on all types of interfaces, protocols, and integration points mentioned.""")
            ])
            
            context_docs = self.retriever.get_relevant_documents(
                "interface API integration external systems database network communication protocols"
            )
            context = "\n\n".join([doc.page_content for doc in context_docs])
            
            chain = prompt | self.llm | StrOutputParser()
            response = chain.invoke({"context": context})
            
            requirements = []
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                if line and (line.startswith('-') or line.startswith('*') or 
                           line.startswith('•') or any(line.startswith(f"{i}.") for i in range(1, 100))):
                    clean_req = re.sub(r'^[-*•]\s*|\d+\.\s*', '', line).strip()
                    if len(clean_req) > 20 and not clean_req.startswith('SI-'):
                        requirements.append(clean_req)
            
            state["system_interfaces"] = requirements
            state["current_step"] = "system_interfaces_extracted"
            logger.info(f"Extracted {len(requirements)} system interface requirements")
            
        except Exception as e:
            error_msg = f"Error extracting system interfaces: {str(e)}"
            logger.error(error_msg)
            state["errors"].append(error_msg)
        
        return state
    
    def _extract_data_requirements(self, state: HybridSRSState) -> HybridSRSState:
        """데이터 요구사항 추출 (더 많이 추출)"""
        logger.info("Extracting data requirements comprehensively...")
        
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """Extract ALL data requirements including:
                - Data entities and their relationships
                - Data storage and retention requirements
                - Data quality and validation rules
                - Data security and privacy requirements
                - Data migration and conversion needs
                - Backup and recovery requirements
                - Data formats and encoding specifications
                
                Be thorough in extracting data-related requirements."""),
                ("user", """Extract comprehensive data requirements:

{context}

Focus on all data-related specifications, formats, and management requirements.""")
            ])
            
            context_docs = self.retriever.get_relevant_documents(
                "data requirements database storage entities relationships quality security formats"
            )
            context = "\n\n".join([doc.page_content for doc in context_docs])
            
            chain = prompt | self.llm | StrOutputParser()
            response = chain.invoke({"context": context})
            
            requirements = []
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                if line and (line.startswith('-') or line.startswith('*') or 
                           line.startswith('•') or any(line.startswith(f"{i}.") for i in range(1, 100))):
                    clean_req = re.sub(r'^[-*•]\s*|\d+\.\s*', '', line).strip()
                    if len(clean_req) > 20 and not clean_req.startswith('DR-'):
                        requirements.append(clean_req)
            
            state["data_requirements"] = requirements
            state["current_step"] = "data_requirements_extracted"
            logger.info(f"Extracted {len(requirements)} data requirements")
            
        except Exception as e:
            error_msg = f"Error extracting data requirements: {str(e)}"
            logger.error(error_msg)
            state["errors"].append(error_msg)
        
        return state
    
    def _extract_performance_requirements(self, state: HybridSRSState) -> HybridSRSState:
        """성능 요구사항 추출 (더 많이 추출)"""
        logger.info("Extracting performance requirements comprehensively...")
        
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """Extract ALL performance requirements including:
                - Response time and latency requirements
                - Throughput and transaction volume requirements
                - Concurrent user and load requirements
                - Resource utilization limits and constraints
                - Scalability requirements and targets
                - Availability and uptime requirements
                
                Extract everything related to system performance mentioned in the documents."""),
                ("user", """Extract comprehensive performance requirements:

{context}

Focus on all performance metrics, benchmarks, and expectations mentioned.""")
            ])
            
            context_docs = self.retriever.get_relevant_documents(
                "performance response time throughput scalability load capacity availability metrics"
            )
            context = "\n\n".join([doc.page_content for doc in context_docs])
            
            chain = prompt | self.llm | StrOutputParser()
            response = chain.invoke({"context": context})
            
            requirements = []
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                if line and (line.startswith('-') or line.startswith('*') or 
                           line.startswith('•') or any(line.startswith(f"{i}.") for i in range(1, 100))):
                    clean_req = re.sub(r'^[-*•]\s*|\d+\.\s*', '', line).strip()
                    if len(clean_req) > 20 and not clean_req.startswith('PR-'):
                        requirements.append(clean_req)
            
            state["performance_requirements"] = requirements
            state["current_step"] = "performance_requirements_extracted"
            logger.info(f"Extracted {len(requirements)} performance requirements")
            
        except Exception as e:
            error_msg = f"Error extracting performance requirements: {str(e)}"
            logger.error(error_msg)
            state["errors"].append(error_msg)
        
        return state
    
    def _enhance_extraction(self, state: HybridSRSState) -> HybridSRSState:
        """추출된 요구사항 보완"""
        logger.info("Enhancing extracted requirements...")
        
        try:
            # 각 카테고리별로 추가 요구사항 탐색
            all_requirements = (
                state["functional_requirements"] +
                state["non_functional_requirements"] +
                state["system_interfaces"] +
                state["data_requirements"] +
                state["performance_requirements"]
            )
            
            # 요구사항 수가 적으면 추가 추출 시도
            if len(all_requirements) < 30:
                logger.info("Low requirement count detected, attempting enhanced extraction...")
                
                enhancement_prompt = ChatPromptTemplate.from_messages([
                    ("system", """You are conducting a second-pass analysis to find additional requirements.
                    Look for implicit requirements, derived requirements, and detailed specifications that might have been missed.
                    
                    Focus on finding:
                    - Implicit functional requirements from system descriptions
                    - Detailed technical specifications
                    - Compliance and regulatory requirements
                    - Additional interface requirements
                    - Operational requirements"""),
                    ("user", """Perform enhanced requirement extraction from this comprehensive context:

{context}

Already found requirements:
{existing_requirements}

Find additional requirements that complement the existing ones.""")
                ])
                
                # 전체 문서에서 추가 컨텍스트 검색
                context_docs = self.retriever.get_relevant_documents(
                    "requirements specifications system must should shall compliance"
                )
                context = "\n\n".join([doc.page_content for doc in context_docs])
                
                chain = enhancement_prompt | self.llm | StrOutputParser()
                response = chain.invoke({
                    "context": context,
                    "existing_requirements": "\n".join(all_requirements[:10])  # 처음 10개만 참조
                })
                
                # 추가 요구사항 파싱
                additional_requirements = []
                lines = response.split('\n')
                for line in lines:
                    line = line.strip()
                    if line and (line.startswith('-') or line.startswith('*') or 
                               line.startswith('•') or any(line.startswith(f"{i}.") for i in range(1, 100))):
                        clean_req = re.sub(r'^[-*•]\s*|\d+\.\s*', '', line).strip()
                        if len(clean_req) > 20:
                            additional_requirements.append(clean_req)
                
                # 기능 요구사항에 추가 (임시로)
                state["functional_requirements"].extend(additional_requirements[:10])
                logger.info(f"Added {len(additional_requirements[:10])} enhanced requirements")
            
            state["current_step"] = "extraction_enhanced"
            
        except Exception as e:
            error_msg = f"Error enhancing extraction: {str(e)}"
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
    
    def _validate_single_requirement(self, requirement: str, documents: List[Document]) -> ValidationResult:
        """단일 요구사항 검증"""
        try:
            # 할루시네이션 패턴 탐지
            hallucination_patterns = [
                r'FR-\d+:|NFR-\d+:|PR-\d+:|DR-\d+:|SI-\d+:',  # 가짜 ID
                r'\d+\.?\d*%',  # 구체적 백분율
                r'\d+\s*(ms|milliseconds|seconds|minutes)',  # 구체적 시간
                r'99\.9%|100%|95%',  # 일반적인 가짜 메트릭
            ]
            
            has_fabrication = any(re.search(pattern, requirement, re.IGNORECASE) 
                                for pattern in hallucination_patterns)
            
            if has_fabrication:
                return ValidationResult(
                    original_requirement=requirement,
                    is_valid=False,
                    confidence_score=0.1,
                    evidence=[],
                    rejection_reason="Contains fabricated identifiers or metrics"
                )
            
            # 문서에서 증거 검색
            query = requirement[:100]  # 요구사항 앞부분으로 검색
            relevant_docs = self.retriever.get_relevant_documents(query)
            
            evidence = []
            confidence = 0.0
            
            for doc in relevant_docs[:3]:  # 상위 3개 문서만 확인
                content = doc.page_content.lower()
                req_lower = requirement.lower()
                
                # 키워드 매칭으로 간단한 검증
                req_keywords = [word for word in req_lower.split() 
                              if len(word) > 3 and word not in ['must', 'shall', 'should', 'system']]
                
                matches = sum(1 for keyword in req_keywords if keyword in content)
                if matches > 0:
                    confidence += min(matches / len(req_keywords), 1.0) * 0.4
                    evidence.append(doc.page_content[:200] + "...")
            
            return ValidationResult(
                original_requirement=requirement,
                is_valid=confidence > 0.3,
                confidence_score=confidence,
                evidence=evidence
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
                    if validation.is_valid and validation.confidence_score > 0.4:
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
**Generated with Hybrid Approach (Enhanced Extraction + Fact Validation)**

**Project:** {analysis.get('project_scope', 'System Requirements Specification')}
**Version:** 1.0
**Date:** {state.get('metadata', {}).get('generation_date', 'Generated by Hybrid SRS Agent')}

---

## Validation Summary

- **Total Validated Requirements:** {fact_check.get('total_validated', 0)}
- **Total Rejected Requirements:** {fact_check.get('total_rejected', 0)}
- **Validation Rate:** {fact_check.get('validation_rate', 0):.1%}
- **Anti-Hallucination:** Enabled
- **Fact-Checking:** Applied

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

- **Generated by:** Hybrid SRS Generation Agent (v1.0)
- **Source Documents:** {', '.join(state.get('spec_documents', []))}
- **Total Requirements:** {sum(len(state.get(req_type, [])) for req_type in ['functional_requirements', 'non_functional_requirements', 'system_interfaces', 'data_requirements', 'performance_requirements'])}
- **Validation Applied:** Fact-checking and hallucination detection enabled
- **Rejected Requirements:** {fact_check.get('total_rejected', 0)} (due to: {', '.join(fact_check.get('rejection_reasons', []))})

---

*This document was generated using a hybrid approach that combines comprehensive requirement extraction with rigorous fact validation to ensure accuracy while maximizing coverage.*
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