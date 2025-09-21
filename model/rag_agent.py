"""
Corrective RAG Agent
GitHub 문서를 기반으로 한 Corrective RAG 시스템
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.tools import TavilySearchResults

from config import get_config
from model.vector_store import DocumentVectorStore

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CorrectiveAgentState:
    """Corrective RAG Agent의 상태를 관리하는 데이터 클래스"""
    user_question: str
    search_results: List[Document] = None
    docs_are_relevant: bool = False
    relevance_score: float = 0.0
    retry_count: int = 0
    max_retries: int = 3
    search_source: str = "db"  # "db" or "web"
    final_answer: str = ""
    current_query: str = ""


class CorrectiveRAGAgent:
    """Corrective RAG Agent 클래스"""
    
    def __init__(self, 
                 vector_store: DocumentVectorStore,
                 model_name: Optional[str] = None):
        """
        CorrectiveRAGAgent 초기화
        
        Args:
            vector_store: 문서 벡터 스토어
            model_name: 사용할 LLM 모델명
        """
        self.config = get_config()
        self.vector_store = vector_store
        self.model_name = model_name or self.config.default_model_name
        
        # LLM 초기화
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=0,
            openai_api_key=self.config.openai_api_key
        )
        
        # 웹 검색 도구 초기화
        self.web_search = TavilySearchResults(
            tavily_api_key=self.config.tavily_api_key
        )
        
        # 프롬프트 템플릿 초기화
        self._init_prompts()
    
    def _init_prompts(self):
        """프롬프트 템플릿 초기화"""
        
        # 관련성 평가 프롬프트
        self.relevance_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 문서의 관련성을 평가하는 전문가입니다.
주어진 질문에 대해 검색된 문서들이 얼마나 관련성이 있는지 0.0부터 1.0까지의 점수로 평가하세요.

평가 기준:
- 0.0-0.3: 전혀 관련 없음
- 0.3-0.5: 약간 관련 있음 (부족함)
- 0.5-0.7: 관련 있음 (보통)
- 0.7-1.0: 매우 관련 있음 (우수)

응답 형식:
점수: [0.0-1.0 사이의 숫자]
이유: [간단한 설명]"""),
            ("human", """질문: {question}

검색된 문서들:
{context}

점수를 평가해주세요.""")
        ])
        
        # 답변 생성 프롬프트
        self.answer_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 주어진 문서를 기반으로 정확하고 유용한 답변을 제공하는 AI 어시스턴트입니다.

규칙:
1. 주어진 문서의 내용만을 기반으로 답변하세요.
2. 문서에 없는 정보는 추측하지 마세요.
3. 답변할 수 없는 경우 "문서에서 해당 정보를 찾을 수 없습니다"라고 명시하세요.
4. 답변의 근거가 되는 문서를 참조하세요.
5. 한국어로 답변하세요."""),
            ("human", """검색된 문서:
{context}

질문: {question}

답변:""")
        ])
        
        # 쿼리 재작성 프롬프트
        self.rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 검색 쿼리를 개선하는 전문가입니다.
기존 검색 결과가 부족할 때, 더 나은 검색 결과를 얻기 위해 쿼리를 재작성하세요.

규칙:
1. 원래 질문의 의도를 유지하세요.
2. 더 구체적이고 명확한 키워드를 사용하세요.
3. 동의어나 관련 용어를 포함하세요.
4. 검색에 적합한 형태로 변환하세요.
5. 한국어로 작성하세요."""),
            ("human", """원래 질문: {question}
기존 검색 쿼리: {current_query}
검색 결과 부족 이유: {reason}

개선된 검색 쿼리를 작성해주세요:""")
        ])
    
    def retrieve_documents(self, query: str, k: int = 5) -> List[Document]:
        """
        문서 검색
        
        Args:
            query: 검색 쿼리
            k: 반환할 문서 수
            
        Returns:
            List[Document]: 검색된 문서 목록
        """
        try:
            logger.info(f"문서 검색 중: {query}")
            results = self.vector_store.similarity_search(query, k=k)
            logger.info(f"검색 완료: {len(results)}개 문서")
            return results
            
        except Exception as e:
            logger.error(f"문서 검색 실패: {e}")
            return []
    
    def grade_relevance(self, question: str, documents: List[Document]) -> Dict[str, Any]:
        """
        문서 관련성 평가
        
        Args:
            question: 사용자 질문
            documents: 평가할 문서 목록
            
        Returns:
            Dict[str, Any]: 평가 결과
        """
        try:
            if not documents:
                return {
                    "docs_are_relevant": False,
                    "relevance_score": 0.0,
                    "reason": "검색된 문서가 없습니다."
                }
            
            # 문서 내용을 하나로 합치기
            context = "\n\n".join([doc.page_content for doc in documents])
            
            # 관련성 평가 실행
            response = self.relevance_prompt.invoke({
                "question": question,
                "context": context
            })
            
            result = self.llm.invoke(response)
            content = result.content
            
            # 점수 추출
            score = 0.0
            reason = "평가 실패"
            
            for line in content.split('\n'):
                if line.startswith('점수:'):
                    try:
                        score = float(line.split(':')[1].strip())
                    except:
                        pass
                elif line.startswith('이유:'):
                    reason = line.split(':', 1)[1].strip()
            
            # 임계값과 비교
            threshold = self.config.relevance_threshold
            is_relevant = score >= threshold
            
            logger.info(f"관련성 평가: {score:.3f} (임계값: {threshold}) - {'통과' if is_relevant else '실패'}")
            
            return {
                "docs_are_relevant": is_relevant,
                "relevance_score": score,
                "reason": reason
            }
            
        except Exception as e:
            logger.error(f"관련성 평가 실패: {e}")
            return {
                "docs_are_relevant": False,
                "relevance_score": 0.0,
                "reason": f"평가 오류: {str(e)}"
            }
    
    def generate_answer(self, question: str, documents: List[Document]) -> str:
        """
        답변 생성
        
        Args:
            question: 사용자 질문
            documents: 참조 문서 목록
            
        Returns:
            str: 생성된 답변
        """
        try:
            if not documents:
                return "죄송합니다. 관련 문서를 찾을 수 없어 답변할 수 없습니다."
            
            # 문서 내용을 하나로 합치기
            context = "\n\n".join([doc.page_content for doc in documents])
            
            # 답변 생성 실행
            response = self.answer_prompt.invoke({
                "question": question,
                "context": context
            })
            
            result = self.llm.invoke(response)
            return result.content
            
        except Exception as e:
            logger.error(f"답변 생성 실패: {e}")
            return f"답변 생성 중 오류가 발생했습니다: {str(e)}"
    
    def rewrite_query(self, question: str, current_query: str, reason: str) -> str:
        """
        검색 쿼리 재작성
        
        Args:
            question: 원래 질문
            current_query: 현재 쿼리
            reason: 재작성 이유
            
        Returns:
            str: 재작성된 쿼리
        """
        try:
            response = self.rewrite_prompt.invoke({
                "question": question,
                "current_query": current_query,
                "reason": reason
            })
            
            result = self.llm.invoke(response)
            return result.content.strip()
            
        except Exception as e:
            logger.error(f"쿼리 재작성 실패: {e}")
            return current_query
    
    def web_search_documents(self, query: str, max_results: int = 3) -> List[Document]:
        """
        웹 검색을 통한 문서 수집
        
        Args:
            query: 검색 쿼리
            max_results: 최대 결과 수
            
        Returns:
            List[Document]: 검색된 문서 목록
        """
        try:
            logger.info(f"웹 검색 중: {query}")
            
            # Tavily 검색 실행
            search_results = self.web_search.run(query)
            
            # Document 객체로 변환
            documents = []
            for result in search_results[:max_results]:
                doc = Document(
                    page_content=result.get('content', ''),
                    metadata={
                        'source': result.get('url', ''),
                        'title': result.get('title', ''),
                        'source_type': 'web'
                    }
                )
                documents.append(doc)
            
            logger.info(f"웹 검색 완료: {len(documents)}개 문서")
            return documents
            
        except Exception as e:
            logger.error(f"웹 검색 실패: {e}")
            return []
    
    def process_question(self, question: str) -> Dict[str, Any]:
        """
        질문 처리 (Corrective RAG 워크플로우)
        
        Args:
            question: 사용자 질문
            
        Returns:
            Dict[str, Any]: 처리 결과
        """
        try:
            logger.info(f"질문 처리 시작: {question}")
            
            # 초기 상태 설정
            state = CorrectiveAgentState(
                user_question=question,
                current_query=question,
                max_retries=self.config.max_retries
            )
            
            # Corrective RAG 워크플로우 실행
            while state.retry_count < state.max_retries:
                logger.info(f"시도 {state.retry_count + 1}/{state.max_retries}")
                
                # 1. 문서 검색
                if state.search_source == "db":
                    state.search_results = self.retrieve_documents(
                        state.current_query, 
                        k=self.config.max_search_results
                    )
                else:
                    state.search_results = self.web_search_documents(
                        state.current_query,
                        max_results=self.config.max_search_results
                    )
                
                # 2. 관련성 평가
                if state.search_results:
                    grade_result = self.grade_relevance(question, state.search_results)
                    state.docs_are_relevant = grade_result["docs_are_relevant"]
                    state.relevance_score = grade_result["relevance_score"]
                    
                    # 관련성이 충분하면 답변 생성
                    if state.docs_are_relevant:
                        logger.info("관련성 통과 - 답변 생성")
                        state.final_answer = self.generate_answer(question, state.search_results)
                        break
                    else:
                        logger.info(f"관련성 부족 ({state.relevance_score:.3f}) - 쿼리 재작성")
                        # 쿼리 재작성
                        state.current_query = self.rewrite_query(
                            question, 
                            state.current_query, 
                            grade_result["reason"]
                        )
                        state.retry_count += 1
                        
                        # DB 검색으로 전환
                        if state.search_source == "web":
                            state.search_source = "db"
                else:
                    logger.info("검색 결과 없음 - 웹 검색으로 전환")
                    state.search_source = "web"
                    state.retry_count += 1
            
            # 최대 시도 횟수 도달 시 최종 답변
            if not state.final_answer:
                if state.search_results:
                    state.final_answer = self.generate_answer(question, state.search_results)
                else:
                    state.final_answer = "죄송합니다. 관련 정보를 찾을 수 없어 답변할 수 없습니다."
            
            logger.info("질문 처리 완료")
            
            return {
                "question": question,
                "answer": state.final_answer,
                "search_source": state.search_source,
                "relevance_score": state.relevance_score,
                "retry_count": state.retry_count,
                "documents_used": len(state.search_results) if state.search_results else 0
            }
            
        except Exception as e:
            logger.error(f"질문 처리 실패: {e}")
            return {
                "question": question,
                "answer": f"처리 중 오류가 발생했습니다: {str(e)}",
                "search_source": "error",
                "relevance_score": 0.0,
                "retry_count": 0,
                "documents_used": 0
            }


# 사용 예제
if __name__ == "__main__":
    from model.vector_store import DocumentVectorStore
    
    # 벡터 스토어 초기화
    vector_store = DocumentVectorStore()
    
    # RAG Agent 초기화
    rag_agent = CorrectiveRAGAgent(vector_store)
    
    # 테스트 질문
    question = "GitHub에서 문서를 추출하는 방법은?"
    
    # 질문 처리
    result = rag_agent.process_question(question)
    print(f"질문: {result['question']}")
    print(f"답변: {result['answer']}")
    print(f"검색 소스: {result['search_source']}")
    print(f"관련성 점수: {result['relevance_score']:.3f}")
    print(f"재시도 횟수: {result['retry_count']}")
    print(f"사용된 문서 수: {result['documents_used']}")
