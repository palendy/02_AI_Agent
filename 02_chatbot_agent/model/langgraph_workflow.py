"""
LangGraph Workflow for Corrective RAG
LangGraph를 사용한 Corrective RAG 워크플로우 구현
"""

import logging
from typing import Dict, Any, List, Optional, TypedDict, Annotated
from dataclasses import dataclass

from langgraph.graph import StateGraph, END
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document

from config import get_config
from model.vector_store import DocumentVectorStore
from model.rag_agent import CorrectiveRAGAgent
from model.chat_history import ChatHistoryManager

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CorrectiveRAGState(TypedDict):
    """Corrective RAG 워크플로우 상태"""
    user_question: str
    search_results: List[Document]
    docs_are_relevant: bool
    relevance_score: float
    retry_count: int
    max_retries: int
    search_source: str
    final_answer: str
    current_query: str
    error_message: str
    similar_questions: List[Dict[str, Any]]
    session_id: str


class CorrectiveRAGWorkflow:
    """LangGraph를 사용한 Corrective RAG 워크플로우"""
    
    def __init__(self, 
                 vector_store: DocumentVectorStore,
                 chat_history_manager: Optional[ChatHistoryManager] = None,
                 model_name: Optional[str] = None):
        """
        CorrectiveRAGWorkflow 초기화
        
        Args:
            vector_store: 문서 벡터 스토어
            chat_history_manager: 채팅 히스토리 매니저
            model_name: 사용할 LLM 모델명
        """
        self.config = get_config()
        self.vector_store = vector_store
        self.chat_history_manager = chat_history_manager
        self.model_name = model_name or self.config.default_model_name
        
        # RAG Agent 초기화
        self.rag_agent = CorrectiveRAGAgent(vector_store, chat_history_manager, model_name)
        
        # 워크플로우 그래프 생성
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """워크플로우 그래프 구성"""
        
        # 상태 그래프 생성
        workflow = StateGraph(CorrectiveRAGState)
        
        # 노드 추가
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("grade", self._grade_node)
        workflow.add_node("generate", self._generate_node)
        workflow.add_node("rewrite", self._rewrite_node)
        workflow.add_node("history_search", self._history_search_node)
        workflow.add_node("final_answer", self._final_answer_node)
        workflow.add_node("error", self._error_node)
        
        # 시작점 설정
        workflow.set_entry_point("retrieve")
        
        # 기본 흐름
        workflow.add_edge("retrieve", "grade")
        
        # 평가 후 분기
        workflow.add_conditional_edges(
            "grade",
            self._should_retry,
            {
                "generate": "generate",
                "rewrite": "rewrite",
                "history_search": "history_search",
                "final_answer": "final_answer",
                "error": "error"
            }
        )
        
        # 재작성 후 검색
        workflow.add_edge("rewrite", "retrieve")
        
        # 채팅 히스토리 검색 후 평가
        workflow.add_edge("history_search", "grade")
        
        # 답변 생성 후 종료
        workflow.add_edge("generate", END)
        
        # 최종 답변 후 종료
        workflow.add_edge("final_answer", END)
        
        # 에러 후 종료
        workflow.add_edge("error", END)
        
        return workflow.compile()
    
    def _retrieve_node(self, state: CorrectiveRAGState) -> Dict[str, Any]:
        """문서 검색 노드"""
        try:
            logger.info(f"문서 검색: {state['current_query']}")
            
            if state.get("search_source", "db") == "db":
                # 벡터 스토어에서 검색
                results = self.vector_store.similarity_search(
                    state["current_query"],
                    k=self.config.max_search_results
                )
            else:
                # 채팅 히스토리 검색
                results = self.rag_agent.search_chat_history(
                    state["current_query"],
                    k=self.config.max_search_results
                )
            
            return {
                "search_results": results,
                "search_source": state.get("search_source", "db")
            }
            
        except Exception as e:
            logger.error(f"문서 검색 실패: {e}")
            return {
                "search_results": [],
                "error_message": f"문서 검색 실패: {str(e)}"
            }
    
    def _grade_node(self, state: CorrectiveRAGState) -> Dict[str, Any]:
        """관련성 평가 노드"""
        try:
            if not state.get("search_results"):
                return {
                    "docs_are_relevant": False,
                    "relevance_score": 0.0,
                    "error_message": "검색 결과가 없습니다."
                }
            
            # 관련성 평가
            grade_result = self.rag_agent.grade_relevance(
                state["user_question"],
                state["search_results"]
            )
            
            return {
                "docs_are_relevant": grade_result["docs_are_relevant"],
                "relevance_score": grade_result["relevance_score"]
            }
            
        except Exception as e:
            logger.error(f"관련성 평가 실패: {e}")
            return {
                "docs_are_relevant": False,
                "relevance_score": 0.0,
                "error_message": f"관련성 평가 실패: {str(e)}"
            }
    
    def _generate_node(self, state: CorrectiveRAGState) -> Dict[str, Any]:
        """답변 생성 노드"""
        try:
            logger.info("답변 생성 중")
            
            answer = self.rag_agent.generate_answer(
                state["user_question"],
                state["search_results"]
            )
            
            return {"final_answer": answer}
            
        except Exception as e:
            logger.error(f"답변 생성 실패: {e}")
            return {
                "final_answer": f"답변 생성 중 오류가 발생했습니다: {str(e)}"
            }
    
    def _rewrite_node(self, state: CorrectiveRAGState) -> Dict[str, Any]:
        """쿼리 재작성 노드"""
        try:
            logger.info("쿼리 재작성 중")
            
            # 재시도 횟수 증가
            retry_count = state.get("retry_count", 0) + 1
            
            # 쿼리 재작성
            new_query = self.rag_agent.rewrite_query(
                state["user_question"],
                state["current_query"],
                f"관련성 점수 부족: {state.get('relevance_score', 0):.3f}"
            )
            
            return {
                "current_query": new_query,
                "retry_count": retry_count,
                "search_source": "db"  # DB 검색으로 전환
            }
            
        except Exception as e:
            logger.error(f"쿼리 재작성 실패: {e}")
            return {
                "error_message": f"쿼리 재작성 실패: {str(e)}"
            }
    
    def _history_search_node(self, state: CorrectiveRAGState) -> Dict[str, Any]:
        """채팅 히스토리 검색 노드"""
        try:
            logger.info("채팅 히스토리 검색 중")
            
            # 채팅 히스토리 검색 실행
            results = self.rag_agent.search_chat_history(
                state["current_query"],
                k=self.config.max_search_results
            )
            
            return {
                "search_results": results,
                "search_source": "history"
            }
            
        except Exception as e:
            logger.error(f"채팅 히스토리 검색 실패: {e}")
            return {
                "search_results": [],
                "error_message": f"채팅 히스토리 검색 실패: {str(e)}"
            }
    
    def _final_answer_node(self, state: CorrectiveRAGState) -> Dict[str, Any]:
        """최종 답변 노드"""
        try:
            logger.info("최종 답변 생성 중")
            
            if state.get("search_results"):
                answer = self.rag_agent.generate_answer(
                    state["user_question"],
                    state["search_results"]
                )
            else:
                answer = "죄송합니다. 관련 정보를 찾을 수 없어 답변할 수 없습니다."
            
            return {"final_answer": answer}
            
        except Exception as e:
            logger.error(f"최종 답변 생성 실패: {e}")
            return {
                "final_answer": f"최종 답변 생성 중 오류가 발생했습니다: {str(e)}"
            }
    
    def _error_node(self, state: CorrectiveRAGState) -> Dict[str, Any]:
        """에러 처리 노드"""
        error_msg = state.get("error_message", "알 수 없는 오류가 발생했습니다.")
        return {"final_answer": f"오류: {error_msg}"}
    
    def _should_retry(self, state: CorrectiveRAGState) -> str:
        """재시도 여부 결정"""
        try:
            # 에러가 있는 경우
            if state.get("error_message"):
                return "error"
            
            # 최대 재시도 횟수 도달
            retry_count = state.get("retry_count", 0)
            max_retries = state.get("max_retries", self.config.max_retries)
            
            if retry_count >= max_retries:
                logger.info(f"최대 재시도 횟수 도달 ({max_retries}회)")
                return "final_answer"
            
            # 관련성 평가 결과 확인
            if not state.get("docs_are_relevant", False):
                relevance_score = state.get("relevance_score", 0.0)
                threshold = self.config.relevance_threshold
                
                if relevance_score < threshold:
                    logger.info(f"관련성 부족 ({relevance_score:.3f} < {threshold}) - 재시도")
                    
                    # 검색 소스 전환: db -> history -> final
                    if state.get("search_source") == "db" and retry_count >= 1:
                        return "history_search"
                    elif state.get("search_source") == "history" and retry_count >= 2:
                        return "final_answer"
                    else:
                        return "rewrite"
            
            # 관련성 통과
            logger.info(f"관련성 통과 ({state.get('relevance_score', 0):.3f})")
            return "generate"
            
        except Exception as e:
            logger.error(f"재시도 결정 실패: {e}")
            return "error"
    
    def process_question(self, question: str, session_id: str = "default") -> Dict[str, Any]:
        """
        질문 처리
        
        Args:
            question: 사용자 질문
            session_id: 세션 ID
            
        Returns:
            Dict[str, Any]: 처리 결과
        """
        try:
            logger.info(f"질문 처리 시작: {question}")
            
            # 초기 상태 설정
            initial_state = {
                "user_question": question,
                "current_query": question,
                "search_results": [],
                "docs_are_relevant": False,
                "relevance_score": 0.0,
                "retry_count": 0,
                "max_retries": self.config.max_retries,
                "search_source": "db",
                "final_answer": "",
                "error_message": "",
                "similar_questions": [],
                "session_id": session_id
            }
            
            # 워크플로우 실행
            result = self.workflow.invoke(initial_state)
            
            # 채팅 히스토리에 질문-답변 저장
            if self.chat_history_manager:
                self.chat_history_manager.add_chat_message(
                    question=question,
                    answer=result.get("final_answer", ""),
                    session_id=session_id,
                    relevance_score=result.get("relevance_score", 0.0),
                    search_source=result.get("search_source", "unknown"),
                    documents_used=len(result.get("search_results", []))
                )
            
            logger.info("질문 처리 완료")
            
            return {
                "question": question,
                "answer": result.get("final_answer", ""),
                "search_source": result.get("search_source", "unknown"),
                "relevance_score": result.get("relevance_score", 0.0),
                "retry_count": result.get("retry_count", 0),
                "documents_used": len(result.get("search_results", [])),
                "error_message": result.get("error_message", ""),
                "similar_questions": result.get("similar_questions", [])
            }
            
        except Exception as e:
            logger.error(f"질문 처리 실패: {e}")
            return {
                "question": question,
                "answer": f"처리 중 오류가 발생했습니다: {str(e)}",
                "search_source": "error",
                "relevance_score": 0.0,
                "retry_count": 0,
                "documents_used": 0,
                "error_message": str(e),
                "similar_questions": []
            }
    
    def get_workflow_info(self) -> Dict[str, Any]:
        """워크플로우 정보 반환"""
        return {
            "model_name": self.model_name,
            "max_retries": self.config.max_retries,
            "relevance_threshold": self.config.relevance_threshold,
            "max_search_results": self.config.max_search_results,
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap
        }


# 사용 예제
if __name__ == "__main__":
    from model.vector_store import DocumentVectorStore
    
    # 벡터 스토어 초기화
    vector_store = DocumentVectorStore()
    
    # 워크플로우 초기화
    workflow = CorrectiveRAGWorkflow(vector_store)
    
    # 워크플로우 정보 출력
    info = workflow.get_workflow_info()
    print(f"워크플로우 정보: {info}")
    
    # 테스트 질문
    question = "GitHub에서 문서를 추출하는 방법은?"
    
    # 질문 처리
    result = workflow.process_question(question)
    print(f"\n질문: {result['question']}")
    print(f"답변: {result['answer']}")
    print(f"검색 소스: {result['search_source']}")
    print(f"관련성 점수: {result['relevance_score']:.3f}")
    print(f"재시도 횟수: {result['retry_count']}")
    print(f"사용된 문서 수: {result['documents_used']}")
    if result['error_message']:
        print(f"오류: {result['error_message']}")
