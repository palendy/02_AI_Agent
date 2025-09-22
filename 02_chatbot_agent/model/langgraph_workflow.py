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
    answer_quality_score: float
    github_issue_suggestion: Optional[Dict[str, Any]]


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
                "final_answer": "final_answer"
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
                # 검색 결과가 없는 경우 에러 메시지 설정
                if not state.get("error_message"):
                    state["error_message"] = "검색 결과가 없습니다."
            
            # 답변 품질 평가
            quality_score = self._evaluate_answer_quality(state["user_question"], answer)
            logger.info(f"답변 품질 점수: {quality_score:.3f}")
            
            # GitHub Issue 제안 여부 결정
            should_suggest_issue = (
                quality_score < 0.5 or  # 답변 품질이 낮은 경우
                not state.get("search_results") or  # 검색 결과가 없는 경우
                state.get("error_message") or  # 에러가 발생한 경우
                "죄송합니다" in answer or "찾을 수 없" in answer  # 부정적인 답변인 경우
            )
            
            logger.info(f"GitHub Issue 제안 여부: {should_suggest_issue}")
            logger.info(f"검색 결과 있음: {bool(state.get('search_results'))}")
            logger.info(f"에러 메시지: {state.get('error_message')}")
            logger.info(f"답변 내용: {answer[:100]}...")
            
            # 상태 업데이트
            state["final_answer"] = answer
            state["answer_quality_score"] = quality_score
            
            # GitHub Issue 제안이 필요한 경우
            if should_suggest_issue:
                logger.info("GitHub Issue 제안 생성 시작")
                try:
                    from model.github_issue_helper import GitHubIssueHelper
                    
                    # 현재 선택된 repository 정보 가져오기
                    current_repo = self.vector_store.repository_url if hasattr(self.vector_store, 'repository_url') else None
                    logger.info(f"현재 repository: {current_repo}")
                    
                    # GitHub Issue Helper 초기화
                    issue_helper = GitHubIssueHelper(current_repo)
                    
                    # 시스템 정보 수집
                    system_info = {
                        'model_name': self.config.default_model_name,
                        'embedding_model': self.config.embedding_model,
                        'relevance_threshold': self.config.relevance_threshold,
                        'max_retries': self.config.max_retries,
                        'document_count': len(state.get("search_results", [])),
                        'conversation_count': 0,  # 필요시 추가
                        'repository_url': current_repo
                    }
                    
                    # Issue 제안 생성
                    issue_suggestion = issue_helper.suggest_issue_creation(
                        question=state["user_question"],
                        error_message=state.get("error_message"),
                        system_info=system_info
                    )
                    
                    state["github_issue_suggestion"] = issue_suggestion
                    
                except Exception as e:
                    logger.error(f"GitHub Issue 제안 생성 실패: {e}")
                    state["github_issue_suggestion"] = {
                        "suggested": False,
                        "message": f"Issue 제안 생성 중 오류: {str(e)}"
                    }
            
            return state
            
        except Exception as e:
            logger.error(f"최종 답변 생성 실패: {e}")
            return {
                "final_answer": f"최종 답변 생성 중 오류가 발생했습니다: {str(e)}",
                "answer_quality_score": 0.0,
                "should_suggest_issue": True,
                "github_issue_suggestion": {
                    "suggested": True,
                    "message": f"시스템 오류로 인해 GitHub Issue를 생성해주세요: {str(e)}"
                }
            }
    
    def _evaluate_answer_quality(self, question: str, answer: str) -> float:
        """
        답변 품질 평가
        
        Args:
            question: 사용자 질문
            answer: 생성된 답변
            
        Returns:
            float: 품질 점수 (0.0-1.0)
        """
        try:
            # 기본 점수
            score = 1.0
            
            # 부정적인 답변 패턴 체크
            negative_patterns = [
                "죄송합니다", "찾을 수 없", "답변할 수 없", "정보가 없",
                "알 수 없", "확인할 수 없", "제공할 수 없"
            ]
            
            for pattern in negative_patterns:
                if pattern in answer:
                    score -= 0.3
                    break
            
            # 답변 길이 체크 (너무 짧으면 낮은 점수)
            if len(answer.strip()) < 20:
                score -= 0.2
            
            # 질문 키워드가 답변에 포함되어 있는지 체크
            question_words = set(question.lower().split())
            answer_words = set(answer.lower().split())
            
            # 공통 키워드 비율 계산
            common_words = question_words.intersection(answer_words)
            if len(question_words) > 0:
                keyword_coverage = len(common_words) / len(question_words)
                score += keyword_coverage * 0.2
            
            # 구체적인 정보가 포함되어 있는지 체크
            specific_patterns = [
                "방법", "단계", "설치", "사용법", "예시", "코드",
                "설정", "구성", "옵션", "파라미터", "명령어"
            ]
            
            has_specific_info = any(pattern in answer for pattern in specific_patterns)
            if has_specific_info:
                score += 0.1
            
            # 점수 범위 제한 (0.0-1.0)
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"답변 품질 평가 실패: {e}")
            return 0.5  # 기본값
    
    def _should_retry(self, state: CorrectiveRAGState) -> str:
        """재시도 여부 결정"""
        try:
            logger.info(f"_should_retry 호출됨 - 상태: {state}")
            
            # 최대 재시도 횟수 도달
            retry_count = state.get("retry_count", 0)
            max_retries = state.get("max_retries", self.config.max_retries)
            
            if retry_count >= max_retries:
                logger.info(f"최대 재시도 횟수 도달 ({max_retries}회) - final_answer로 이동")
                return "final_answer"
            
            # 관련성 평가 결과 확인
            docs_are_relevant = state.get("docs_are_relevant", False)
            relevance_score = state.get("relevance_score", 0.0)
            search_source = state.get("search_source", "unknown")
            
            logger.info(f"관련성 평가 결과: docs_are_relevant={docs_are_relevant}, relevance_score={relevance_score:.3f}, search_source={search_source}")
            
            if not docs_are_relevant:
                threshold = self.config.relevance_threshold
                
                if relevance_score < threshold:
                    logger.info(f"관련성 부족 ({relevance_score:.3f} < {threshold}) - 재시도")
                    
                    # 검색 소스 전환: db -> history -> final
                    if search_source == "db" and retry_count >= 1:
                        logger.info("DB 검색 후 재시도 - history_search로 이동")
                        return "history_search"
                    elif search_source == "history":
                        logger.info("History 검색 후 - final_answer로 이동")
                        return "final_answer"
                    else:
                        logger.info("쿼리 재작성으로 이동")
                        return "rewrite"
            
            # 관련성 통과
            logger.info(f"관련성 통과 ({relevance_score:.3f}) - generate로 이동")
            return "generate"
            
        except Exception as e:
            logger.error(f"재시도 결정 실패: {e}")
            return "final_answer"  # 에러가 발생해도 final_answer로 이동
    
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
            
            # 디버깅: GitHub Issue 제안 확인
            if result.get("github_issue_suggestion"):
                logger.info(f"워크플로우 결과에 GitHub Issue 제안 있음: {result.get('github_issue_suggestion', {}).get('suggested', False)}")
            else:
                logger.info("워크플로우 결과에 GitHub Issue 제안 없음")
            
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
                "similar_questions": result.get("similar_questions", []),
                "answer_quality_score": result.get("answer_quality_score", 0.0),
                "github_issue_suggestion": result.get("github_issue_suggestion")
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
