"""
Multi-Agent Workflow
Langchain_Teddy 패턴을 참조한 Multi-Agent 워크플로우
"""

import logging
from typing import Dict, Any, Optional, List, Annotated
from datetime import datetime

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

from agents.chat_agent import ChatAgent
from agents.rag_agent import RAGAgent
from agents.issue_agent import IssueAgent
from mcp_clients.github_mcp_client import GitHubMCPClient


class MultiAgentState:
    """Multi-Agent 워크플로우 상태"""
    def __init__(self):
        # 입력
        self.user_message: str = ""
        self.session_id: str = "default"
        self.context: Optional[Dict[str, Any]] = None
        
        # ChatAgent 결과
        self.message_type: str = "UNKNOWN"
        self.classification_confidence: float = 0.0
        self.classification_reasoning: str = ""
        self.routing_agent: str = "chat"
        
        # RAG Agent 결과
        self.rag_answer: str = ""
        self.rag_documents: List[Dict[str, Any]] = []
        self.rag_relevance_score: float = 0.0
        self.rag_search_source: str = "unknown"
        
        # Issue Agent 결과
        self.issue_answer: str = ""
        self.similar_issues: List[Dict[str, Any]] = []
        self.issue_creation_suggestion: Optional[Dict[str, Any]] = None
        self.new_issue_created: bool = False
        
        # 최종 결과
        self.final_answer: str = ""
        self.processing_agent: str = "unknown"
        self.processing_time: float = 0.0
        self.error_message: str = ""
        
        # 메타데이터
        self.workflow_start_time: float = 0.0
        self.total_retries: int = 0
        self.max_retries: int = 3


class MultiAgentWorkflow:
    """Multi-Agent 워크플로우"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        MultiAgentWorkflow 초기화
        
        Args:
            config: 설정 딕셔너리
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Agent들 초기화
        self.chat_agent = ChatAgent(config)
        self.rag_agent = RAGAgent(config)
        self.issue_agent = IssueAgent(config)
        
        # MCP 클라이언트 초기화
        self.github_mcp = GitHubMCPClient(config)
        
        # 워크플로우 그래프 생성
        self.workflow = self._build_workflow()
        
        # 초기화 상태
        self.is_initialized = False
    
    async def initialize(self) -> bool:
        """워크플로우 초기화"""
        try:
            # 모든 Agent 초기화
            chat_init = await self.chat_agent.initialize()
            rag_init = await self.rag_agent.initialize()
            issue_init = await self.issue_agent.initialize()
            mcp_init = await self.github_mcp.initialize()
            
            self.is_initialized = all([chat_init, rag_init, issue_init, mcp_init])
            
            if self.is_initialized:
                self.logger.info("Langchain Multi-Agent Workflow 초기화 완료")
            else:
                self.logger.error("Langchain Multi-Agent Workflow 초기화 실패")
            
            return self.is_initialized
            
        except Exception as e:
            self.logger.error(f"워크플로우 초기화 중 오류: {e}")
            return False
    
    def _build_workflow(self) -> StateGraph:
        """워크플로우 그래프 구성"""
        
        # 상태 그래프 생성
        workflow = StateGraph(MultiAgentState)
        
        # 노드 추가
        workflow.add_node("chat_agent", self._chat_agent_node)
        workflow.add_node("rag_agent", self._rag_agent_node)
        workflow.add_node("issue_agent", self._issue_agent_node)
        workflow.add_node("final_response", self._final_response_node)
        
        # 시작점 설정
        workflow.set_entry_point("chat_agent")
        
        # ChatAgent에서 라우팅 결정
        workflow.add_conditional_edges(
            "chat_agent",
            self._determine_routing,
            {
                "rag_agent": "rag_agent",
                "issue_agent": "issue_agent",
                "final_response": "final_response"
            }
        )
        
        # RAG Agent에서 최종 응답으로
        workflow.add_edge("rag_agent", "final_response")
        
        # Issue Agent에서 최종 응답으로
        workflow.add_edge("issue_agent", "final_response")
        
        # 최종 응답 후 종료
        workflow.add_edge("final_response", END)
        
        return workflow.compile()
    
    async def _chat_agent_node(self, state: MultiAgentState) -> MultiAgentState:
        """ChatAgent 노드 - 메시지 분류 및 라우팅"""
        try:
            start_time = datetime.now()
            self.logger.info(f"🤖 [CHAT] 메시지 분류 시작: '{state.user_message}'")
            
            # ChatAgent로 메시지 처리
            result = await self.chat_agent.process_message(
                state.user_message, 
                state.context
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            if result["success"]:
                self.logger.info(f"✅ [CHAT] 분류 완료: {result['classification']['message_type']} -> {result['routing']['target_agent']}")
                
                # 상태 업데이트
                state.message_type = result["classification"]["message_type"]
                state.classification_confidence = result["classification"]["confidence"]
                state.classification_reasoning = result["classification"]["reasoning"]
                state.routing_agent = result["routing"]["target_agent"]
                state.processing_time = processing_time
                
            else:
                self.logger.error(f"❌ [CHAT] 분류 실패: {result.get('error')}")
                state.message_type = "UNKNOWN"
                state.classification_confidence = 0.0
                state.classification_reasoning = f"분류 실패: {result.get('error')}"
                state.routing_agent = "chat"
                state.error_message = result.get("error", "ChatAgent 처리 실패")
                state.processing_time = processing_time
            
            return state
                
        except Exception as e:
            self.logger.error(f"❌ [CHAT] ChatAgent 노드 오류: {e}")
            state.message_type = "UNKNOWN"
            state.classification_confidence = 0.0
            state.classification_reasoning = f"오류 발생: {str(e)}"
            state.routing_agent = "chat"
            state.error_message = str(e)
            state.processing_time = 0.0
            return state
    
    async def _rag_agent_node(self, state: MultiAgentState) -> MultiAgentState:
        """RAG Agent 노드 - 정보 검색 및 답변 생성"""
        try:
            start_time = datetime.now()
            self.logger.info(f"📚 [RAG] 정보 검색 시작: '{state.user_message}'")
            
            # RAG Agent로 메시지 처리
            result = await self.rag_agent.process_message(
                state.user_message,
                state.context
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            if result["success"]:
                self.logger.info(f"✅ [RAG] 답변 생성 완료: {len(result['answer'])}자")
                
                # 상태 업데이트
                state.rag_answer = result["answer"]
                state.rag_documents = result.get("documents", [])
                state.rag_relevance_score = result.get("relevance_score", 0.0)
                state.rag_search_source = result.get("search_source", "unknown")
                state.processing_agent = "rag"
                state.processing_time = processing_time
                
            else:
                self.logger.error(f"❌ [RAG] 답변 생성 실패: {result.get('error')}")
                state.rag_answer = f"정보를 찾을 수 없습니다: {result.get('error')}"
                state.rag_documents = []
                state.rag_relevance_score = 0.0
                state.rag_search_source = "error"
                state.processing_agent = "rag"
                state.error_message = result.get("error", "RAG Agent 처리 실패")
                state.processing_time = processing_time
            
            return state
                
        except Exception as e:
            self.logger.error(f"❌ [RAG] RAG Agent 노드 오류: {e}")
            state.rag_answer = f"정보 검색 중 오류가 발생했습니다: {str(e)}"
            state.rag_documents = []
            state.rag_relevance_score = 0.0
            state.rag_search_source = "error"
            state.processing_agent = "rag"
            state.error_message = str(e)
            state.processing_time = 0.0
            return state
    
    async def _issue_agent_node(self, state: MultiAgentState) -> MultiAgentState:
        """Issue Agent 노드 - 이슈 검색 및 관리"""
        try:
            start_time = datetime.now()
            self.logger.info(f"🔧 [ISSUE] 이슈 검색 시작: '{state.user_message}'")
            
            # Issue Agent로 메시지 처리
            result = await self.issue_agent.process_message(
                state.user_message,
                state.context
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            if result["success"]:
                self.logger.info(f"✅ [ISSUE] 이슈 처리 완료: {len(result.get('similar_issues', []))}개 이슈 발견")
                
                # 상태 업데이트
                state.issue_answer = result["answer"]
                state.similar_issues = result.get("similar_issues", [])
                state.issue_creation_suggestion = result.get("issue_creation_suggestion")
                state.new_issue_created = result.get("new_issue_created", False)
                state.processing_agent = "issue"
                state.processing_time = processing_time
                
            else:
                self.logger.error(f"❌ [ISSUE] 이슈 처리 실패: {result.get('error')}")
                state.issue_answer = f"이슈를 처리할 수 없습니다: {result.get('error')}"
                state.similar_issues = []
                state.issue_creation_suggestion = None
                state.new_issue_created = False
                state.processing_agent = "issue"
                state.error_message = result.get("error", "Issue Agent 처리 실패")
                state.processing_time = processing_time
            
            return state
                
        except Exception as e:
            self.logger.error(f"❌ [ISSUE] Issue Agent 노드 오류: {e}")
            state.issue_answer = f"이슈 처리 중 오류가 발생했습니다: {str(e)}"
            state.similar_issues = []
            state.issue_creation_suggestion = None
            state.new_issue_created = False
            state.processing_agent = "issue"
            state.error_message = str(e)
            state.processing_time = 0.0
            return state
    
    async def _final_response_node(self, state: MultiAgentState) -> MultiAgentState:
        """최종 응답 노드 - 최종 답변 생성"""
        try:
            start_time = datetime.now()
            self.logger.info(f"🏁 [FINAL] 최종 응답 생성 시작")
            
            # 처리된 Agent에 따라 최종 답변 결정
            processing_agent = state.processing_agent
            
            if processing_agent == "rag":
                final_answer = state.rag_answer
                self.logger.info(f"📚 [FINAL] RAG Agent 답변 사용: {len(final_answer)}자")
                
            elif processing_agent == "issue":
                final_answer = state.issue_answer
                self.logger.info(f"🔧 [FINAL] Issue Agent 답변 사용: {len(final_answer)}자")
                
            else:  # chat
                # ChatAgent가 직접 처리한 경우
                final_answer = "안녕하세요! 도움이 필요하시면 구체적인 질문을 해주세요."
                self.logger.info(f"💬 [FINAL] Chat Agent 기본 답변 사용")
            
            # 오류 메시지가 있는 경우 추가
            if state.error_message:
                final_answer += f"\n\n⚠️ 참고: {state.error_message}"
            
            processing_time = (datetime.now() - start_time).total_seconds()
            total_time = (datetime.now().timestamp() - state.workflow_start_time)
            
            self.logger.info(f"✅ [FINAL] 최종 응답 생성 완료: {len(final_answer)}자, 총 처리시간: {total_time:.2f}초")
            
            # 상태 업데이트
            state.final_answer = final_answer
            state.processing_time = processing_time
            
            return state
            
        except Exception as e:
            self.logger.error(f"❌ [FINAL] 최종 응답 생성 오류: {e}")
            state.final_answer = f"응답 생성 중 오류가 발생했습니다: {str(e)}"
            state.processing_agent = "error"
            state.error_message = str(e)
            state.processing_time = 0.0
            return state
    
    def _determine_routing(self, state: MultiAgentState) -> str:
        """라우팅 결정"""
        try:
            routing_agent = state.routing_agent
            confidence = state.classification_confidence
            
            self.logger.info(f"🤔 [ROUTING] 라우팅 결정: {routing_agent} (신뢰도: {confidence:.3f})")
            
            # 신뢰도가 낮으면 Chat Agent가 직접 처리
            if confidence < 0.3:
                self.logger.info(f"⚠️ [ROUTING] 낮은 신뢰도로 인한 직접 처리")
                return "final_response"
            
            # 라우팅 결정
            if routing_agent == "rag":
                return "rag_agent"
            elif routing_agent == "issue":
                return "issue_agent"
            else:
                return "final_response"
                
        except Exception as e:
            self.logger.error(f"❌ [ROUTING] 라우팅 결정 오류: {e}")
            return "final_response"
    
    async def process_message(self, message: str, session_id: str = "default", context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        메시지 처리
        
        Args:
            message: 사용자 메시지
            session_id: 세션 ID
            context: 추가 컨텍스트
            
        Returns:
            Dict[str, Any]: 처리 결과
        """
        if not self.is_initialized:
            return {
                "success": False,
                "error": "워크플로우가 초기화되지 않았습니다.",
                "answer": "시스템을 초기화하는 중입니다. 잠시 후 다시 시도해주세요."
            }
        
        try:
            start_time = datetime.now().timestamp()
            self.logger.info(f"🚀 [WORKFLOW] 메시지 처리 시작: '{message}'")
            
            # 초기 상태 설정
            initial_state = MultiAgentState()
            initial_state.user_message = message
            initial_state.session_id = session_id
            initial_state.context = context
            initial_state.workflow_start_time = start_time
            initial_state.total_retries = 0
            initial_state.max_retries = self.config.get("max_retries", 3)
            
            # 워크플로우 실행
            result_state = self.workflow.invoke(initial_state)
            
            # 처리 시간 계산
            total_time = datetime.now().timestamp() - start_time
            
            self.logger.info(f"✅ [WORKFLOW] 메시지 처리 완료: {total_time:.2f}초")
            
            return {
                "success": True,
                "message": message,
                "answer": result_state.final_answer,
                "processing_agent": result_state.processing_agent,
                "processing_time": total_time,
                "message_type": result_state.message_type,
                "confidence": result_state.classification_confidence,
                "rag_info": {
                    "documents_used": len(result_state.rag_documents),
                    "relevance_score": result_state.rag_relevance_score,
                    "search_source": result_state.rag_search_source
                },
                "issue_info": {
                    "similar_issues": len(result_state.similar_issues),
                    "issue_creation_suggestion": result_state.issue_creation_suggestion,
                    "new_issue_created": result_state.new_issue_created
                },
                "error_message": result_state.error_message
            }
            
        except Exception as e:
            self.logger.error(f"❌ [WORKFLOW] 메시지 처리 실패: {e}")
            return {
                "success": False,
                "message": message,
                "answer": f"처리 중 오류가 발생했습니다: {str(e)}",
                "error": str(e),
                "processing_agent": "error",
                "processing_time": 0.0
            }
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """워크플로우 상태 정보 반환"""
        return {
            "is_initialized": self.is_initialized,
            "agents": {
                "chat_agent": self.chat_agent.get_status(),
                "rag_agent": self.rag_agent.get_status(),
                "issue_agent": self.issue_agent.get_status()
            },
            "mcp_clients": {
                "github_mcp": self.github_mcp.get_status()
            },
            "config": {
                "max_retries": self.config.get("max_retries", 3),
                "relevance_threshold": self.config.get("relevance_threshold", 0.6)
            }
        }
    
    async def reset(self):
        """워크플로우 리셋"""
        try:
            # 모든 Agent 히스토리 초기화
            self.chat_agent.clear_history()
            self.rag_agent.clear_history()
            self.issue_agent.clear_history()
            
            self.logger.info("워크플로우 리셋 완료")
            
        except Exception as e:
            self.logger.error(f"워크플로우 리셋 실패: {e}")
    
    async def close(self):
        """워크플로우 종료"""
        try:
            # MCP 서버 종료
            await self.github_mcp.close()
            
            self.logger.info("워크플로우 종료 완료")
            
        except Exception as e:
            self.logger.error(f"워크플로우 종료 실패: {e}")


# 사용 예제
if __name__ == "__main__":
    import asyncio
    
    async def test_workflow():
        # 설정
        config = {
            "openai_api_key": "your-api-key",
            "github_token": "your-github-token",
            "chat_agent_model": "gpt-4o-mini",
            "rag_agent_model": "gpt-4o-mini",
            "issue_agent_model": "gpt-4o-mini",
            "max_retries": 3,
            "relevance_threshold": 0.6
        }
        
        # 워크플로우 초기화
        workflow = MultiAgentWorkflow(config)
        await workflow.initialize()
        
        # 테스트 메시지
        test_messages = [
            "안녕하세요!",
            "GitHub에서 문서를 추출하는 방법을 알려주세요",
            "버그가 발생했어요. 도움을 주세요",
            "이 기능을 개선해주세요"
        ]
        
        for message in test_messages:
            print(f"\n질문: {message}")
            result = await workflow.process_message(message)
            print(f"답변: {result['answer']}")
            print(f"처리 Agent: {result['processing_agent']}")
            print(f"처리 시간: {result['processing_time']:.2f}초")
        
        # 워크플로우 종료
        await workflow.close()
    
    # 테스트 실행
    asyncio.run(test_workflow())
