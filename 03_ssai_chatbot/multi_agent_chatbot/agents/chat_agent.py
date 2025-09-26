"""
Langchain-based Chat Agent - 질문 분류 및 라우팅을 담당하는 Agent
Langchain_Teddy 패턴을 참조한 Chat Agent 구현
"""

import logging
from typing import Dict, Any, Optional, List, Annotated
from datetime import datetime

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import JsonOutputParser

from .base_agent import BaseAgent, AgentType, MessageType


class ChatAgent(BaseAgent):
    """Langchain 기반 Chat Agent - 질문 분류 및 라우팅"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(AgentType.CHAT, config)
        self.llm: Optional[ChatOpenAI] = None
        self.agent_executor: Optional[AgentExecutor] = None
        self.json_parser = JsonOutputParser()
        
        # 분류 프롬프트 템플릿
        self.classification_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""
당신은 사용자의 질문을 분석하여 적절한 Agent로 라우팅하는 Chat Agent입니다.

사용자의 질문을 다음 카테고리로 분류하세요:

1. INFO_REQUEST: 정보를 요청하는 질문
   - "어떻게 사용하나요?", "설명해주세요", "방법을 알려주세요" 등
   - 문서나 가이드를 찾는 질문
   - 기술적인 정보를 요청하는 질문

2. VOC: 고객의 소리 (Voice of Customer)
   - "개선해주세요", "문제가 있어요", "불편해요" 등
   - 피드백이나 개선 요청
   - 사용자 경험 관련 의견

3. ISSUE_REPORT: 버그나 이슈 신고
   - "오류가 발생했어요", "작동하지 않아요", "문제가 생겼어요" 등
   - 버그 리포트
   - 시스템 오류 신고

4. QUESTION: 일반적인 질문
   - "안녕하세요", "뭐해요?" 등
   - 일반적인 대화

분류 결과를 JSON 형태로 반환하세요:
{
    "message_type": "INFO_REQUEST|VOC|ISSUE_REPORT|QUESTION",
    "confidence": 0.0-1.0,
    "reasoning": "분류 이유",
    "routing_agent": "rag|issue|chat",
    "keywords": ["키워드1", "키워드2"]
}
            """),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessage(content="{input}")
        ])
        
        # 일반 대화 프롬프트 템플릿
        self.conversation_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="당신은 친근한 AI 어시스턴트입니다. 사용자의 질문에 정중하고 도움이 되는 답변을 제공하세요."),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessage(content="{input}")
        ])
    
    async def initialize(self) -> bool:
        """ChatAgent 초기화"""
        try:
            # LLM 초기화
            self.llm = ChatOpenAI(
                model=self.config.get("chat_agent_model", "gpt-4o-mini"),
                temperature=0.1,
                api_key=self.config.get("openai_api_key")
            )
            
            # 도구 정의
            tools = [self._create_classification_tool()]
            
            # Agent 생성
            agent = create_openai_tools_agent(
                llm=self.llm,
                tools=tools,
                prompt=self.classification_prompt
            )
            
            # Agent Executor 생성
            self.agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=True,
                max_iterations=3,
                return_intermediate_steps=True
            )
            
            self.is_initialized = True
            self.logger.info("Langchain ChatAgent 초기화 완료")
            return True
            
        except Exception as e:
            self.logger.error(f"Langchain ChatAgent 초기화 실패: {e}")
            return False
    
    def _create_classification_tool(self):
        """분류 도구 생성"""
        @tool
        def classify_message(message: str) -> str:
            """사용자 메시지를 분류하고 라우팅 정보를 반환합니다."""
            try:
                # 간단한 키워드 기반 분류 (실제로는 LLM 사용)
                message_lower = message.lower()
                
                # 정보 요청 키워드
                info_keywords = ["어떻게", "방법", "설명", "가이드", "사용법", "설치", "설정"]
                if any(keyword in message_lower for keyword in info_keywords):
                    return '{"message_type": "INFO_REQUEST", "confidence": 0.8, "reasoning": "정보 요청 키워드 감지", "routing_agent": "rag", "keywords": ["정보", "가이드"]}'
                
                # VOC 키워드
                voc_keywords = ["개선", "불편", "문제", "피드백", "제안"]
                if any(keyword in message_lower for keyword in voc_keywords):
                    return '{"message_type": "VOC", "confidence": 0.8, "reasoning": "VOC 키워드 감지", "routing_agent": "issue", "keywords": ["개선", "피드백"]}'
                
                # 이슈 신고 키워드
                issue_keywords = ["오류", "버그", "에러", "작동하지", "문제가 생겼"]
                if any(keyword in message_lower for keyword in issue_keywords):
                    return '{"message_type": "ISSUE_REPORT", "confidence": 0.9, "reasoning": "이슈 신고 키워드 감지", "routing_agent": "issue", "keywords": ["오류", "버그"]}'
                
                # 기본값
                return '{"message_type": "QUESTION", "confidence": 0.5, "reasoning": "일반 질문으로 분류", "routing_agent": "chat", "keywords": []}'
                
            except Exception as e:
                return f'{{"message_type": "QUESTION", "confidence": 0.0, "reasoning": "분류 오류: {str(e)}", "routing_agent": "chat", "keywords": []}}'
        
        return classify_message
    
    async def process_message(self, message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        메시지를 분석하고 적절한 Agent로 라우팅
        
        Args:
            message: 사용자 메시지
            context: 추가 컨텍스트
            
        Returns:
            Dict[str, Any]: 분류 결과 및 라우팅 정보
        """
        if not self.is_initialized:
            return {
                "success": False,
                "error": "ChatAgent가 초기화되지 않았습니다.",
                "routing_agent": "chat"
            }
        
        try:
            # 채팅 히스토리 가져오기
            chat_history = self._get_chat_history_for_llm()
            
            # 분류 실행
            classification_result = await self._classify_message_with_llm(message, chat_history)
            
            # 라우팅 결정
            routing_decision = self._determine_routing(classification_result, context)
            
            # 응답 생성
            response = await self._generate_response(message, classification_result, routing_decision, chat_history)
            
            # 히스토리에 추가
            self.add_to_history(message, response["message"], {
                "classification": classification_result,
                "routing": routing_decision
            })
            
            return {
                "success": True,
                "message": response["message"],
                "classification": classification_result,
                "routing": routing_decision,
                "routing_agent": routing_decision["target_agent"],
                "confidence": classification_result.get("confidence", 0.0)
            }
            
        except Exception as e:
            self.logger.error(f"메시지 처리 중 오류: {e}")
            return {
                "success": False,
                "error": f"메시지 처리 중 오류가 발생했습니다: {e}",
                "routing_agent": "chat"
            }
    
    async def _classify_message_with_llm(self, message: str, chat_history: List) -> Dict[str, Any]:
        """LLM을 사용한 메시지 분류"""
        try:
            # 분류 체인 구성
            classification_chain = (
                {
                    "input": RunnablePassthrough(),
                    "chat_history": lambda x: chat_history
                }
                | self.classification_prompt
                | self.llm
                | self.json_parser
            )
            
            # 분류 실행
            result = await classification_chain.ainvoke(message)
            
            return result
            
        except Exception as e:
            self.logger.error(f"LLM 분류 실패: {e}")
            # 폴백: 도구 기반 분류
            return self._fallback_classification(message)
    
    def _fallback_classification(self, message: str) -> Dict[str, Any]:
        """폴백 분류 (도구 기반)"""
        try:
            # 분류 도구 사용
            tool = self._create_classification_tool()
            result_str = tool.invoke({"message": message})
            
            # JSON 파싱
            import json
            return json.loads(result_str)
            
        except Exception as e:
            self.logger.error(f"폴백 분류 실패: {e}")
            return {
                "message_type": "QUESTION",
                "confidence": 0.0,
                "reasoning": f"분류 실패: {str(e)}",
                "routing_agent": "chat",
                "keywords": []
            }
    
    def _determine_routing(self, classification: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """라우팅 결정"""
        message_type = classification.get("message_type", "QUESTION")
        confidence = classification.get("confidence", 0.0)
        
        # 신뢰도가 낮으면 Chat Agent가 직접 처리
        if confidence < 0.3:
            return {
                "target_agent": "chat",
                "reason": "낮은 신뢰도로 인한 직접 처리",
                "confidence": confidence
            }
        
        # 메시지 타입에 따른 라우팅
        routing_map = {
            "INFO_REQUEST": "rag",
            "VOC": "issue",
            "ISSUE_REPORT": "issue",
            "QUESTION": "chat"
        }
        
        target_agent = routing_map.get(message_type, "chat")
        
        return {
            "target_agent": target_agent,
            "reason": f"{message_type} 타입으로 {target_agent} agent로 라우팅",
            "confidence": confidence,
            "message_type": message_type
        }
    
    async def _generate_response(self, message: str, classification: Dict[str, Any], routing: Dict[str, Any], chat_history: List) -> Dict[str, Any]:
        """응답 생성"""
        target_agent = routing["target_agent"]
        
        if target_agent == "chat":
            # Chat Agent가 직접 처리
            response = await self._generate_direct_response(message, chat_history)
        else:
            # 다른 Agent로 라우팅
            response = {
                "message": f"질문을 분석한 결과, {target_agent.upper()} Agent가 더 적합한 답변을 제공할 수 있습니다. 해당 Agent로 문의를 전달하겠습니다.",
                "routing_required": True,
                "target_agent": target_agent
            }
        
        return response
    
    async def _generate_direct_response(self, message: str, chat_history: List) -> Dict[str, Any]:
        """Chat Agent가 직접 응답 생성"""
        try:
            # 대화 체인 구성
            conversation_chain = (
                {
                    "input": RunnablePassthrough(),
                    "chat_history": lambda x: chat_history
                }
                | self.conversation_prompt
                | self.llm
            )
            
            # 응답 생성
            response = await conversation_chain.ainvoke(message)
            
            return {
                "message": response.content,
                "routing_required": False
            }
            
        except Exception as e:
            self.logger.error(f"직접 응답 생성 중 오류: {e}")
            return {
                "message": "죄송합니다. 현재 응답을 생성할 수 없습니다. 잠시 후 다시 시도해주세요.",
                "routing_required": False
            }
    
    def _get_chat_history_for_llm(self) -> List:
        """LLM용 채팅 히스토리 변환"""
        history = []
        
        for entry in self.conversation_history[-10:]:  # 최근 10개만
            history.append(HumanMessage(content=entry["message"]))
            history.append(AIMessage(content=entry["response"]))
        
        return history
    
    def get_classification_stats(self) -> Dict[str, Any]:
        """분류 통계 정보 반환"""
        if not self.conversation_history:
            return {"total_messages": 0, "classifications": {}}
        
        classifications = {}
        for entry in self.conversation_history:
            metadata = entry.get("metadata", {})
            classification = metadata.get("classification", {})
            message_type = classification.get("message_type", "UNKNOWN")
            classifications[message_type] = classifications.get(message_type, 0) + 1
        
        return {
            "total_messages": len(self.conversation_history),
            "classifications": classifications
        }