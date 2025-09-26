"""
Base Agent class for Multi-Agent Chatbot System
모든 Agent의 기본 클래스
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from enum import Enum
import logging
from datetime import datetime


class AgentType(Enum):
    """Agent 타입 정의"""
    CHAT = "chat"
    RAG = "rag"
    ISSUE = "issue"


class MessageType(Enum):
    """메시지 타입 정의"""
    QUESTION = "question"
    VOC = "voc"
    INFO_REQUEST = "info_request"
    ISSUE_REPORT = "issue_report"
    UNKNOWN = "unknown"


class BaseAgent(ABC):
    """모든 Agent의 기본 클래스"""
    
    def __init__(self, agent_type: AgentType, config: Dict[str, Any]):
        """
        BaseAgent 초기화
        
        Args:
            agent_type: Agent 타입
            config: 설정 딕셔너리
        """
        self.agent_type = agent_type
        self.config = config
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.conversation_history: List[Dict[str, Any]] = []
        self.is_initialized = False
        
    @abstractmethod
    async def initialize(self) -> bool:
        """
        Agent 초기화
        
        Returns:
            bool: 초기화 성공 여부
        """
        pass
    
    @abstractmethod
    async def process_message(self, message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        메시지 처리
        
        Args:
            message: 처리할 메시지
            context: 추가 컨텍스트 정보
            
        Returns:
            Dict[str, Any]: 처리 결과
        """
        pass
    
    def add_to_history(self, message: str, response: str, metadata: Optional[Dict[str, Any]] = None):
        """
        대화 기록에 추가
        
        Args:
            message: 사용자 메시지
            response: Agent 응답
            metadata: 추가 메타데이터
        """
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "response": response,
            "metadata": metadata or {},
            "agent_type": self.agent_type.value
        })
        
        # 히스토리 크기 제한
        max_history = self.config.get("max_conversation_history", 50)
        if len(self.conversation_history) > max_history:
            self.conversation_history = self.conversation_history[-max_history:]
    
    def get_conversation_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        대화 기록 조회
        
        Args:
            limit: 조회할 기록 수 제한
            
        Returns:
            List[Dict[str, Any]]: 대화 기록
        """
        if limit is None:
            return self.conversation_history.copy()
        return self.conversation_history[-limit:]
    
    def clear_history(self):
        """대화 기록 초기화"""
        self.conversation_history.clear()
    
    def get_status(self) -> Dict[str, Any]:
        """
        Agent 상태 정보 반환
        
        Returns:
            Dict[str, Any]: 상태 정보
        """
        return {
            "agent_type": self.agent_type.value,
            "is_initialized": self.is_initialized,
            "conversation_count": len(self.conversation_history),
            "last_activity": self.conversation_history[-1]["timestamp"] if self.conversation_history else None
        }
    
    def __str__(self) -> str:
        """Agent 정보를 문자열로 반환"""
        return f"{self.__class__.__name__}(type={self.agent_type.value}, initialized={self.is_initialized})"
