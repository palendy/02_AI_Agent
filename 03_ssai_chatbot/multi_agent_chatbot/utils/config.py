"""
Configuration management for Multi-Agent Chatbot System
환경 변수를 로드하고 애플리케이션 설정을 관리합니다.
"""

import os
from typing import Optional, List
from dotenv import load_dotenv
from pathlib import Path


class Config:
    """Multi-Agent Chatbot 설정을 관리하는 클래스"""
    
    def __init__(self, env_file: Optional[str] = None):
        """
        Config 객체 초기화
        
        Args:
            env_file: .env 파일 경로 (기본값: 프로젝트 루트의 .env)
        """
        # .env 파일 로드
        if env_file is None:
            # 프로젝트 루트 디렉토리에서 .env 파일 찾기
            current_dir = Path(__file__).parent.parent
            env_file = current_dir / ".env"
        
        if os.path.exists(env_file):
            load_dotenv(env_file)
            print(f"✅ 환경 변수 로드 완료: {env_file}")
        else:
            print(f"⚠️  .env 파일을 찾을 수 없습니다: {env_file}")
            print("   환경 변수를 시스템에서 직접 읽어옵니다.")
    
    # API 키 설정
    @property
    def openai_api_key(self) -> str:
        """OpenAI API 키"""
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
        return key
    
    @property
    def github_token(self) -> Optional[str]:
        """GitHub Personal Access Token"""
        return os.getenv("GITHUB_TOKEN")
    
    # 모델 설정
    @property
    def chat_agent_model(self) -> str:
        """Chat Agent 모델명"""
        return os.getenv("CHAT_AGENT_MODEL", "gpt-4o-mini")
    
    @property
    def rag_agent_model(self) -> str:
        """RAG Agent 모델명"""
        return os.getenv("RAG_AGENT_MODEL", "gpt-4o-mini")
    
    @property
    def issue_agent_model(self) -> str:
        """Issue Agent 모델명"""
        return os.getenv("ISSUE_AGENT_MODEL", "gpt-4o-mini")
    
    @property
    def embedding_model(self) -> str:
        """임베딩 모델명"""
        return os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    
    # MCP 서버 설정
    @property
    def mcp_server_url(self) -> str:
        """MCP 서버 URL"""
        return os.getenv("MCP_SERVER_URL", "localhost:8000")
    
    @property
    def mcp_server_timeout(self) -> int:
        """MCP 서버 타임아웃 (초)"""
        return int(os.getenv("MCP_SERVER_TIMEOUT", "30"))
    
    # Agent 설정
    @property
    def max_retries(self) -> int:
        """최대 재시도 횟수"""
        return int(os.getenv("MAX_RETRIES", "3"))
    
    @property
    def relevance_threshold(self) -> float:
        """관련성 임계값"""
        return float(os.getenv("RELEVANCE_THRESHOLD", "0.6"))
    
    @property
    def max_search_results(self) -> int:
        """최대 검색 결과 수"""
        return int(os.getenv("MAX_SEARCH_RESULTS", "8"))
    
    # GitHub 설정
    @property
    def github_repositories(self) -> List[str]:
        """GitHub repository URL 목록"""
        repos = os.getenv("GITHUB_REPOSITORIES", "")
        if not repos:
            return []
        return [repo.strip() for repo in repos.split(",") if repo.strip()]
    
    @property
    def max_file_size(self) -> int:
        """최대 파일 크기 (bytes)"""
        return int(os.getenv("MAX_FILE_SIZE", str(10 * 1024 * 1024)))  # 10MB
    
    @property
    def supported_file_extensions(self) -> List[str]:
        """지원하는 파일 확장자 목록"""
        extensions = os.getenv("SUPPORTED_FILE_EXTENSIONS", 
                             ".md,.txt,.py,.js,.ts,.html,.htm,.pdf,.doc,.docx,.rst,.json,.yaml,.yml")
        return [ext.strip() for ext in extensions.split(",") if ext.strip()]
    
    # RAG 설정
    @property
    def chunk_size(self) -> int:
        """문서 청크 크기"""
        return int(os.getenv("CHUNK_SIZE", "1500"))
    
    @property
    def chunk_overlap(self) -> int:
        """문서 청크 오버랩"""
        return int(os.getenv("CHUNK_OVERLAP", "300"))
    
    # 데이터베이스 설정
    @property
    def database_url(self) -> str:
        """데이터베이스 URL"""
        return os.getenv("DATABASE_URL", "sqlite:///multi_agent_chatbot.db")
    
    # 로깅 설정
    @property
    def log_level(self) -> str:
        """로그 레벨"""
        return os.getenv("LOG_LEVEL", "INFO")
    
    @property
    def log_file(self) -> str:
        """로그 파일 경로"""
        return os.getenv("LOG_FILE", "multi_agent_chatbot.log")
    
    # 서버 설정
    @property
    def host(self) -> str:
        """서버 호스트"""
        return os.getenv("HOST", "0.0.0.0")
    
    @property
    def port(self) -> int:
        """서버 포트"""
        return int(os.getenv("PORT", "8000"))
    
    @property
    def debug(self) -> bool:
        """디버그 모드"""
        return os.getenv("DEBUG", "False").lower() == "true"
    
    # 대화 설정
    @property
    def max_conversation_history(self) -> int:
        """최대 대화 기록 수"""
        return int(os.getenv("MAX_CONVERSATION_HISTORY", "50"))
    
    @property
    def conversation_timeout(self) -> int:
        """대화 타임아웃 (초)"""
        return int(os.getenv("CONVERSATION_TIMEOUT", "3600"))
    
    def validate(self) -> bool:
        """
        필수 설정값들이 모두 있는지 검증
        
        Returns:
            bool: 모든 필수 설정이 있으면 True, 아니면 False
        """
        try:
            # 필수 API 키들 확인
            _ = self.openai_api_key
            return True
        except ValueError as e:
            print(f"❌ 설정 검증 실패: {e}")
            return False
    
    def get_agent_config(self) -> dict:
        """
        Agent들을 위한 설정 반환
        
        Returns:
            dict: Agent 설정
        """
        return {
            'openai_api_key': self.openai_api_key,
            'github_token': self.github_token,
            'chat_agent_model': self.chat_agent_model,
            'rag_agent_model': self.rag_agent_model,
            'issue_agent_model': self.issue_agent_model,
            'embedding_model': self.embedding_model,
            'max_retries': self.max_retries,
            'relevance_threshold': self.relevance_threshold,
            'max_search_results': self.max_search_results,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'max_conversation_history': self.max_conversation_history,
            'openai_timeout': 30
        }
    
    def get_mcp_config(self) -> dict:
        """
        MCP 서버들을 위한 설정 반환
        
        Returns:
            dict: MCP 서버 설정
        """
        return {
            'github_token': self.github_token,
            'mcp_server_url': self.mcp_server_url,
            'mcp_server_timeout': self.mcp_server_timeout,
            'max_file_size': self.max_file_size,
            'supported_extensions': self.supported_file_extensions
        }
    
    def __str__(self) -> str:
        """설정 정보를 문자열로 반환 (민감한 정보는 마스킹)"""
        return f"""
Multi-Agent Chatbot Config 정보:
- Chat Agent 모델: {self.chat_agent_model}
- RAG Agent 모델: {self.rag_agent_model}
- Issue Agent 모델: {self.issue_agent_model}
- 임베딩 모델: {self.embedding_model}
- 최대 재시도: {self.max_retries}
- 관련성 임계값: {self.relevance_threshold}
- 최대 검색 결과: {self.max_search_results}
- GitHub 토큰: {'설정됨' if self.github_token else '설정되지 않음'}
- GitHub 저장소: {len(self.github_repositories)}개
- 최대 파일 크기: {self.max_file_size // (1024*1024)}MB
- 지원 확장자: {', '.join(self.supported_file_extensions[:5])}{'...' if len(self.supported_file_extensions) > 5 else ''}
- 청크 크기: {self.chunk_size}
- 청크 오버랩: {self.chunk_overlap}
- 데이터베이스: {self.database_url}
- 로그 레벨: {self.log_level}
- 서버: {self.host}:{self.port}
- 디버그: {self.debug}
- 최대 대화 기록: {self.max_conversation_history}
- 대화 타임아웃: {self.conversation_timeout}초
"""


# 전역 설정 인스턴스
config = Config()


def get_config() -> Config:
    """
    전역 설정 인스턴스를 반환
    
    Returns:
        Config: 설정 객체
    """
    return config


# 설정 검증 및 초기화
if __name__ == "__main__":
    print("🔧 Multi-Agent Chatbot 설정 초기화 중...")
    
    if config.validate():
        print("✅ 모든 설정이 올바르게 로드되었습니다.")
        print(config)
    else:
        print("❌ 설정 검증에 실패했습니다.")
        print("   .env 파일을 확인하고 필요한 API 키를 설정하세요.")
