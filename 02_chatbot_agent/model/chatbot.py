"""
AI Agent Chatbot
GitHub 문서를 기반으로 한 지능형 챗봇
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from config import get_config
from model.vector_store import DocumentVectorStore
from model.github_extractor import GitHubDocumentExtractor
from model.langgraph_workflow import CorrectiveRAGWorkflow
from model.chat_history import ChatHistoryManager
from model.github_issue_helper import GitHubIssueHelper

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AIChatbot:
    """AI Agent Chatbot 클래스"""
    
    def __init__(self, 
                 collection_name: str = "github_documents",
                 persist_directory: str = "./chroma_db"):
        """
        AIChatbot 초기화
        
        Args:
            collection_name: 벡터 스토어 컬렉션 이름
            persist_directory: 벡터 DB 저장 디렉토리
        """
        self.config = get_config()
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # 컴포넌트 초기화
        self.vector_stores = {}  # repository별 벡터 스토어 저장
        self.current_vector_store = None  # 현재 선택된 벡터 스토어
        self.github_extractor = None
        self.workflow = None
        self.chat_history_manager = None
        self.github_issue_helper = None
        
        # 대화 기록
        self.conversation_history = []
        
        # 초기화 실행
        self._initialize_components()
    
    def _initialize_components(self):
        """컴포넌트 초기화"""
        try:
            logger.info("AI Chatbot 컴포넌트 초기화 중...")
            
            # 벡터 스토어는 repository별로 동적 생성
            logger.info("벡터 스토어는 repository별로 동적 생성됩니다.")
            
            # GitHub Extractor 초기화
            extractor_config = self.config.get_github_extractor_config()
            self.github_extractor = GitHubDocumentExtractor(**extractor_config)
            logger.info("GitHub Extractor 초기화 완료")
            
            # 채팅 히스토리 매니저 초기화
            self.chat_history_manager = ChatHistoryManager()
            logger.info("채팅 히스토리 매니저 초기화 완료")
            
            # GitHub Issue Helper 초기화
            self.github_issue_helper = GitHubIssueHelper()
            logger.info("GitHub Issue Helper 초기화 완료")
            
            # LangGraph 워크플로우는 repository 선택 후 초기화
            self.workflow = None
            logger.info("LangGraph 워크플로우 초기화 완료")
            
            # 설정된 repository들 자동 로드
            self._load_configured_repositories()
            
            logger.info("AI Chatbot 초기화 완료")
            
        except Exception as e:
            logger.error(f"컴포넌트 초기화 실패: {e}")
            raise
    
    def _load_configured_repositories(self):
        """설정된 repository들을 자동으로 로드"""
        try:
            repositories = self.config.github_repositories
            
            if not repositories:
                logger.info("설정된 repository가 없습니다.")
                return
            
            logger.info(f"설정된 Repository 로드 중: {len(repositories)}개")
            
            # Repository들 로드
            for url in repositories:
                try:
                    logger.info(f"Repository 처리 중: {url}")
                    
                    # URL 정규화
                    normalized_url = url
                    if '/tree/' in url:
                        normalized_url = url.split('/tree/')[0]
                        logger.info(f"URL 정규화: {url} -> {normalized_url}")
                    
                    # 이미 로드된 repository인지 확인
                    if normalized_url in self.vector_stores:
                        logger.info(f"✅ Repository 이미 로드됨: {normalized_url}")
                        continue
                    
                    result = self.add_github_repository(url)
                    if result["success"]:
                        logger.info(f"✅ Repository 로드 성공: {url} ({result['documents_count']}개 문서)")
                    else:
                        logger.warning(f"⚠️ Repository 로드 실패: {url} - {result['message']}")
                except Exception as e:
                    logger.error(f"❌ Repository 로드 중 오류: {url} - {e}")
                    continue
            
            # 첫 번째 repository를 기본 선택으로 설정
            if self.vector_stores:
                first_repo = list(self.vector_stores.keys())[0]
                self.set_current_repository(first_repo)
                logger.info(f"기본 repository 설정: {first_repo}")
            
            logger.info("설정된 Repository 로드 완료")
            
        except Exception as e:
            logger.error(f"설정된 Repository 로드 실패: {e}")
            # 초기화 실패를 방지하기 위해 예외를 다시 발생시키지 않음
    
    def set_current_repository(self, repository_url: str) -> bool:
        """
        현재 사용할 repository 설정
        
        Args:
            repository_url: 선택할 repository URL
            
        Returns:
            bool: 설정 성공 여부
        """
        try:
            if repository_url not in self.vector_stores:
                logger.error(f"Repository가 로드되지 않았습니다: {repository_url}")
                return False
            
            self.current_vector_store = self.vector_stores[repository_url]
            
            # 워크플로우 초기화 (현재 벡터 스토어 사용)
            self.workflow = CorrectiveRAGWorkflow(
                self.current_vector_store, 
                self.chat_history_manager
            )
            
            logger.info(f"현재 repository 설정: {repository_url}")
            return True
            
        except Exception as e:
            logger.error(f"Repository 설정 실패: {e}")
            return False
    
    def get_available_repositories(self) -> List[Dict[str, Any]]:
        """
        사용 가능한 repository 목록 반환
        
        Returns:
            List[Dict[str, Any]]: repository 정보 목록
        """
        repositories = []
        for url, vector_store in self.vector_stores.items():
            try:
                info = vector_store.get_collection_info()
                repositories.append({
                    'url': url,
                    'name': vector_store._get_repository_name(url),
                    'document_count': info.get('document_count', 0),
                    'collection_name': info.get('collection_name', 'Unknown')
                })
            except Exception as e:
                logger.error(f"Repository 정보 조회 실패: {url}, {e}")
                continue
        
        return repositories
    
    def get_current_repository(self) -> Optional[str]:
        """
        현재 선택된 repository URL 반환
        
        Returns:
            Optional[str]: 현재 repository URL
        """
        if not self.current_vector_store:
            return None
        
        for url, vector_store in self.vector_stores.items():
            if vector_store == self.current_vector_store:
                return url
        return None
    
    def add_github_repository(self, repository_url: str) -> Dict[str, Any]:
        """
        GitHub repository 추가 및 문서 추출
        
        Args:
            repository_url: GitHub repository URL
            
        Returns:
            Dict[str, Any]: 처리 결과
        """
        try:
            logger.info(f"GitHub repository 추가 중: {repository_url}")
            
            # URL 정규화 (tree 경로 제거)
            normalized_url = repository_url
            if '/tree/' in repository_url:
                normalized_url = repository_url.split('/tree/')[0]
                logger.info(f"URL 정규화: {repository_url} -> {normalized_url}")
            
            # Repository 정보 조회
            repo_info = self.github_extractor.get_repository_info(normalized_url)
            logger.info(f"Repository 정보: {repo_info.get('full_name', 'Unknown')}")
            
            # Repository별 벡터 스토어 생성 또는 가져오기
            if normalized_url not in self.vector_stores:
                self.vector_stores[normalized_url] = DocumentVectorStore(
                    collection_name=self.collection_name,
                    persist_directory=self.persist_directory,
                    repository_url=normalized_url
                )
                logger.info(f"새 벡터 스토어 생성: {normalized_url}")
            
            # 기존 문서 수 확인
            existing_docs = self.vector_stores[normalized_url].get_collection_info().get('document_count', 0)
            
            if existing_docs > 0:
                logger.info(f"✅ Repository 이미 로드됨: {normalized_url} ({existing_docs}개 문서)")
                return {
                    "success": True,
                    "message": f"Repository가 이미 로드되어 있습니다.",
                    "repository_url": normalized_url,
                    "original_url": repository_url,
                    "repository_name": repo_info.get('full_name', 'Unknown'),
                    "documents_count": existing_docs,
                    "repository_info": repo_info,
                    "cached": True
                }
            
            # 문서 추출
            documents = self.github_extractor.extract_documents(
                normalized_url,
                split_documents=True
            )
            
            if not documents:
                return {
                    "success": False,
                    "message": "추출된 문서가 없습니다.",
                    "repository_url": repository_url,
                    "documents_count": 0
                }
            
            # 벡터 스토어에 추가
            doc_ids = self.vector_stores[normalized_url].add_github_documents(normalized_url, documents)
            
            logger.info(f"Repository 추가 완료: {len(doc_ids)}개 문서")
            
            return {
                "success": True,
                "message": f"Repository가 성공적으로 추가되었습니다.",
                "repository_url": normalized_url,
                "original_url": repository_url,
                "repository_name": repo_info.get('full_name', 'Unknown'),
                "documents_count": len(doc_ids),
                "repository_info": repo_info,
                "cached": False
            }
            
        except Exception as e:
            logger.error(f"Repository 추가 실패: {e}")
            return {
                "success": False,
                "message": f"Repository 추가 실패: {str(e)}",
                "repository_url": repository_url,
                "documents_count": 0
            }
    
    def add_multiple_repositories(self, repository_urls: List[str]) -> Dict[str, Any]:
        """
        여러 GitHub repository 추가
        
        Args:
            repository_urls: GitHub repository URL 목록
            
        Returns:
            Dict[str, Any]: 처리 결과
        """
        try:
            logger.info(f"여러 Repository 추가 중: {len(repository_urls)}개")
            
            results = []
            success_count = 0
            total_documents = 0
            
            for url in repository_urls:
                result = self.add_github_repository(url)
                results.append(result)
                
                if result["success"]:
                    success_count += 1
                    total_documents += result["documents_count"]
            
            logger.info(f"여러 Repository 추가 완료: {success_count}/{len(repository_urls)} 성공")
            
            return {
                "success": success_count > 0,
                "message": f"{success_count}개 Repository가 성공적으로 추가되었습니다.",
                "total_repositories": len(repository_urls),
                "success_count": success_count,
                "total_documents": total_documents,
                "results": results
            }
            
        except Exception as e:
            logger.error(f"여러 Repository 추가 실패: {e}")
            return {
                "success": False,
                "message": f"여러 Repository 추가 실패: {str(e)}",
                "total_repositories": len(repository_urls),
                "success_count": 0,
                "total_documents": 0,
                "results": []
            }
    
    def load_configured_repositories(self) -> Dict[str, Any]:
        """
        설정된 repository들을 자동으로 로드
        
        Returns:
            Dict[str, Any]: 처리 결과
        """
        try:
            repositories = self.config.github_repositories
            
            if not repositories:
                return {
                    "success": False,
                    "message": "설정된 repository가 없습니다.",
                    "repositories": []
                }
            
            logger.info(f"설정된 Repository 로드 중: {len(repositories)}개")
            
            return self.add_multiple_repositories(repositories)
            
        except Exception as e:
            logger.error(f"설정된 Repository 로드 실패: {e}")
            return {
                "success": False,
                "message": f"설정된 Repository 로드 실패: {str(e)}",
                "repositories": []
            }
    
    def chat(self, question: str, session_id: str = "default") -> Dict[str, Any]:
        """
        챗봇과 대화
        
        Args:
            question: 사용자 질문
            session_id: 세션 ID
            
        Returns:
            Dict[str, Any]: 답변 결과
        """
        try:
            logger.info(f"질문 처리 중: {question}")
            
            # 현재 벡터 스토어가 설정되어 있는지 확인
            if not self.current_vector_store or not self.workflow:
                return {
                    "success": False,
                    "question": question,
                    "answer": "Repository가 선택되지 않았습니다. 먼저 사용할 Repository를 선택해주세요.",
                    "search_source": "error",
                    "relevance_score": 0.0,
                    "retry_count": 0,
                    "documents_used": 0,
                    "timestamp": datetime.now().isoformat(),
                    "error_message": "No repository selected",
                    "similar_questions": []
                }
            
            # 질문 처리
            result = self.workflow.process_question(question, session_id)
            
            # 대화 기록에 추가
            conversation_entry = {
                "timestamp": datetime.now().isoformat(),
                "question": question,
                "answer": result["answer"],
                "search_source": result["search_source"],
                "relevance_score": result["relevance_score"],
                "retry_count": result["retry_count"],
                "documents_used": result["documents_used"],
                "session_id": session_id,
                "answer_quality_score": result.get("answer_quality_score", 0.0),
                "github_issue_suggestion": result.get("github_issue_suggestion")
            }
            
            self.conversation_history.append(conversation_entry)
            
            # 최근 대화 기록 유지 (최대 100개)
            if len(self.conversation_history) > 100:
                self.conversation_history = self.conversation_history[-100:]
            
            logger.info("질문 처리 완료")
            
            return {
                "success": True,
                "question": question,
                "answer": result["answer"],
                "search_source": result["search_source"],
                "relevance_score": result["relevance_score"],
                "retry_count": result["retry_count"],
                "documents_used": result["documents_used"],
                "timestamp": conversation_entry["timestamp"],
                "error_message": result.get("error_message", ""),
                "similar_questions": result.get("similar_questions", []),
                "answer_quality_score": result.get("answer_quality_score", 0.0),
                "github_issue_suggestion": result.get("github_issue_suggestion")
            }
            
        except Exception as e:
            logger.error(f"질문 처리 실패: {e}")
            
            # GitHub Issue 제안 생성
            issue_suggestion = None
            if self.github_issue_helper:
                try:
                    system_info = self.get_system_info()
                    issue_suggestion = self.github_issue_helper.suggest_issue_creation(
                        question=question,
                        error_message=str(e),
                        system_info=system_info
                    )
                except Exception as issue_error:
                    logger.error(f"GitHub Issue 제안 생성 실패: {issue_error}")
            
            return {
                "success": False,
                "question": question,
                "answer": f"질문 처리 중 오류가 발생했습니다: {str(e)}",
                "search_source": "error",
                "relevance_score": 0.0,
                "retry_count": 0,
                "documents_used": 0,
                "timestamp": datetime.now().isoformat(),
                "error_message": str(e),
                "similar_questions": [],
                "github_issue_suggestion": issue_suggestion
            }
    
    def get_conversation_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        대화 기록 조회
        
        Args:
            limit: 조회할 대화 수
            
        Returns:
            List[Dict[str, Any]]: 대화 기록
        """
        return self.conversation_history[-limit:] if self.conversation_history else []
    
    def clear_conversation_history(self):
        """대화 기록 초기화"""
        self.conversation_history = []
        logger.info("대화 기록 초기화 완료")
    
    def get_chat_history(self, session_id: str = "default", limit: int = 50) -> List[Dict[str, Any]]:
        """
        채팅 히스토리 조회
        
        Args:
            session_id: 세션 ID
            limit: 최대 조회 수
            
        Returns:
            List[Dict[str, Any]]: 채팅 히스토리
        """
        # conversation_history에서 해당 세션의 메시지만 필터링
        session_messages = [
            msg for msg in self.conversation_history 
            if msg.get('session_id') == session_id
        ]
        
        # 디버깅: GitHub Issue 제안이 있는 메시지 확인
        for i, msg in enumerate(session_messages):
            if msg.get('github_issue_suggestion'):
                logger.info(f"메시지 {i}에 GitHub Issue 제안 있음: {msg.get('github_issue_suggestion', {}).get('suggested', False)}")
        
        # 최신 순으로 정렬하고 limit만큼 반환
        return session_messages[-limit:] if session_messages else []
    
    def get_similar_questions(self, question: str, session_id: str = "default", k: int = 3) -> List[Dict[str, Any]]:
        """
        유사한 질문 검색
        
        Args:
            question: 검색할 질문
            session_id: 세션 ID
            k: 반환할 결과 수
            
        Returns:
            List[Dict[str, Any]]: 유사한 질문 목록
        """
        if not self.chat_history_manager:
            return []
        
        return self.chat_history_manager.search_similar_questions(question, k, session_id)
    
    def get_all_sessions(self) -> List[str]:
        """
        모든 세션 ID 목록 조회
        
        Returns:
            List[str]: 세션 ID 목록
        """
        if not self.chat_history_manager:
            return []
        
        return self.chat_history_manager.get_all_sessions()
    
    def delete_session(self, session_id: str) -> bool:
        """
        특정 세션의 모든 메시지 삭제
        
        Args:
            session_id: 삭제할 세션 ID
            
        Returns:
            bool: 삭제 성공 여부
        """
        if not self.chat_history_manager:
            return False
        
        return self.chat_history_manager.delete_session(session_id)
    
    def get_chat_history_stats(self) -> Dict[str, Any]:
        """
        채팅 히스토리 통계 정보 조회
        
        Returns:
            Dict[str, Any]: 통계 정보
        """
        if not self.chat_history_manager:
            return {"error": "채팅 히스토리 매니저가 없습니다."}
        
        return self.chat_history_manager.get_collection_stats()
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        시스템 정보 조회
        
        Returns:
            Dict[str, Any]: 시스템 정보
        """
        try:
            # 현재 벡터 스토어 정보
            if self.current_vector_store:
                vector_store_info = self.current_vector_store.get_collection_info()
            else:
                vector_store_info = {
                    'collection_name': 'No repository selected',
                    'document_count': 0,
                    'persist_directory': self.persist_directory
                }
            
            # 워크플로우 정보
            if self.workflow:
                workflow_info = self.workflow.get_workflow_info()
            else:
                workflow_info = {
                    'model_name': 'Not initialized',
                    'max_retries': 0,
                    'relevance_threshold': 0.0
                }
            
            # 설정 정보
            config_info = {
                "model_name": self.config.default_model_name,
                "embedding_model": self.config.embedding_model,
                "max_retries": self.config.max_retries,
                "relevance_threshold": self.config.relevance_threshold,
                "max_search_results": self.config.max_search_results,
                "chunk_size": self.config.chunk_size,
                "chunk_overlap": self.config.chunk_overlap
            }
            
            # 채팅 히스토리 정보
            chat_history_info = self.get_chat_history_stats()
            
            return {
                "vector_store": vector_store_info,
                "workflow": workflow_info,
                "config": config_info,
                "chat_history": chat_history_info,
                "conversation_count": len(self.conversation_history),
                "initialized": all([
                    len(self.vector_stores) > 0,
                    self.github_extractor is not None,
                    self.workflow is not None,
                    self.chat_history_manager is not None
                ])
            }
            
        except Exception as e:
            logger.error(f"시스템 정보 조회 실패: {e}")
            return {
                "error": str(e),
                "initialized": False
            }
    
    def reset_system(self) -> bool:
        """
        시스템 초기화
        
        Returns:
            bool: 초기화 성공 여부
        """
        try:
            logger.info("시스템 초기화 중...")
            
            # 모든 벡터 스토어 초기화
            for url, vector_store in self.vector_stores.items():
                try:
                    vector_store.reset_collection()
                    logger.info(f"벡터 스토어 초기화 완료: {url}")
                except Exception as e:
                    logger.error(f"벡터 스토어 초기화 실패: {url}, {e}")
            
            # 벡터 스토어 딕셔너리 초기화
            self.vector_stores = {}
            self.current_vector_store = None
            self.workflow = None
            
            # 대화 기록 초기화
            self.clear_conversation_history()
            
            logger.info("시스템 초기화 완료")
            return True
            
        except Exception as e:
            logger.error(f"시스템 초기화 실패: {e}")
            return False


# 사용 예제
if __name__ == "__main__":
    # 챗봇 초기화
    chatbot = AIChatbot()
    
    # 시스템 정보 출력
    info = chatbot.get_system_info()
    print(f"시스템 정보: {info}")
    
    # 설정된 repository 로드
    load_result = chatbot.load_configured_repositories()
    print(f"Repository 로드 결과: {load_result}")
    
    # 테스트 대화
    questions = [
        "안녕하세요!",
        "GitHub에서 문서를 추출하는 방법은?",
        "이 시스템은 어떻게 작동하나요?"
    ]
    
    for question in questions:
        print(f"\n질문: {question}")
        result = chatbot.chat(question)
        print(f"답변: {result['answer']}")
        print(f"검색 소스: {result['search_source']}")
        print(f"관련성 점수: {result['relevance_score']:.3f}")
    
    # 대화 기록 조회
    history = chatbot.get_conversation_history(5)
    print(f"\n최근 대화 기록: {len(history)}개")
