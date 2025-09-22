"""
채팅 히스토리 관리 모듈
ChromaDB를 사용하여 질문-답변 쌍을 저장하고 유사한 질문을 검색하는 기능
"""

import logging
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document

from config import get_config

# 로깅 설정
logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """채팅 메시지 데이터 클래스"""
    question: str
    answer: str
    timestamp: datetime
    session_id: str
    relevance_score: float = 0.0
    search_source: str = "db"
    documents_used: int = 0


class ChatHistoryManager:
    """채팅 히스토리 관리 클래스"""
    
    def __init__(self, collection_name: str = "chat_history"):
        """
        ChatHistoryManager 초기화
        
        Args:
            collection_name: ChromaDB 컬렉션 이름
        """
        self.config = get_config()
        self.collection_name = collection_name
        
        # ChromaDB 클라이언트 초기화
        self.client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # 임베딩 모델 초기화
        self.embedding_model = OpenAIEmbeddings(
            openai_api_key=self.config.openai_api_key,
            model=self.config.embedding_model
        )
        
        # 컬렉션 초기화
        self.collection = self._init_collection()
        
        logger.info(f"채팅 히스토리 매니저 초기화 완료: {collection_name}")
    
    def _init_collection(self) -> chromadb.Collection:
        """ChromaDB 컬렉션 초기화"""
        try:
            # 기존 컬렉션 가져오기 또는 새로 생성
            collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "채팅 히스토리 저장소"}
            )
            logger.info(f"컬렉션 '{self.collection_name}' 초기화 완료")
            return collection
            
        except Exception as e:
            logger.error(f"컬렉션 초기화 실패: {e}")
            raise
    
    def add_chat_message(self, 
                        question: str, 
                        answer: str, 
                        session_id: str,
                        relevance_score: float = 0.0,
                        search_source: str = "db",
                        documents_used: int = 0) -> str:
        """
        채팅 메시지 추가 (크기 제한 적용)
        
        Args:
            question: 사용자 질문
            answer: AI 답변
            session_id: 세션 ID
            relevance_score: 관련성 점수
            search_source: 검색 소스
            documents_used: 사용된 문서 수
            
        Returns:
            str: 메시지 ID
        """
        try:
            # 메시지 ID 생성
            message_id = str(uuid.uuid4())
            timestamp = datetime.now()
            
            # 질문과 답변을 결합하여 임베딩용 텍스트 생성
            combined_text = f"질문: {question}\n답변: {answer}"
            
            # 임베딩 생성
            embedding = self.embedding_model.embed_query(combined_text)
            
            # 메타데이터 준비
            metadata = {
                "question": question,
                "answer": answer,
                "timestamp": timestamp.isoformat(),
                "session_id": session_id,
                "relevance_score": relevance_score,
                "search_source": search_source,
                "documents_used": documents_used,
                "message_id": message_id
            }
            
            # 크기 제한 확인 및 교체 로직 실행
            self._check_and_manage_size(combined_text, metadata, embedding)
            
            logger.info(f"채팅 메시지 저장 완료: {message_id}")
            return message_id
            
        except Exception as e:
            logger.error(f"채팅 메시지 저장 실패: {e}")
            raise
    
    def search_similar_questions(self, 
                                question: str, 
                                k: int = 5,  # 기본 검색 결과 수 증가 (3 -> 5)
                                session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        유사한 질문 검색
        
        Args:
            question: 검색할 질문
            k: 반환할 결과 수
            session_id: 특정 세션의 메시지만 검색 (None이면 모든 세션)
            
        Returns:
            List[Dict[str, Any]]: 유사한 질문과 답변 목록
        """
        try:
            # 검색 쿼리 준비
            where_clause = {}
            if session_id:
                where_clause["session_id"] = session_id
            
            # 유사도 검색 실행
            results = self.collection.query(
                query_texts=[question],
                n_results=k,
                where=where_clause if where_clause else None
            )
            
            # 결과 변환
            similar_questions = []
            if results['ids'] and results['ids'][0]:
                for i, message_id in enumerate(results['ids'][0]):
                    metadata = results['metadatas'][0][i]
                    distance = results['distances'][0][i] if results['distances'] else 0.0
                    
                    similar_questions.append({
                        "message_id": message_id,
                        "question": metadata["question"],
                        "answer": metadata["answer"],
                        "timestamp": metadata["timestamp"],
                        "session_id": metadata["session_id"],
                        "relevance_score": metadata["relevance_score"],
                        "search_source": metadata["search_source"],
                        "documents_used": metadata["documents_used"],
                        "similarity_score": 1.0 - distance,  # 거리를 유사도로 변환
                        "distance": distance
                    })
            
            logger.info(f"유사한 질문 검색 완료: {len(similar_questions)}개")
            return similar_questions
            
        except Exception as e:
            logger.error(f"유사한 질문 검색 실패: {e}")
            return []
    
    def get_chat_history(self, 
                        session_id: str, 
                        limit: int = 50) -> List[Dict[str, Any]]:
        """
        특정 세션의 채팅 히스토리 조회
        
        Args:
            session_id: 세션 ID
            limit: 최대 조회 수
            
        Returns:
            List[Dict[str, Any]]: 채팅 히스토리 목록
        """
        try:
            # 세션별 메시지 조회
            results = self.collection.get(
                where={"session_id": session_id},
                limit=limit
            )
            
            # 결과 변환 및 정렬
            chat_history = []
            if results['ids']:
                for i, message_id in enumerate(results['ids']):
                    metadata = results['metadatas'][i]
                    
                    chat_history.append({
                        "message_id": message_id,
                        "question": metadata["question"],
                        "answer": metadata["answer"],
                        "timestamp": metadata["timestamp"],
                        "session_id": metadata["session_id"],
                        "relevance_score": metadata["relevance_score"],
                        "search_source": metadata["search_source"],
                        "documents_used": metadata["documents_used"]
                    })
                
                # 시간순 정렬 (오래된 것부터)
                chat_history.sort(key=lambda x: x["timestamp"])
            
            logger.info(f"채팅 히스토리 조회 완료: {len(chat_history)}개")
            return chat_history
            
        except Exception as e:
            logger.error(f"채팅 히스토리 조회 실패: {e}")
            return []
    
    def get_all_sessions(self) -> List[str]:
        """
        모든 세션 ID 목록 조회
        
        Returns:
            List[str]: 세션 ID 목록
        """
        try:
            # 모든 메시지 조회
            results = self.collection.get()
            
            # 고유한 세션 ID 추출
            session_ids = set()
            if results['metadatas']:
                for metadata in results['metadatas']:
                    session_ids.add(metadata["session_id"])
            
            session_list = list(session_ids)
            logger.info(f"세션 목록 조회 완료: {len(session_list)}개")
            return session_list
            
        except Exception as e:
            logger.error(f"세션 목록 조회 실패: {e}")
            return []
    
    def delete_session(self, session_id: str) -> bool:
        """
        특정 세션의 모든 메시지 삭제
        
        Args:
            session_id: 삭제할 세션 ID
            
        Returns:
            bool: 삭제 성공 여부
        """
        try:
            # 세션의 모든 메시지 ID 조회
            results = self.collection.get(
                where={"session_id": session_id}
            )
            
            if results['ids']:
                # 메시지 삭제
                self.collection.delete(ids=results['ids'])
                logger.info(f"세션 '{session_id}' 삭제 완료: {len(results['ids'])}개 메시지")
                return True
            else:
                logger.info(f"삭제할 세션 '{session_id}' 없음")
                return True
                
        except Exception as e:
            logger.error(f"세션 삭제 실패: {e}")
            return False
    
    def _check_and_manage_size(self, combined_text: str, metadata: dict, embedding: list) -> None:
        """
        크기 제한 확인 및 교체 로직 실행
        
        Args:
            combined_text: 저장할 텍스트
            metadata: 메타데이터
            embedding: 임베딩 벡터
        """
        try:
            # 현재 컬렉션 크기 확인
            current_size = self._get_collection_size_bytes()
            new_message_size = len(combined_text.encode('utf-8'))
            max_size = self.config.chat_history_max_size
            
            logger.info(f"현재 컬렉션 크기: {current_size / (1024*1024):.2f}MB, 새 메시지 크기: {new_message_size / 1024:.2f}KB, 최대 크기: {max_size / (1024*1024*1024):.2f}GB")
            
            # 크기 초과 시 교체 로직 실행
            if current_size + new_message_size > max_size:
                logger.info(f"크기 초과 예상 ({current_size + new_message_size} > {max_size}) - 교체 로직 실행")
                self._replace_oldest_messages(combined_text, metadata, embedding)
            else:
                # 일반적인 추가
                self.collection.add(
                    ids=[metadata["message_id"]],
                    embeddings=[embedding],
                    documents=[combined_text],
                    metadatas=[metadata]
                )
                
        except Exception as e:
            logger.error(f"크기 관리 실패: {e}")
            # 실패 시에도 메시지는 저장
            self.collection.add(
                ids=[metadata["message_id"]],
                embeddings=[embedding],
                documents=[combined_text],
                metadatas=[metadata]
            )
    
    def _get_collection_size_bytes(self) -> int:
        """
        현재 컬렉션의 총 크기 (bytes) 계산
        
        Returns:
            int: 총 크기 (bytes)
        """
        try:
            # 모든 메시지 조회
            results = self.collection.get()
            
            total_size = 0
            if results['documents']:
                for document in results['documents']:
                    total_size += len(document.encode('utf-8'))
            
            return total_size
            
        except Exception as e:
            logger.error(f"컬렉션 크기 계산 실패: {e}")
            return 0
    
    def _replace_oldest_messages(self, new_text: str, new_metadata: dict, new_embedding: list) -> None:
        """
        가장 오래된 메시지들을 새 메시지로 교체
        
        Args:
            new_text: 새 메시지 텍스트
            new_metadata: 새 메시지 메타데이터
            new_embedding: 새 메시지 임베딩
        """
        try:
            # 모든 메시지 조회 (시간순 정렬)
            results = self.collection.get()
            
            if not results['ids']:
                # 기존 메시지가 없으면 새 메시지만 추가
                self.collection.add(
                    ids=[new_metadata["message_id"]],
                    embeddings=[new_embedding],
                    documents=[new_text],
                    metadatas=[new_metadata]
                )
                return
            
            # 메시지들을 시간순으로 정렬
            messages_with_timestamp = []
            for i, message_id in enumerate(results['ids']):
                metadata = results['metadatas'][i]
                document = results['documents'][i]
                embedding = results['embeddings'][i]
                
                messages_with_timestamp.append({
                    'id': message_id,
                    'metadata': metadata,
                    'document': document,
                    'embedding': embedding,
                    'timestamp': metadata['timestamp']
                })
            
            # 시간순 정렬 (오래된 것부터)
            messages_with_timestamp.sort(key=lambda x: x['timestamp'])
            
            # 새 메시지 크기
            new_message_size = len(new_text.encode('utf-8'))
            
            # 교체할 메시지들 선택 (새 메시지 크기만큼)
            messages_to_remove = []
            removed_size = 0
            
            for msg in messages_with_timestamp:
                msg_size = len(msg['document'].encode('utf-8'))
                messages_to_remove.append(msg['id'])
                removed_size += msg_size
                
                # 충분한 공간이 확보되면 중단
                if removed_size >= new_message_size:
                    break
            
            # 오래된 메시지들 삭제
            if messages_to_remove:
                self.collection.delete(ids=messages_to_remove)
                logger.info(f"오래된 메시지 삭제 완료: {len(messages_to_remove)}개")
            
            # 새 메시지 추가
            self.collection.add(
                ids=[new_metadata["message_id"]],
                embeddings=[new_embedding],
                documents=[new_text],
                metadatas=[new_metadata]
            )
            
            logger.info(f"새 메시지 추가 완료: {new_metadata['message_id']}")
            
        except Exception as e:
            logger.error(f"메시지 교체 실패: {e}")
            # 실패 시에도 새 메시지는 저장
            self.collection.add(
                ids=[new_metadata["message_id"]],
                embeddings=[new_embedding],
                documents=[new_text],
                metadatas=[new_metadata]
            )
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        컬렉션 통계 정보 조회
        
        Returns:
            Dict[str, Any]: 통계 정보
        """
        try:
            # 전체 메시지 수
            total_count = self.collection.count()
            
            # 세션 수
            sessions = self.get_all_sessions()
            session_count = len(sessions)
            
            # 최근 메시지 시간
            recent_results = self.collection.get(limit=1)
            latest_timestamp = None
            if recent_results['metadatas']:
                latest_timestamp = recent_results['metadatas'][0]["timestamp"]
            
            # 컬렉션 크기
            collection_size = self._get_collection_size_bytes()
            max_size = self.config.chat_history_max_size
            
            return {
                "total_messages": total_count,
                "total_sessions": session_count,
                "latest_message": latest_timestamp,
                "collection_name": self.collection_name,
                "collection_size_mb": round(collection_size / (1024*1024), 2),
                "max_size_gb": round(max_size / (1024*1024*1024), 2),
                "usage_percentage": round((collection_size / max_size) * 100, 2) if max_size > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"통계 정보 조회 실패: {e}")
            return {
                "total_messages": 0,
                "total_sessions": 0,
                "latest_message": None,
                "collection_name": self.collection_name,
                "error": str(e)
            }


# 사용 예제
if __name__ == "__main__":
    # 채팅 히스토리 매니저 초기화
    history_manager = ChatHistoryManager()
    
    # 테스트 메시지 추가
    session_id = "test_session_001"
    
    message_id = history_manager.add_chat_message(
        question="GitHub에서 문서를 추출하는 방법은?",
        answer="GitHub에서 문서를 추출하는 방법은 GitPython과 PyGithub 라이브러리를 사용하는 것입니다.",
        session_id=session_id,
        relevance_score=0.8,
        search_source="db",
        documents_used=5
    )
    
    print(f"메시지 저장 완료: {message_id}")
    
    # 유사한 질문 검색
    similar = history_manager.search_similar_questions(
        "GitHub 문서 추출 방법",
        k=3
    )
    
    print(f"유사한 질문 검색 결과: {len(similar)}개")
    for item in similar:
        print(f"- {item['question']} (유사도: {item['similarity_score']:.3f})")
    
    # 통계 정보
    stats = history_manager.get_collection_stats()
    print(f"컬렉션 통계: {stats}")

