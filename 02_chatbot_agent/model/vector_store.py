"""
Vector Store for Document Embeddings
GitHub에서 추출한 문서를 벡터화하여 저장하고 검색하는 클래스
"""

import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

import chromadb
from chromadb.config import Settings
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import get_config

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentVectorStore:
    """문서를 벡터화하여 저장하고 검색하는 클래스"""
    
    def __init__(self, 
                 collection_name: str = "github_documents",
                 persist_directory: str = "./chroma_db"):
        """
        DocumentVectorStore 초기화
        
        Args:
            collection_name: ChromaDB 컬렉션 이름
            persist_directory: 벡터 DB 저장 디렉토리
        """
        self.config = get_config()
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # 임베딩 모델 초기화
        self.embedding_model = OpenAIEmbeddings(
            model=self.config.embedding_model,
            openai_api_key=self.config.openai_api_key
        )
        
        # 텍스트 분할기 초기화
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # ChromaDB 클라이언트 초기화
        self._init_chroma_client()
        
        # 벡터 스토어 초기화
        self.vector_store = self._init_vector_store()
    
    def _init_chroma_client(self):
        """ChromaDB 클라이언트 초기화"""
        try:
            # 디렉토리 생성
            os.makedirs(self.persist_directory, exist_ok=True)
            
            # ChromaDB 클라이언트 설정
            self.chroma_client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            logger.info(f"ChromaDB 클라이언트 초기화 완료: {self.persist_directory}")
            
        except Exception as e:
            logger.error(f"ChromaDB 클라이언트 초기화 실패: {e}")
            raise
    
    def _init_vector_store(self) -> Chroma:
        """벡터 스토어 초기화"""
        try:
            # 기존 컬렉션이 있는지 확인
            existing_collections = [col.name for col in self.chroma_client.list_collections()]
            
            if self.collection_name in existing_collections:
                logger.info(f"기존 컬렉션 사용: {self.collection_name}")
                vector_store = Chroma(
                    client=self.chroma_client,
                    collection_name=self.collection_name,
                    embedding_function=self.embedding_model
                )
            else:
                logger.info(f"새 컬렉션 생성: {self.collection_name}")
                vector_store = Chroma(
                    client=self.chroma_client,
                    collection_name=self.collection_name,
                    embedding_function=self.embedding_model,
                    persist_directory=self.persist_directory
                )
            
            return vector_store
            
        except Exception as e:
            logger.error(f"벡터 스토어 초기화 실패: {e}")
            raise
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        문서를 벡터 스토어에 추가
        
        Args:
            documents: 추가할 문서 목록
            
        Returns:
            List[str]: 추가된 문서의 ID 목록
        """
        try:
            if not documents:
                logger.warning("추가할 문서가 없습니다.")
                return []
            
            # 문서 분할
            split_documents = self.text_splitter.split_documents(documents)
            logger.info(f"문서 분할 완료: {len(split_documents)}개 청크")
            
            # 벡터 스토어에 추가
            doc_ids = self.vector_store.add_documents(split_documents)
            logger.info(f"문서 추가 완료: {len(doc_ids)}개 청크")
            
            return doc_ids
            
        except Exception as e:
            logger.error(f"문서 추가 실패: {e}")
            raise
    
    def add_github_documents(self, repository_url: str, documents: List[Document]) -> List[str]:
        """
        GitHub repository에서 추출한 문서를 벡터 스토어에 추가
        
        Args:
            repository_url: GitHub repository URL
            documents: 추출된 문서 목록
            
        Returns:
            List[str]: 추가된 문서의 ID 목록
        """
        try:
            # 메타데이터에 repository 정보 추가
            for doc in documents:
                doc.metadata.update({
                    'source_type': 'github',
                    'repository_url': repository_url
                })
            
            return self.add_documents(documents)
            
        except Exception as e:
            logger.error(f"GitHub 문서 추가 실패: {e}")
            raise
    
    def similarity_search(self, 
                         query: str, 
                         k: int = 5,
                         filter_dict: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        유사도 검색 수행
        
        Args:
            query: 검색 쿼리
            k: 반환할 문서 수
            filter_dict: 필터 조건
            
        Returns:
            List[Document]: 검색된 문서 목록
        """
        try:
            # 유사도 검색 수행
            results = self.vector_store.similarity_search(
                query=query,
                k=k,
                filter=filter_dict
            )
            
            logger.info(f"유사도 검색 완료: {len(results)}개 문서")
            return results
            
        except Exception as e:
            logger.error(f"유사도 검색 실패: {e}")
            return []
    
    def similarity_search_with_score(self, 
                                   query: str, 
                                   k: int = 5,
                                   filter_dict: Optional[Dict[str, Any]] = None) -> List[tuple]:
        """
        유사도 점수와 함께 검색 수행
        
        Args:
            query: 검색 쿼리
            k: 반환할 문서 수
            filter_dict: 필터 조건
            
        Returns:
            List[tuple]: (문서, 점수) 튜플 목록
        """
        try:
            # 유사도 검색 수행 (점수 포함)
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter_dict
            )
            
            logger.info(f"유사도 검색 완료 (점수 포함): {len(results)}개 문서")
            return results
            
        except Exception as e:
            logger.error(f"유사도 검색 실패: {e}")
            return []
    
    def get_repository_documents(self, repository_url: str) -> List[Document]:
        """
        특정 repository의 모든 문서 조회
        
        Args:
            repository_url: GitHub repository URL
            
        Returns:
            List[Document]: 해당 repository의 문서 목록
        """
        try:
            # repository 필터로 검색
            filter_dict = {
                'repository_url': repository_url
            }
            
            # 모든 문서 검색 (k를 크게 설정)
            results = self.similarity_search(
                query="",  # 빈 쿼리로 모든 문서 검색
                k=1000,    # 충분히 큰 수
                filter_dict=filter_dict
            )
            
            logger.info(f"Repository 문서 조회 완료: {len(results)}개 문서")
            return results
            
        except Exception as e:
            logger.error(f"Repository 문서 조회 실패: {e}")
            return []
    
    def delete_repository_documents(self, repository_url: str) -> bool:
        """
        특정 repository의 모든 문서 삭제
        
        Args:
            repository_url: GitHub repository URL
            
        Returns:
            bool: 삭제 성공 여부
        """
        try:
            # 해당 repository의 문서 ID 조회
            documents = self.get_repository_documents(repository_url)
            
            if not documents:
                logger.info(f"삭제할 문서가 없습니다: {repository_url}")
                return True
            
            # 문서 ID 추출 (ChromaDB의 경우 메타데이터에서 ID 확인)
            doc_ids = []
            for doc in documents:
                if 'id' in doc.metadata:
                    doc_ids.append(doc.metadata['id'])
            
            if doc_ids:
                # 벡터 스토어에서 삭제
                self.vector_store.delete(doc_ids)
                logger.info(f"Repository 문서 삭제 완료: {len(doc_ids)}개 문서")
            
            return True
            
        except Exception as e:
            logger.error(f"Repository 문서 삭제 실패: {e}")
            return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        컬렉션 정보 조회
        
        Returns:
            Dict[str, Any]: 컬렉션 정보
        """
        try:
            collection = self.chroma_client.get_collection(self.collection_name)
            count = collection.count()
            
            return {
                'collection_name': self.collection_name,
                'document_count': count,
                'persist_directory': self.persist_directory
            }
            
        except Exception as e:
            logger.error(f"컬렉션 정보 조회 실패: {e}")
            return {}
    
    def reset_collection(self) -> bool:
        """
        컬렉션 초기화
        
        Returns:
            bool: 초기화 성공 여부
        """
        try:
            # 기존 컬렉션 삭제
            try:
                self.chroma_client.delete_collection(self.collection_name)
            except:
                pass  # 컬렉션이 없는 경우 무시
            
            # 새 벡터 스토어 초기화
            self.vector_store = self._init_vector_store()
            
            logger.info(f"컬렉션 초기화 완료: {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"컬렉션 초기화 실패: {e}")
            return False


# 사용 예제
if __name__ == "__main__":
    # 벡터 스토어 초기화
    vector_store = DocumentVectorStore()
    
    # 컬렉션 정보 출력
    info = vector_store.get_collection_info()
    print(f"벡터 스토어 정보: {info}")
    
    # 테스트 검색
    results = vector_store.similarity_search("test query", k=3)
    print(f"검색 결과: {len(results)}개 문서")
