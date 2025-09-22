"""
GitHub Repository Document Extractor
GitHub repository에서 문서를 다운로드하고 추출하는 클래스
"""

import os
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Union
from urllib.parse import urlparse
import requests
from git import Repo
import logging

from langchain.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader,
    UnstructuredWordDocumentLoader,
    DirectoryLoader
)
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 로깅 설정
logger = logging.getLogger(__name__)


class GitHubDocumentExtractor:
    """GitHub repository에서 문서를 추출하는 클래스"""
    
    def __init__(self, 
                 github_token: Optional[str] = None,
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 supported_extensions: List[str] = None):
        """
        GitHubDocumentExtractor 초기화
        
        Args:
            github_token: GitHub Personal Access Token (private repo 접근용)
            max_file_size: 최대 파일 크기 (bytes)
            supported_extensions: 지원하는 파일 확장자 목록
        """
        self.github_token = github_token
        self.max_file_size = max_file_size
        self.supported_extensions = supported_extensions or [
            '.md', '.txt', '.py', '.js', '.ts', '.html', '.htm', 
            '.pdf', '.doc', '.docx', '.rst', '.json', '.yaml', '.yml',
            '.xml', '.csv', '.sql', '.sh', '.bat', '.ps1'
        ]
        
        # 텍스트 분할기 설정
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def _normalize_github_url(self, url: str) -> str:
        """
        GitHub URL을 정규화 (tree 경로 제거)
        
        Args:
            url: GitHub repository URL
            
        Returns:
            str: 정규화된 URL
        """
        # tree 경로가 있는 경우 제거
        if '/tree/' in url:
            url = url.split('/tree/')[0]
        
        # .git이 없는 경우 추가
        if not url.endswith('.git'):
            url = f"{url}.git"
        
        return url
    
    def _parse_github_url(self, url: str) -> Dict[str, str]:
        """
        GitHub URL을 파싱하여 owner, repo, branch 정보 추출
        
        Args:
            url: GitHub repository URL
            
        Returns:
            Dict: {'owner': str, 'repo': str, 'branch': str}
        """
        # URL 정규화
        normalized_url = self._normalize_github_url(url)
        parsed = urlparse(normalized_url)
        
        if parsed.hostname not in ['github.com', 'www.github.com']:
            raise ValueError(f"지원하지 않는 GitHub URL입니다: {url}")
        
        path_parts = parsed.path.strip('/').split('/')
        if len(path_parts) < 2:
            raise ValueError(f"올바르지 않은 GitHub URL 형식입니다: {url}")
        
        owner = path_parts[0]
        repo = path_parts[1].replace('.git', '')
        branch = 'main'  # 기본값
        
        # 원본 URL에서 branch 정보 추출 (tree 경로가 있는 경우)
        if '/tree/' in url:
            try:
                tree_parts = url.split('/tree/')
                if len(tree_parts) > 1:
                    branch = tree_parts[1].split('/')[0]
            except:
                pass  # branch 추출 실패 시 기본값 사용
        
        return {
            'owner': owner,
            'repo': repo,
            'branch': branch
        }
    
    def _clone_repository(self, url: str, target_dir: str) -> str:
        """
        GitHub repository를 로컬에 클론
        
        Args:
            url: GitHub repository URL
            target_dir: 클론할 대상 디렉토리
            
        Returns:
            str: 클론된 repository 경로
        """
        try:
            # URL 정규화
            normalized_url = self._normalize_github_url(url)
            
            # Private repository의 경우 토큰 사용
            clone_url = normalized_url
            if self.github_token:
                # HTTPS URL에 토큰 추가
                if normalized_url.startswith('https://'):
                    clone_url = normalized_url.replace('https://', f'https://{self.github_token}@')
                elif normalized_url.startswith('git@'):
                    # SSH URL은 그대로 사용 (SSH 키 설정 필요)
                    pass
            
            logger.info(f"Repository 클론 중: {clone_url}")
            repo = Repo.clone_from(clone_url, target_dir)
            
            # 특정 브랜치로 체크아웃 (기본 브랜치가 아닌 경우)
            repo_info = self._parse_github_url(url)
            if repo_info['branch'] != 'main':
                try:
                    repo.git.checkout(repo_info['branch'])
                except Exception as e:
                    logger.warning(f"브랜치 체크아웃 실패, 기본 브랜치 사용: {e}")
            
            logger.info(f"Repository 클론 완료: {target_dir}")
            return target_dir
            
        except Exception as e:
            logger.error(f"Repository 클론 실패: {e}")
            raise
    
    def _is_supported_file(self, file_path: Path) -> bool:
        """
        파일이 지원되는 확장자인지 확인
        
        Args:
            file_path: 파일 경로
            
        Returns:
            bool: 지원되는 파일이면 True
        """
        if not file_path.is_file():
            return False
        
        # 파일 크기 확인
        if file_path.stat().st_size > self.max_file_size:
            logger.warning(f"파일 크기가 너무 큼 (건너뜀): {file_path}")
            return False
        
        # 확장자 확인
        return file_path.suffix.lower() in self.supported_extensions
    
    def _load_document(self, file_path: Path) -> List[Document]:
        """
        단일 파일을 로드하여 Document 객체로 변환
        
        Args:
            file_path: 파일 경로
            
        Returns:
            List[Document]: 로드된 문서 목록
        """
        try:
            file_extension = file_path.suffix.lower()
            
            # 파일 확장자에 따른 로더 선택
            if file_extension in ['.md', '.rst']:
                loader = UnstructuredMarkdownLoader(str(file_path))
            elif file_extension in ['.html', '.htm']:
                loader = UnstructuredHTMLLoader(str(file_path))
            elif file_extension == '.pdf':
                loader = PyPDFLoader(str(file_path))
            elif file_extension in ['.doc', '.docx']:
                loader = UnstructuredWordDocumentLoader(str(file_path))
            else:
                # 기본적으로 텍스트 파일로 처리
                loader = TextLoader(str(file_path), encoding='utf-8')
            
            documents = loader.load()
            
            # 메타데이터 추가
            for doc in documents:
                doc.metadata.update({
                    'source': str(file_path),
                    'file_type': file_extension,
                    'file_name': file_path.name,
                    'file_size': file_path.stat().st_size
                })
            
            return documents
            
        except Exception as e:
            logger.error(f"파일 로드 실패: {file_path}, 오류: {e}")
            return []
    
    def _load_directory(self, directory_path: Path) -> List[Document]:
        """
        디렉토리 내의 모든 지원되는 파일을 로드
        
        Args:
            directory_path: 디렉토리 경로
            
        Returns:
            List[Document]: 로드된 문서 목록
        """
        all_documents = []
        
        # 지원되는 파일들만 필터링
        supported_files = [
            f for f in directory_path.rglob('*')
            if self._is_supported_file(f)
        ]
        
        logger.info(f"로드할 파일 수: {len(supported_files)}")
        
        for file_path in supported_files:
            try:
                documents = self._load_document(file_path)
                all_documents.extend(documents)
                logger.info(f"로드 완료: {file_path.name}")
            except Exception as e:
                logger.error(f"파일 로드 실패: {file_path}, 오류: {e}")
                continue
        
        return all_documents
    
    def extract_documents(self, 
                         repository_url: str,
                         target_directory: Optional[str] = None,
                         split_documents: bool = True) -> List[Document]:
        """
        GitHub repository에서 문서를 추출
        
        Args:
            repository_url: GitHub repository URL
            target_directory: 임시 디렉토리 경로 (None이면 자동 생성)
            split_documents: 문서를 청크로 분할할지 여부
            
        Returns:
            List[Document]: 추출된 문서 목록
        """
        temp_dir = None
        
        try:
            # 임시 디렉토리 생성
            if target_directory is None:
                temp_dir = tempfile.mkdtemp(prefix="github_extract_")
                target_directory = temp_dir
            else:
                os.makedirs(target_directory, exist_ok=True)
            
            # Repository 클론
            repo_path = self._clone_repository(repository_url, target_directory)
            
            # 문서 로드
            documents = self._load_directory(Path(repo_path))
            
            # 문서 분할 (옵션)
            if split_documents and documents:
                logger.info("문서를 청크로 분할 중...")
                documents = self.text_splitter.split_documents(documents)
                logger.info(f"분할된 청크 수: {len(documents)}")
            
            # Repository 정보를 메타데이터에 추가
            repo_info = self._parse_github_url(repository_url)
            for doc in documents:
                doc.metadata.update({
                    'repository_url': repository_url,
                    'repository_owner': repo_info['owner'],
                    'repository_name': repo_info['repo'],
                    'repository_branch': repo_info['branch']
                })
            
            logger.info(f"문서 추출 완료: {len(documents)}개 문서")
            return documents
            
        except Exception as e:
            logger.error(f"문서 추출 실패: {e}")
            raise
        
        finally:
            # 임시 디렉토리 정리
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                logger.info("임시 디렉토리 정리 완료")
    
    def extract_multiple_repositories(self, 
                                    repository_urls: List[str],
                                    split_documents: bool = True) -> Dict[str, List[Document]]:
        """
        여러 GitHub repository에서 문서를 추출
        
        Args:
            repository_urls: GitHub repository URL 목록
            split_documents: 문서를 청크로 분할할지 여부
            
        Returns:
            Dict[str, List[Document]]: repository URL을 키로 하는 문서 목록
        """
        results = {}
        
        for url in repository_urls:
            try:
                logger.info(f"Repository 처리 중: {url}")
                documents = self.extract_documents(url, split_documents=split_documents)
                results[url] = documents
                logger.info(f"완료: {url} ({len(documents)}개 문서)")
            except Exception as e:
                logger.error(f"Repository 처리 실패: {url}, 오류: {e}")
                results[url] = []
        
        return results
    
    def get_repository_info(self, repository_url: str) -> Dict[str, str]:
        """
        GitHub repository 정보 조회
        
        Args:
            repository_url: GitHub repository URL
            
        Returns:
            Dict[str, str]: repository 정보
        """
        try:
            repo_info = self._parse_github_url(repository_url)
            
            # GitHub API를 통한 추가 정보 조회
            api_url = f"https://api.github.com/repos/{repo_info['owner']}/{repo_info['repo']}"
            headers = {}
            
            if self.github_token:
                headers['Authorization'] = f'token {self.github_token}'
            
            response = requests.get(api_url, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'name': data.get('name', ''),
                    'full_name': data.get('full_name', ''),
                    'description': data.get('description', ''),
                    'language': data.get('language', ''),
                    'stars': data.get('stargazers_count', 0),
                    'forks': data.get('forks_count', 0),
                    'created_at': data.get('created_at', ''),
                    'updated_at': data.get('updated_at', ''),
                    'clone_url': data.get('clone_url', ''),
                    'default_branch': data.get('default_branch', 'main')
                }
            else:
                logger.warning(f"GitHub API 호출 실패: {response.status_code}")
                return repo_info
                
        except Exception as e:
            logger.error(f"Repository 정보 조회 실패: {e}")
            return self._parse_github_url(repository_url)


# 사용 예제
if __name__ == "__main__":
    # GitHub 토큰 설정 (선택사항)
    github_token = os.getenv("GITHUB_TOKEN")
    
    # Extractor 초기화
    extractor = GitHubDocumentExtractor(github_token=github_token)
    
    # 단일 repository 추출
    repository_url = "https://github.com/example/repository"
    
    try:
        documents = extractor.extract_documents(repository_url)
        print(f"추출된 문서 수: {len(documents)}")
        
        # 첫 번째 문서 정보 출력
        if documents:
            doc = documents[0]
            print(f"첫 번째 문서: {doc.metadata.get('file_name', 'Unknown')}")
            print(f"내용 미리보기: {doc.page_content[:200]}...")
    
    except Exception as e:
        print(f"오류 발생: {e}")
