"""
GitHub Issue Helper
에러 발생 시 GitHub Issue 등록을 도와주는 클래스
"""

import os
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import requests
from urllib.parse import quote

from config import get_config

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GitHubIssueHelper:
    """GitHub Issue 등록을 도와주는 클래스"""
    
    def __init__(self, repository_url: str = None):
        """
        GitHubIssueHelper 초기화
        
        Args:
            repository_url: GitHub repository URL (예: https://github.com/owner/repo)
        """
        self.config = get_config()
        self.repository_url = repository_url or self._get_default_repository()
        self.github_token = self.config.github_token
        
        # Repository 정보 파싱
        self.owner, self.repo = self._parse_repository_url()
    
    def _get_default_repository(self) -> str:
        """기본 repository URL 반환"""
        repositories = self.config.github_repositories
        if repositories:
            return repositories[0]
        return "https://github.com/your-username/your-repository"
    
    def _parse_repository_url(self) -> tuple:
        """Repository URL에서 owner와 repo 추출"""
        try:
            if not self.repository_url:
                return "unknown", "unknown"
            
            # https://github.com/owner/repo 형태에서 owner와 repo 추출
            parts = self.repository_url.replace("https://github.com/", "").split("/")
            if len(parts) >= 2:
                return parts[0], parts[1].replace(".git", "")
            return "unknown", "unknown"
        except Exception as e:
            logger.error(f"Repository URL 파싱 실패: {e}")
            return "unknown", "unknown"
    
    def create_issue_url(self, 
                        title: str, 
                        body: str, 
                        labels: list = None) -> str:
        """
        GitHub Issue 생성 URL 생성
        
        Args:
            title: Issue 제목
            body: Issue 내용
            labels: 라벨 목록
            
        Returns:
            str: GitHub Issue 생성 URL
        """
        try:
            base_url = f"https://github.com/{self.owner}/{self.repo}/issues/new"
            
            # URL 파라미터 구성
            params = {
                'title': title,
                'body': body
            }
            
            if labels:
                params['labels'] = ','.join(labels)
            
            # URL 인코딩
            query_string = '&'.join([f"{k}={quote(str(v))}" for k, v in params.items()])
            issue_url = f"{base_url}?{query_string}"
            
            logger.info(f"GitHub Issue URL 생성: {issue_url}")
            return issue_url
            
        except Exception as e:
            logger.error(f"GitHub Issue URL 생성 실패: {e}")
            return f"https://github.com/{self.owner}/{self.repo}/issues/new"
    
    def create_issue_template(self, 
                            question: str, 
                            error_message: str = None,
                            system_info: Dict[str, Any] = None) -> Dict[str, str]:
        """
        GitHub Issue 템플릿 생성
        
        Args:
            question: 사용자 질문
            error_message: 에러 메시지
            system_info: 시스템 정보
            
        Returns:
            Dict[str, str]: Issue 제목과 내용
        """
        try:
            # Issue 제목 생성
            title = f"질문 답변 실패: {question[:50]}{'...' if len(question) > 50 else ''}"
            
            # Issue 내용 생성
            body = f"""## 🐛 질문 답변 실패 보고

### 📝 질문 내용
```
{question}
```

### ❌ 발생한 문제
"""
            
            if error_message:
                body += f"```\n{error_message}\n```\n\n"
            else:
                body += "질문에 대한 적절한 답변을 생성하지 못했습니다.\n\n"
            
            # 시스템 정보 추가
            if system_info:
                body += f"""### 🔧 시스템 정보
- **모델**: {system_info.get('model_name', 'Unknown')}
- **임베딩 모델**: {system_info.get('embedding_model', 'Unknown')}
- **관련성 임계값**: {system_info.get('relevance_threshold', 'Unknown')}
- **최대 재시도**: {system_info.get('max_retries', 'Unknown')}
- **문서 수**: {system_info.get('document_count', 'Unknown')}
- **채팅 히스토리 수**: {system_info.get('conversation_count', 'Unknown')}

"""
            
            # 추가 정보
            body += f"""### 📅 발생 시간
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### 🔍 재현 단계
1. 위의 질문을 입력
2. 시스템이 적절한 답변을 생성하지 못함

### 💡 예상되는 원인
- 관련 문서가 벡터 스토어에 없음
- 질문이 너무 구체적이거나 모호함
- 시스템 설정 문제
- API 호출 오류

### 🎯 개선 제안
- [ ] 관련 문서 추가 필요
- [ ] 질문 재구성 필요
- [ ] 시스템 설정 조정 필요
- [ ] 기타: ___________

---
*이 Issue는 AI Agent Chatbot에서 자동으로 생성되었습니다.*
"""
            
            return {
                'title': title,
                'body': body
            }
            
        except Exception as e:
            logger.error(f"Issue 템플릿 생성 실패: {e}")
            return {
                'title': f"질문 답변 실패: {question[:50]}",
                'body': f"질문: {question}\n\n에러: {error_message or '답변 생성 실패'}"
            }
    
    def suggest_issue_creation(self, 
                              question: str, 
                              error_message: str = None,
                              system_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        GitHub Issue 생성 제안
        
        Args:
            question: 사용자 질문
            error_message: 에러 메시지
            system_info: 시스템 정보
            
        Returns:
            Dict[str, Any]: Issue 생성 제안 정보
        """
        try:
            # Issue 템플릿 생성
            template = self.create_issue_template(question, error_message, system_info)
            
            # Issue URL 생성
            issue_url = self.create_issue_url(
                title=template['title'],
                body=template['body'],
                labels=['bug', 'question-answer-failure', 'auto-generated']
            )
            
            return {
                'suggested': True,
                'title': template['title'],
                'body': template['body'],
                'url': issue_url,
                'repository': f"{self.owner}/{self.repo}",
                'message': f"질문에 답변하지 못했습니다. GitHub Issue를 생성하여 문제를 보고해주세요."
            }
            
        except Exception as e:
            logger.error(f"Issue 생성 제안 실패: {e}")
            return {
                'suggested': False,
                'message': f"Issue 생성 제안 중 오류가 발생했습니다: {str(e)}"
            }
    
    def get_repository_info(self) -> Dict[str, str]:
        """Repository 정보 반환"""
        return {
            'owner': self.owner,
            'repo': self.repo,
            'url': self.repository_url,
            'issues_url': f"https://github.com/{self.owner}/{self.repo}/issues"
        }


# 사용 예제
if __name__ == "__main__":
    # Issue Helper 초기화
    issue_helper = GitHubIssueHelper()
    
    # 테스트 질문
    question = "GitHub에서 문서를 추출하는 방법은?"
    error_message = "관련 문서를 찾을 수 없습니다."
    
    # Issue 생성 제안
    suggestion = issue_helper.suggest_issue_creation(
        question=question,
        error_message=error_message
    )
    
    print(f"제안: {suggestion['suggested']}")
    print(f"제목: {suggestion['title']}")
    print(f"URL: {suggestion['url']}")
