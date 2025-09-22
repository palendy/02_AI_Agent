"""
GitHub Issue Helper
에러 발생 시 GitHub Issue 등록을 도와주는 클래스
"""

import os
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import requests
from urllib.parse import quote
import re

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
    
    def search_similar_issues(self, question: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        유사한 GitHub Issue 검색
        
        Args:
            question: 검색할 질문
            max_results: 최대 결과 수
            
        Returns:
            List[Dict[str, Any]]: 유사한 이슈 목록
        """
        try:
            if not self.github_token:
                logger.warning("GitHub 토큰이 없어 이슈 검색을 할 수 없습니다.")
                return []
            
            # GitHub API URL
            api_url = f"https://api.github.com/repos/{self.owner}/{self.repo}/issues"
            
            # 검색 쿼리 구성 (제목과 본문에서 검색)
            search_terms = self._extract_search_terms(question)
            query = " ".join(search_terms)
            
            # API 요청 헤더
            headers = {
                'Authorization': f'token {self.github_token}',
                'Accept': 'application/vnd.github.v3+json',
                'User-Agent': 'AI-Agent-Chatbot'
            }
            
            # API 파라미터
            params = {
                'q': f"{query} repo:{self.owner}/{self.repo}",
                'sort': 'updated',
                'order': 'desc',
                'per_page': max_results,
                'state': 'all'  # open과 closed 모두 검색
            }
            
            # GitHub Search API 사용
            search_url = "https://api.github.com/search/issues"
            
            logger.info(f"GitHub 이슈 검색 중: {query}")
            response = requests.get(search_url, headers=headers, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                issues = data.get('items', [])
                
                # 이슈 정보 정리
                similar_issues = []
                for issue in issues:
                    issue_info = {
                        'number': issue.get('number'),
                        'title': issue.get('title'),
                        'body': issue.get('body', '')[:500] + '...' if issue.get('body') and len(issue.get('body', '')) > 500 else issue.get('body', ''),
                        'state': issue.get('state'),
                        'url': issue.get('html_url'),
                        'created_at': issue.get('created_at'),
                        'updated_at': issue.get('updated_at'),
                        'labels': [label.get('name') for label in issue.get('labels', [])],
                        'similarity_score': self._calculate_similarity(question, issue.get('title', ''))
                    }
                    similar_issues.append(issue_info)
                
                # 유사도 순으로 정렬
                similar_issues.sort(key=lambda x: x['similarity_score'], reverse=True)
                
                logger.info(f"유사한 이슈 {len(similar_issues)}개 발견")
                return similar_issues
                
            else:
                logger.error(f"GitHub API 요청 실패: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            logger.error(f"이슈 검색 실패: {e}")
            return []
    
    def _extract_search_terms(self, question: str) -> List[str]:
        """질문에서 검색 키워드 추출"""
        try:
            # 특수문자 제거 및 소문자 변환
            cleaned = re.sub(r'[^\w\s]', ' ', question.lower())
            
            # 불용어 제거 (간단한 버전)
            stop_words = {'이', '가', '을', '를', '에', '에서', '로', '으로', '의', '와', '과', '는', '은', '도', '만', '부터', '까지', '에게', '한테', '께', '처럼', '같이', '보다', '마다', '조차', '마저', '뿐', '만', '뿐만', '아니라', '또한', '그리고', '하지만', '그런데', '그러나', '따라서', '그래서', '왜냐하면', '때문에', '위해', '위해서', '통해', '통해서', '대해', '대해서', '관해', '관해서', '대한', '관한', '위한', '통한', '통해', '통해서', '대해', '대해서', '관해', '관해서', '대한', '관한', '위한', '통한', '어떻게', '무엇', '언제', '어디서', '왜', '누가', '어느', '몇', '얼마나', '어느', '몇', '얼마나', '어느', '몇', '얼마나'}
            
            # 단어 분리 및 불용어 제거
            words = [word for word in cleaned.split() if word not in stop_words and len(word) > 1]
            
            # 상위 5개 키워드 반환
            return words[:5]
            
        except Exception as e:
            logger.error(f"검색 키워드 추출 실패: {e}")
            return [question]
    
    def _calculate_similarity(self, question: str, title: str) -> float:
        """질문과 이슈 제목 간의 유사도 계산 (간단한 버전)"""
        try:
            if not title:
                return 0.0
            
            # 소문자 변환 및 특수문자 제거
            q_clean = re.sub(r'[^\w\s]', ' ', question.lower())
            t_clean = re.sub(r'[^\w\s]', ' ', title.lower())
            
            # 단어 분리
            q_words = set(q_clean.split())
            t_words = set(t_clean.split())
            
            if not q_words or not t_words:
                return 0.0
            
            # Jaccard 유사도 계산
            intersection = len(q_words.intersection(t_words))
            union = len(q_words.union(t_words))
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            logger.error(f"유사도 계산 실패: {e}")
            return 0.0
    
    def get_issue_answer(self, issue: Dict[str, Any]) -> Optional[str]:
        """
        Closed 이슈에서 답변 추출
        
        Args:
            issue: 이슈 정보
            
        Returns:
            Optional[str]: 답변 내용 (있다면)
        """
        try:
            if issue.get('state') != 'closed':
                return None
            
            # 이슈 본문에서 답변 관련 내용 찾기
            body = issue.get('body', '')
            if not body:
                return None
            
            # 답변 관련 키워드가 있는지 확인
            answer_keywords = ['해결', '답변', '해결방법', '해결책', '방법', '해결됨', '수정됨', '완료', '답변드립니다', '해결되었습니다']
            
            has_answer = any(keyword in body for keyword in answer_keywords)
            if not has_answer:
                return None
            
            # 답변 부분 추출 (간단한 버전)
            lines = body.split('\n')
            answer_lines = []
            in_answer_section = False
            
            for line in lines:
                line = line.strip()
                if any(keyword in line for keyword in answer_keywords):
                    in_answer_section = True
                
                if in_answer_section and line:
                    answer_lines.append(line)
                    
                    # 답변 섹션이 끝나는 조건
                    if line.startswith('---') or line.startswith('##') or line.startswith('###'):
                        break
            
            if answer_lines:
                return '\n'.join(answer_lines[:10])  # 최대 10줄
                
            return None
            
        except Exception as e:
            logger.error(f"이슈 답변 추출 실패: {e}")
            return None
    
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
