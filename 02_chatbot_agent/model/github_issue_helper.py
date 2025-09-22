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
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import math

from config import get_config
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

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
        
        # 임베딩 모델 초기화
        self.embedding_model = OpenAIEmbeddings(
            openai_api_key=self.config.openai_api_key,
            model=self.config.embedding_model
        )
        
        # 텍스트 분할기 초기화
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # 불용어 목록 (한국어 + 영어)
        self.stop_words = {
            '이', '가', '을', '를', '에', '에서', '로', '으로', '의', '와', '과', '는', '은', '도', '만', 
            '부터', '까지', '에게', '한테', '께', '처럼', '같이', '보다', '마다', '조차', '마저', '뿐',
            '어떻게', '무엇', '언제', '어디서', '왜', '누가', '어느', '몇', '얼마나',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must', 'shall'
        }
    
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
        하이브리드 검색 기반 유사한 GitHub Issue 검색
        (BM25 + 의미적 유사도 + 키워드 매칭)
        
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
            
            logger.info(f"하이브리드 GitHub 이슈 검색 시작: {question}")
            
            # 1단계: 키워드 기반 초기 검색으로 후보 이슈들 수집
            candidate_issues = self._get_candidate_issues(question, max_results * 4)  # 4배 더 많이 가져와서 필터링
            
            if not candidate_issues:
                logger.info("후보 이슈가 없습니다.")
                return []
            
            # 2단계: 하이브리드 스코어링 (BM25 + 의미적 유사도 + 키워드 매칭)
            scored_issues = self._calculate_hybrid_similarity(question, candidate_issues)
            
            # 3단계: 최종 스코어 순으로 정렬하고 상위 결과 반환
            scored_issues.sort(key=lambda x: x['final_score'], reverse=True)
            top_issues = scored_issues[:max_results]
            
            logger.info(f"하이브리드 검색으로 유사한 이슈 {len(top_issues)}개 발견")
            return top_issues
                
        except Exception as e:
            logger.error(f"하이브리드 이슈 검색 실패: {e}")
            return []
    
    def _get_candidate_issues(self, question: str, max_candidates: int = 15) -> List[Dict[str, Any]]:
        """키워드 기반으로 후보 이슈들을 가져오기"""
        try:
            # 검색 쿼리 구성
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
                'per_page': max_candidates,
                'state': 'all'
            }
            
            # GitHub Search API 사용
            search_url = "https://api.github.com/search/issues"
            
            logger.info(f"후보 이슈 검색 중: {query}")
            response = requests.get(search_url, headers=headers, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                issues = data.get('items', [])
                
                # 이슈 정보 정리
                candidate_issues = []
                for issue in issues:
                    issue_info = {
                        'number': issue.get('number'),
                        'title': issue.get('title'),
                        'body': issue.get('body', ''),
                        'state': issue.get('state'),
                        'url': issue.get('html_url'),
                        'created_at': issue.get('created_at'),
                        'updated_at': issue.get('updated_at'),
                        'labels': [label.get('name') for label in issue.get('labels', [])]
                    }
                    candidate_issues.append(issue_info)
                
                logger.info(f"후보 이슈 {len(candidate_issues)}개 수집")
                return candidate_issues
                
            else:
                logger.error(f"GitHub API 요청 실패: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            logger.error(f"후보 이슈 검색 실패: {e}")
            return []
    
    def _calculate_hybrid_similarity(self, question: str, candidate_issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """하이브리드 스코어링 (BM25 + 의미적 유사도 + 키워드 매칭)"""
        try:
            if not candidate_issues:
                return []
            
            # 질문 전처리
            question_tokens = self._preprocess_text(question)
            
            # 각 이슈에 대해 여러 스코어 계산
            for issue in candidate_issues:
                issue_text = f"{issue.get('title', '')} {issue.get('body', '')[:2000]}"
                issue_tokens = self._preprocess_text(issue_text)
                
                # 1. BM25 스코어 계산
                bm25_score = self._calculate_bm25_score(question_tokens, issue_tokens, candidate_issues)
                
                # 2. 의미적 유사도 계산
                semantic_score = self._calculate_semantic_score(question, issue_text)
                
                # 3. 키워드 매칭 스코어 계산
                keyword_score = self._calculate_keyword_score(question_tokens, issue_tokens)
                
                # 4. 최종 스코어 계산 (가중 평균)
                final_score = (
                    bm25_score * 0.4 +      # BM25 가중치 40%
                    semantic_score * 0.4 +  # 의미적 유사도 가중치 40%
                    keyword_score * 0.2     # 키워드 매칭 가중치 20%
                )
                
                # 각 스코어를 이슈에 저장
                issue['bm25_score'] = bm25_score
                issue['semantic_score'] = semantic_score
                issue['keyword_score'] = keyword_score
                issue['final_score'] = final_score
                issue['similarity_score'] = final_score  # 기존 호환성을 위해 유지
            
            # 최종 스코어가 0.2 이상인 이슈만 반환
            filtered_issues = [issue for issue in candidate_issues if issue['final_score'] >= 0.2]
            
            logger.info(f"하이브리드 스코어링 완료: {len(filtered_issues)}개 이슈가 임계값(0.2) 이상")
            return filtered_issues
            
        except Exception as e:
            logger.error(f"하이브리드 스코어링 실패: {e}")
            # 실패시 기존 방식으로 폴백
            return self._fallback_similarity_calculation(question, candidate_issues)
    
    def _preprocess_text(self, text: str) -> List[str]:
        """텍스트 전처리 (토큰화, 불용어 제거, 정규화)"""
        try:
            # 소문자 변환 및 특수문자 제거
            text = re.sub(r'[^\w\s]', ' ', text.lower())
            
            # 토큰화
            tokens = text.split()
            
            # 불용어 제거 및 길이 필터링
            filtered_tokens = [
                token for token in tokens 
                if token not in self.stop_words and len(token) > 1
            ]
            
            return filtered_tokens
            
        except Exception as e:
            logger.error(f"텍스트 전처리 실패: {e}")
            return text.split()
    
    def _calculate_bm25_score(self, query_tokens: List[str], doc_tokens: List[str], all_docs: List[Dict[str, Any]]) -> float:
        """BM25 스코어 계산"""
        try:
            if not query_tokens or not doc_tokens:
                return 0.0
            
            # BM25 파라미터
            k1 = 1.2
            b = 0.75
            
            # 문서 길이
            doc_length = len(doc_tokens)
            
            # 전체 문서의 평균 길이 계산
            total_docs = len(all_docs)
            if total_docs == 0:
                return 0.0
            
            avg_doc_length = sum(len(self._preprocess_text(f"{doc.get('title', '')} {doc.get('body', '')[:2000]}")) 
                               for doc in all_docs) / total_docs
            
            # 문서 내 토큰 빈도 계산
            doc_token_counts = Counter(doc_tokens)
            
            # 전체 문서에서의 토큰 빈도 계산
            all_doc_tokens = []
            for doc in all_docs:
                all_doc_tokens.extend(self._preprocess_text(f"{doc.get('title', '')} {doc.get('body', '')[:2000]}"))
            all_token_counts = Counter(all_doc_tokens)
            
            # BM25 스코어 계산
            score = 0.0
            for term in query_tokens:
                if term in doc_token_counts:
                    tf = doc_token_counts[term]
                    df = all_token_counts.get(term, 1)
                    idf = math.log((total_docs - df + 0.5) / (df + 0.5))
                    
                    # BM25 공식
                    term_score = idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_length / avg_doc_length)))
                    score += term_score
            
            return score
            
        except Exception as e:
            logger.error(f"BM25 스코어 계산 실패: {e}")
            return 0.0
    
    def _calculate_semantic_score(self, question: str, issue_text: str) -> float:
        """의미적 유사도 계산 (임베딩 기반)"""
        try:
            # 질문과 이슈 텍스트를 임베딩으로 변환
            question_embedding = self.embedding_model.embed_query(question)
            issue_embedding = self.embedding_model.embed_query(issue_text)
            
            # 코사인 유사도 계산
            question_array = np.array(question_embedding).reshape(1, -1)
            issue_array = np.array(issue_embedding).reshape(1, -1)
            
            similarity = cosine_similarity(question_array, issue_array)[0][0]
            return float(similarity)
            
        except Exception as e:
            logger.error(f"의미적 유사도 계산 실패: {e}")
            return 0.0
    
    def _calculate_keyword_score(self, query_tokens: List[str], doc_tokens: List[str]) -> float:
        """키워드 매칭 스코어 계산"""
        try:
            if not query_tokens or not doc_tokens:
                return 0.0
            
            # 공통 토큰 찾기
            common_tokens = set(query_tokens) & set(doc_tokens)
            
            if not common_tokens:
                return 0.0
            
            # Jaccard 유사도 계산
            union_tokens = set(query_tokens) | set(doc_tokens)
            jaccard_score = len(common_tokens) / len(union_tokens)
            
            # 가중치 적용 (공통 토큰의 비율)
            weighted_score = jaccard_score * (len(common_tokens) / len(query_tokens))
            
            return weighted_score
            
        except Exception as e:
            logger.error(f"키워드 매칭 스코어 계산 실패: {e}")
            return 0.0
    
    def _fallback_similarity_calculation(self, question: str, candidate_issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """LLM 실패시 기존 방식으로 폴백"""
        try:
            for issue in candidate_issues:
                issue['similarity_score'] = self._calculate_similarity(question, issue.get('title', ''))
            
            # 유사도가 0.1 이상인 이슈만 반환
            filtered_issues = [issue for issue in candidate_issues if issue['similarity_score'] >= 0.1]
            
            logger.info(f"폴백 방식으로 유사도 계산: {len(filtered_issues)}개 이슈")
            return filtered_issues
            
        except Exception as e:
            logger.error(f"폴백 유사도 계산 실패: {e}")
            return candidate_issues
    
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
