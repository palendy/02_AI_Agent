"""
GitHub Issue Helper
Hybrid Search + Cross-Encoder Re-ranking을 사용한 GitHub Issue 검색
"""

import os
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import requests
from urllib.parse import quote
import re
import numpy as np
from collections import Counter
import math

from config import get_config
from langchain.embeddings import OpenAIEmbeddings
from sentence_transformers import CrossEncoder

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GitHubIssueHelper:
    """Hybrid Search + Cross-Encoder Re-ranking을 사용한 GitHub Issue 검색"""
    
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
        
        # Dense Embedding 모델 초기화 (OpenAI)
        self.embedding_model = OpenAIEmbeddings(
            openai_api_key=self.config.openai_api_key,
            model=self.config.embedding_model
        )
        
        # Cross-Encoder 모델 초기화 (Re-ranking용)
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # BM25 파라미터
        self.bm25_k1 = 1.2
        self.bm25_b = 0.75
        
        # 불용어 목록
        self.stop_words = {
            '이', '가', '을', '를', '에', '에서', '로', '으로', '의', '와', '과', '는', '은', '도', '만', 
            '부터', '까지', '에게', '한테', '께', '처럼', '같이', '보다', '마다', '조차', '마저', '뿐',
            '어떻게', '무엇', '언제', '어디서', '왜', '누가', '어느', '몇', '얼마나', '뭐야', '뭐',
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
        Hybrid Search + Cross-Encoder Re-ranking을 사용한 GitHub Issue 검색
        
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
            
            logger.info(f"Hybrid Search + Cross-Encoder Re-ranking 시작: {question}")
            
            # 1단계: GitHub API로 후보 이슈들 수집
            candidate_issues = self._get_candidate_issues(question, max_results * 3)
            
            if not candidate_issues:
                logger.info("후보 이슈가 없습니다.")
                return []
            
            # 2단계: Hybrid Search (BM25 + Dense Embedding)
            hybrid_scores = self._calculate_hybrid_scores(question, candidate_issues)
            
            # 3단계: Cross-Encoder Re-ranking
            reranked_issues = self._cross_encoder_rerank(question, hybrid_scores, max_results)
            
            logger.info(f"Hybrid Search + Re-ranking으로 유사한 이슈 {len(reranked_issues)}개 발견")
            return reranked_issues
                
        except Exception as e:
            logger.error(f"Hybrid Search + Re-ranking 실패: {e}")
            return []
    
    def _get_candidate_issues(self, question: str, max_candidates: int = 15) -> List[Dict[str, Any]]:
        """GitHub API로 후보 이슈들을 가져오기"""
        try:
            # 검색 쿼리 생성
            search_queries = self._generate_search_queries(question)
            
            all_issues = []
            seen_issues = set()
            
            for query in search_queries[:3]:  # 최대 3개 쿼리만 사용
                if len(all_issues) >= max_candidates:
                    break
                    
                issues = self._search_github_api(query, max_candidates // len(search_queries) + 1)
                
                for issue in issues:
                    issue_id = issue.get('number')
                    if issue_id not in seen_issues:
                        all_issues.append(issue)
                        seen_issues.add(issue_id)
            
            logger.info(f"후보 이슈 {len(all_issues)}개 수집")
            return all_issues
            
        except Exception as e:
            logger.error(f"후보 이슈 검색 실패: {e}")
            return []
    
    def _generate_search_queries(self, question: str) -> List[str]:
        """질문에서 다양한 검색 쿼리 생성"""
        try:
            # 기본 키워드 추출
            keywords = self._extract_keywords(question)
            
            queries = []
            
            # 1. 원본 질문
            queries.append(question)
            
            # 2. 키워드만으로 검색
            if keywords:
                queries.append(" ".join(keywords))
                
                # 3. 개별 키워드들
                for keyword in keywords[:3]:
                    queries.append(keyword)
            
            # 4. 특별한 패턴 검색
            if "module not found" in question.lower() or "module not foun" in question.lower():
                queries.extend([
                    "module not found",
                    "module error",
                    "import error",
                    "no module named"
                ])
            
            # 중복 제거하고 최대 5개 쿼리만 사용
            unique_queries = []
            for query in queries:
                if query and query not in unique_queries:
                    unique_queries.append(query)
                if len(unique_queries) >= 5:
                    break
            
            return unique_queries
            
        except Exception as e:
            logger.error(f"검색 쿼리 생성 실패: {e}")
            return [question]
    
    def _extract_keywords(self, question: str) -> List[str]:
        """질문에서 핵심 키워드 추출"""
        try:
            # 소문자 변환 및 특수문자 제거
            cleaned = re.sub(r'[^\w\s]', ' ', question.lower())
            
            # 단어 분리 및 불용어 제거
            words = [word for word in cleaned.split() 
                    if word not in self.stop_words and len(word) > 1]
            
            return words[:5]  # 상위 5개 키워드
            
        except Exception as e:
            logger.error(f"키워드 추출 실패: {e}")
            return [question]
    
    def _search_github_api(self, query: str, per_page: int = 10) -> List[Dict[str, Any]]:
        """GitHub API로 이슈 검색"""
        try:
            headers = {
                'Authorization': f'token {self.github_token}',
                'Accept': 'application/vnd.github.v3+json',
                'User-Agent': 'AI-Agent-Chatbot'
            }
            
            params = {
                'q': f"{query} repo:{self.owner}/{self.repo}",
                'sort': 'updated',
                'order': 'desc',
                'per_page': per_page,
                'state': 'all'
            }
            
            search_url = "https://api.github.com/search/issues"
            
            response = requests.get(search_url, headers=headers, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                issues = data.get('items', [])
                
                issue_list = []
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
                    issue_list.append(issue_info)
                
                return issue_list
                
            else:
                logger.error(f"GitHub API 요청 실패: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"GitHub API 검색 실패: {e}")
            return []
    
    def _calculate_hybrid_scores(self, question: str, candidate_issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Hybrid Search (BM25 + Dense Embedding)"""
        try:
            if not candidate_issues:
                return []
            
            logger.info(f"Hybrid Search 계산 시작: {len(candidate_issues)}개 이슈")
            
            # 질문 전처리
            question_tokens = self._preprocess_text(question)
            
            # 모든 이슈 텍스트 수집 (BM25 계산용)
            all_issue_texts = []
            for issue in candidate_issues:
                issue_text = f"{issue.get('title', '')} {issue.get('body', '')[:1000]}"
                all_issue_texts.append(issue_text)
            
            # 각 이슈에 대해 BM25 + Dense Embedding 스코어 계산
            for i, issue in enumerate(candidate_issues):
                issue_text = f"{issue.get('title', '')} {issue.get('body', '')[:1000]}"
                issue_tokens = self._preprocess_text(issue_text)
                
                # 1. BM25 스코어 계산
                bm25_score = self._calculate_bm25_score(question_tokens, issue_tokens, all_issue_texts)
                
                # 2. Dense Embedding 스코어 계산
                dense_score = self._calculate_dense_score(question, issue_text)
                
                # 3. Hybrid 스코어 계산 (BM25 60% + Dense 40%)
                hybrid_score = bm25_score * 0.6 + dense_score * 0.4
                
                # 스코어 저장
                issue['bm25_score'] = bm25_score
                issue['dense_score'] = dense_score
                issue['hybrid_score'] = hybrid_score
                issue['similarity_score'] = hybrid_score  # 기존 호환성을 위해 유지
            
            # Hybrid 스코어 순으로 정렬
            candidate_issues.sort(key=lambda x: x['hybrid_score'], reverse=True)
            
            logger.info(f"Hybrid Search 완료: {len(candidate_issues)}개 이슈")
            return candidate_issues
            
        except Exception as e:
            logger.error(f"Hybrid Search 실패: {e}")
            return candidate_issues
    
    def _preprocess_text(self, text: str) -> List[str]:
        """텍스트 전처리 (토큰화, 불용어 제거)"""
        try:
            if not text:
                return []
            
            # 소문자 변환 및 특수문자 제거
            cleaned = re.sub(r'[^\w\s]', ' ', text.lower())
            
            # 토큰화
            tokens = cleaned.split()
            
            # 불용어 제거 및 길이 필터링
            filtered_tokens = [
                token for token in tokens 
                if token not in self.stop_words and len(token) > 1
            ]
            
            return filtered_tokens
            
        except Exception as e:
            logger.error(f"텍스트 전처리 실패: {e}")
            return text.split()
    
    def _calculate_bm25_score(self, query_tokens: List[str], doc_tokens: List[str], all_docs: List[str]) -> float:
        """BM25 스코어 계산"""
        try:
            if not query_tokens or not doc_tokens:
                return 0.0
            
            # 문서 길이
            doc_length = len(doc_tokens)
            
            # 전체 문서의 평균 길이 계산
            total_docs = len(all_docs)
            if total_docs == 0:
                return 0.0
            
            avg_doc_length = sum(len(self._preprocess_text(doc)) for doc in all_docs) / total_docs
            
            # 문서 내 토큰 빈도 계산
            doc_token_counts = Counter(doc_tokens)
            
            # 전체 문서에서의 토큰 빈도 계산
            all_token_counts = Counter()
            for doc in all_docs:
                all_token_counts.update(self._preprocess_text(doc))
            
            # BM25 스코어 계산
            score = 0.0
            for term in query_tokens:
                if term in doc_token_counts:
                    tf = doc_token_counts[term]
                    df = all_token_counts.get(term, 1)
                    idf = math.log((total_docs - df + 0.5) / (df + 0.5))
                    
                    # BM25 공식
                    term_score = idf * (tf * (self.bm25_k1 + 1)) / (tf + self.bm25_k1 * (1 - self.bm25_b + self.bm25_b * (doc_length / avg_doc_length)))
                    score += term_score
            
            return score
            
        except Exception as e:
            logger.error(f"BM25 스코어 계산 실패: {e}")
            return 0.0
    
    def _calculate_dense_score(self, question: str, issue_text: str) -> float:
        """Dense Embedding 스코어 계산"""
        try:
            # 질문과 이슈 텍스트를 임베딩으로 변환
            question_embedding = self.embedding_model.embed_query(question)
            issue_embedding = self.embedding_model.embed_query(issue_text)
            
            # 코사인 유사도 계산
            question_array = np.array(question_embedding)
            issue_array = np.array(issue_embedding)
            
            # 정규화
            question_norm = question_array / np.linalg.norm(question_array)
            issue_norm = issue_array / np.linalg.norm(issue_array)
            
            # 코사인 유사도
            similarity = np.dot(question_norm, issue_norm)
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Dense Embedding 스코어 계산 실패: {e}")
            return 0.0
    
    def _cross_encoder_rerank(self, question: str, hybrid_scores: List[Dict[str, Any]], max_results: int) -> List[Dict[str, Any]]:
        """Cross-Encoder Re-ranking"""
        try:
            if not hybrid_scores:
                return []
            
            logger.info(f"Cross-Encoder Re-ranking 시작: {len(hybrid_scores)}개 이슈")
            
            # 상위 이슈들만 Re-ranking (성능 최적화)
            top_issues = hybrid_scores[:min(len(hybrid_scores), max_results * 2)]
            
            # Cross-Encoder로 Re-ranking
            reranked_issues = []
            for issue in top_issues:
                issue_text = f"{issue.get('title', '')} {issue.get('body', '')[:500]}"
                
                # Cross-Encoder 점수 계산
                cross_score = self.cross_encoder.predict([question, issue_text])
                
                # 최종 점수 (Hybrid 70% + Cross-Encoder 30%)
                final_score = issue.get('hybrid_score', 0) * 0.7 + cross_score[0] * 0.3
                
                issue['cross_encoder_score'] = cross_score[0]
                issue['final_score'] = final_score
                issue['similarity_score'] = final_score  # 기존 호환성을 위해 유지
                
                reranked_issues.append(issue)
            
            # 최종 점수 순으로 정렬
            reranked_issues.sort(key=lambda x: x['final_score'], reverse=True)
            
            # 상위 결과만 반환
            result = reranked_issues[:max_results]
            
            logger.info(f"Cross-Encoder Re-ranking 완료: {len(result)}개 이슈")
            return result
            
        except Exception as e:
            logger.error(f"Cross-Encoder Re-ranking 실패: {e}")
            # 실패시 Hybrid Search 결과 반환
            return hybrid_scores[:max_results]
    
    
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
