"""
Langchain-based RAG Agent - 정보 검색 및 안내를 담당하는 Agent
Langchain_Teddy 패턴을 참조한 RAG Agent 구현
"""

import logging
from typing import Dict, Any, Optional, List, Annotated
from datetime import datetime

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from .base_agent import BaseAgent, AgentType


class RAGAgent(BaseAgent):
    """Langchain 기반 RAG Agent - 정보 검색 및 안내"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(AgentType.RAG, config)
        self.llm: Optional[ChatOpenAI] = None
        self.agent_executor: Optional[AgentExecutor] = None
        self.retriever: Optional[VectorStoreRetriever] = None
        
        # RAG 프롬프트 템플릿
        self.rag_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""
당신은 정보 검색 및 안내를 담당하는 RAG Agent입니다.

사용자의 질문에 대해 GitHub 저장소에서 관련 문서를 검색하고, 
찾은 정보를 바탕으로 정확하고 도움이 되는 답변을 제공하세요.

답변 시 다음 사항을 고려하세요:
1. 검색된 문서의 내용을 정확히 반영
2. 구체적인 예시나 코드가 있다면 포함
3. 관련 링크나 참조 정보 제공
4. 불확실한 정보는 명시적으로 표시
5. 추가 도움이 필요한 경우 안내

답변 형식:
- 명확하고 구조화된 답변
- 관련 코드 예시 (있는 경우)
- 참조 문서 링크 (있는 경우)
- 추가 정보가 필요한 경우 안내
            """),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessage(content="질문: {question}\n\n참고 문서:\n{context}")
        ])
        
        # 문서 검색 프롬프트
        self.search_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="사용자 질문에 가장 관련성 높은 문서를 검색하세요."),
            HumanMessage(content="{question}")
        ])
    
    async def initialize(self) -> bool:
        """RAG Agent 초기화"""
        try:
            # LLM 초기화
            self.llm = ChatOpenAI(
                model=self.config.get("rag_agent_model", "gpt-4o-mini"),
                temperature=0.3,
                api_key=self.config.get("openai_api_key")
            )
            
            # 도구 정의
            tools = [
                self._create_document_search_tool(),
                self._create_relevance_evaluation_tool(),
                self._create_answer_generation_tool()
            ]
            
            # Agent 생성
            agent = create_openai_tools_agent(
                llm=self.llm,
                tools=tools,
                prompt=self.rag_prompt
            )
            
            # Agent Executor 생성
            self.agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=True,
                max_iterations=5,
                return_intermediate_steps=True
            )
            
            self.is_initialized = True
            self.logger.info("Langchain RAG Agent 초기화 완료")
            return True
            
        except Exception as e:
            self.logger.error(f"Langchain RAG Agent 초기화 실패: {e}")
            return False
    
    def _create_document_search_tool(self):
        """문서 검색 도구 생성"""
        @tool
        def search_documents(question: str) -> str:
            """사용자 질문에 관련된 문서를 검색합니다."""
            try:
                # 실제 구현에서는 벡터 스토어에서 검색
                # 여기서는 시뮬레이션된 결과 반환
                
                # 시뮬레이션된 검색 결과
                mock_documents = [
                    {
                        "title": "GitHub API 사용법",
                        "content": "GitHub API를 사용하여 저장소 정보를 가져오는 방법을 설명합니다. REST API와 GraphQL API 두 가지 방식을 지원합니다.",
                        "url": "https://github.com/example/repo/blob/main/docs/api.md",
                        "relevance_score": 0.85
                    },
                    {
                        "title": "문서 추출 가이드",
                        "content": "GitHub에서 문서를 추출하는 단계별 가이드입니다. Python을 사용한 예제 코드를 포함합니다.",
                        "url": "https://github.com/example/repo/blob/main/README.md",
                        "relevance_score": 0.92
                    },
                    {
                        "title": "설치 및 설정",
                        "content": "프로젝트 설치 방법과 기본 설정에 대한 안내입니다.",
                        "url": "https://github.com/example/repo/blob/main/docs/installation.md",
                        "relevance_score": 0.78
                    }
                ]
                
                # 질문과 관련성이 높은 문서 필터링
                relevant_docs = [
                    doc for doc in mock_documents 
                    if doc["relevance_score"] > 0.7
                ]
                
                # 문서 정보를 문자열로 변환
                result = "검색된 문서:\n\n"
                for i, doc in enumerate(relevant_docs, 1):
                    result += f"문서 {i}:\n"
                    result += f"제목: {doc['title']}\n"
                    result += f"내용: {doc['content']}\n"
                    result += f"URL: {doc['url']}\n"
                    result += f"관련성: {doc['relevance_score']:.2f}\n\n"
                
                return result
                
            except Exception as e:
                return f"문서 검색 중 오류가 발생했습니다: {str(e)}"
        
        return search_documents
    
    def _create_relevance_evaluation_tool(self):
        """관련성 평가 도구 생성"""
        @tool
        def evaluate_relevance(question: str, documents: str) -> str:
            """검색된 문서들의 관련성을 평가합니다."""
            try:
                # 간단한 키워드 매칭으로 관련성 평가
                question_words = set(question.lower().split())
                
                # 문서에서 키워드 추출 (간단한 구현)
                doc_words = set(documents.lower().split())
                common_words = question_words.intersection(doc_words)
                
                if len(question_words) > 0:
                    relevance_score = len(common_words) / len(question_words)
                else:
                    relevance_score = 0.0
                
                return f"관련성 점수: {relevance_score:.3f} (공통 키워드: {len(common_words)}개)"
                
            except Exception as e:
                return f"관련성 평가 중 오류가 발생했습니다: {str(e)}"
        
        return evaluate_relevance
    
    def _create_answer_generation_tool(self):
        """답변 생성 도구 생성"""
        @tool
        def generate_answer(question: str, context: str) -> str:
            """검색된 문서를 바탕으로 답변을 생성합니다."""
            try:
                # 간단한 답변 생성 (실제로는 LLM 사용)
                if "github" in question.lower() and "api" in question.lower():
                    return """
GitHub API 사용법:

1. REST API 사용
   - GET /repos/{owner}/{repo} 엔드포인트 사용
   - 인증 토큰 필요

2. GraphQL API 사용
   - POST /graphql 엔드포인트 사용
   - 쿼리 언어로 데이터 요청

3. 예제 코드
   ```python
   import requests
   
   headers = {'Authorization': f'token {token}'}
   response = requests.get(f'https://api.github.com/repos/{owner}/{repo}', headers=headers)
   ```

더 자세한 내용은 공식 문서를 참조하세요.
                    """
                elif "문서" in question and "추출" in question:
                    return """
문서 추출 방법:

1. GitHub API를 사용한 방법
   - 저장소 내용 조회
   - 파일 다운로드

2. Git 클론을 사용한 방법
   - 저장소 클론
   - 로컬에서 파일 접근

3. 웹 스크래핑 방법
   - GitHub 웹 페이지에서 직접 추출
   - BeautifulSoup 등 라이브러리 사용

단계별 가이드와 예제 코드는 관련 문서를 참조하세요.
                    """
                else:
                    return f"질문에 대한 답변을 찾았습니다. 관련 문서를 확인해보세요:\n\n{context}"
                
            except Exception as e:
                return f"답변 생성 중 오류가 발생했습니다: {str(e)}"
        
        return generate_answer
    
    async def process_message(self, message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        메시지 처리 - 정보 검색 및 답변 생성
        
        Args:
            message: 사용자 메시지
            context: 추가 컨텍스트
            
        Returns:
            Dict[str, Any]: 처리 결과
        """
        if not self.is_initialized:
            return {
                "success": False,
                "error": "RAG Agent가 초기화되지 않았습니다.",
                "answer": "RAG Agent를 초기화하는 중입니다."
            }
        
        try:
            # 채팅 히스토리 가져오기
            chat_history = self._get_chat_history_for_llm()
            
            # RAG 체인 구성
            rag_chain = (
                {
                    "question": RunnablePassthrough(),
                    "context": self._search_documents_chain,
                    "chat_history": lambda x: chat_history
                }
                | self.rag_prompt
                | self.llm
                | StrOutputParser()
            )
            
            # 답변 생성
            answer = await rag_chain.ainvoke(message)
            
            # 문서 검색 결과 가져오기
            search_results = await self._get_search_results(message)
            
            # 관련성 평가
            relevance_score = await self._evaluate_relevance(message, search_results)
            
            # 히스토리에 추가
            self.add_to_history(message, answer, {
                "documents_found": len(search_results),
                "relevance_score": relevance_score,
                "search_source": "langchain_rag"
            })
            
            return {
                "success": True,
                "answer": answer,
                "documents": search_results,
                "relevance_score": relevance_score,
                "search_source": "langchain_rag",
                "documents_used": len(search_results)
            }
            
        except Exception as e:
            self.logger.error(f"RAG Agent 메시지 처리 중 오류: {e}")
            return {
                "success": False,
                "error": f"정보 검색 중 오류가 발생했습니다: {e}",
                "answer": "죄송합니다. 정보를 검색할 수 없습니다. 잠시 후 다시 시도해주세요."
            }
    
    async def _search_documents_chain(self, question: str) -> str:
        """문서 검색 체인"""
        try:
            # 문서 검색 도구 사용
            search_tool = self._create_document_search_tool()
            search_result = search_tool.invoke({"question": question})
            
            return search_result
            
        except Exception as e:
            self.logger.error(f"문서 검색 실패: {e}")
            return "관련 문서를 찾을 수 없습니다."
    
    async def _get_search_results(self, question: str) -> List[Dict[str, Any]]:
        """검색 결과 가져오기"""
        try:
            # 시뮬레이션된 검색 결과
            mock_results = [
                {
                    "title": "GitHub API 사용법",
                    "content": "GitHub API를 사용하여 저장소 정보를 가져오는 방법...",
                    "url": "https://github.com/example/repo/blob/main/docs/api.md",
                    "relevance_score": 0.85
                },
                {
                    "title": "문서 추출 가이드",
                    "content": "GitHub에서 문서를 추출하는 단계별 가이드...",
                    "url": "https://github.com/example/repo/blob/main/README.md",
                    "relevance_score": 0.92
                }
            ]
            
            return mock_results
            
        except Exception as e:
            self.logger.error(f"검색 결과 가져오기 실패: {e}")
            return []
    
    async def _evaluate_relevance(self, question: str, documents: List[Dict[str, Any]]) -> float:
        """관련성 평가"""
        try:
            if not documents:
                return 0.0
            
            # 문서들의 평균 관련성 점수 계산
            total_score = sum(doc.get("relevance_score", 0.0) for doc in documents)
            avg_score = total_score / len(documents)
            
            # 임계값 적용
            threshold = self.config.get("relevance_threshold", 0.6)
            return min(avg_score, 1.0) if avg_score >= threshold else avg_score
            
        except Exception as e:
            self.logger.error(f"관련성 평가 실패: {e}")
            return 0.0
    
    def _get_chat_history_for_llm(self) -> List:
        """LLM용 채팅 히스토리 변환"""
        history = []
        
        for entry in self.conversation_history[-10:]:  # 최근 10개만
            history.append(HumanMessage(content=entry["message"]))
            history.append(AIMessage(content=entry["response"]))
        
        return history
    
    async def search_similar_questions(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """유사한 질문 검색"""
        try:
            # 채팅 히스토리에서 유사한 질문 검색
            history = self.get_conversation_history(limit * 2)
            
            # 간단한 키워드 매칭으로 유사 질문 찾기
            query_words = set(query.lower().split())
            similar_questions = []
            
            for entry in history:
                question_words = set(entry["message"].lower().split())
                common_words = query_words.intersection(question_words)
                
                if len(common_words) > 0:
                    similarity = len(common_words) / len(query_words.union(question_words))
                    if similarity > 0.3:  # 30% 이상 유사
                        similar_questions.append({
                            "question": entry["message"],
                            "answer": entry["response"],
                            "similarity": similarity,
                            "timestamp": entry["timestamp"]
                        })
            
            # 유사도 순으로 정렬
            similar_questions.sort(key=lambda x: x["similarity"], reverse=True)
            
            return similar_questions[:limit]
            
        except Exception as e:
            self.logger.error(f"유사 질문 검색 실패: {e}")
            return []
    
    def get_search_stats(self) -> Dict[str, Any]:
        """검색 통계 정보 반환"""
        if not self.conversation_history:
            return {"total_searches": 0, "avg_relevance": 0.0}
        
        total_searches = len(self.conversation_history)
        total_relevance = 0.0
        valid_searches = 0
        
        for entry in self.conversation_history:
            metadata = entry.get("metadata", {})
            relevance = metadata.get("relevance_score", 0.0)
            if relevance > 0:
                total_relevance += relevance
                valid_searches += 1
        
        avg_relevance = total_relevance / valid_searches if valid_searches > 0 else 0.0
        
        return {
            "total_searches": total_searches,
            "avg_relevance": avg_relevance,
            "valid_searches": valid_searches
        }