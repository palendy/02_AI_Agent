"""
Langchain-based Issue Agent - 이슈 검색, 생성, 관리를 담당하는 Agent
Langchain_Teddy 패턴을 참조한 Issue Agent 구현
"""

import logging
from typing import Dict, Any, Optional, List, Annotated
from datetime import datetime

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent

from .base_agent import BaseAgent, AgentType


class IssueAgent(BaseAgent):
    """Langchain 기반 Issue Agent - 이슈 검색, 생성, 관리"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(AgentType.ISSUE, config)
        self.llm: Optional[ChatOpenAI] = None
        self.agent_executor: Optional[AgentExecutor] = None
        self.json_parser = JsonOutputParser()
        
        # 이슈 분석 프롬프트
        self.issue_analysis_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""
당신은 이슈 검색, 생성, 관리를 담당하는 Issue Agent입니다.

사용자의 VOC(Voice of Customer)나 이슈 신고를 분석하여:
1. 이슈 타입 분류 (BUG, FEATURE_REQUEST, IMPROVEMENT, QUESTION, COMPLAINT)
2. 우선순위 결정 (high, medium, low)
3. 적절한 라벨 제안
4. 이슈 설명 생성

분석 결과를 JSON 형태로 반환하세요:
{
    "issue_type": "BUG|FEATURE_REQUEST|IMPROVEMENT|QUESTION|COMPLAINT",
    "priority": "high|medium|low",
    "labels": ["label1", "label2"],
    "title": "이슈 제목",
    "description": "이슈 설명",
    "reasoning": "분석 이유"
}
            """),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessage(content="{input}")
        ])
        
        # 이슈 응답 프롬프트
        self.issue_response_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""
당신은 이슈 검색, 생성, 관리를 담당하는 Issue Agent입니다.

사용자의 VOC(Voice of Customer)나 이슈 신고에 대해:
1. 유사한 이슈가 있는지 검색
2. 유사한 이슈가 있으면 해당 이슈 정보 제공
3. 유사한 이슈가 없으면 새로운 이슈 생성 제안
4. 이슈 생성 시 적절한 제목, 라벨, 우선순위 제안

답변 시 다음 사항을 고려하세요:
1. 이슈의 심각도와 우선순위 판단
2. 적절한 라벨과 카테고리 제안
3. 재현 단계나 환경 정보 수집
4. 관련 담당자나 팀 안내
5. 해결 방안이나 대안 제시

답변 형식:
- 이슈 검색 결과 (있는 경우)
- 새로운 이슈 생성 제안 (필요한 경우)
- 해결 방안이나 다음 단계 안내
- 관련 링크나 참조 정보
            """),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessage(content="사용자 메시지: {message}\n\n이슈 분석 결과: {analysis}\n\n유사한 이슈들: {similar_issues}")
        ])
    
    async def initialize(self) -> bool:
        """Issue Agent 초기화"""
        try:
            # LLM 초기화
            self.llm = ChatOpenAI(
                model=self.config.get("issue_agent_model", "gpt-4o-mini"),
                temperature=0.3,
                api_key=self.config.get("openai_api_key")
            )
            
            # 도구 정의
            tools = [
                self._create_issue_search_tool(),
                self._create_issue_analysis_tool(),
                self._create_issue_creation_tool(),
                self._create_issue_response_tool()
            ]
            
            # Agent 생성
            agent = create_openai_tools_agent(
                llm=self.llm,
                tools=tools,
                prompt=self.issue_response_prompt
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
            self.logger.info("Langchain Issue Agent 초기화 완료")
            return True
            
        except Exception as e:
            self.logger.error(f"Langchain Issue Agent 초기화 실패: {e}")
            return False
    
    def _create_issue_search_tool(self):
        """이슈 검색 도구 생성"""
        @tool
        def search_similar_issues(question: str) -> str:
            """유사한 이슈를 검색합니다."""
            try:
                # 실제 구현에서는 GitHub MCP Server와 통신
                # 여기서는 시뮬레이션된 결과 반환
                
                # 시뮬레이션된 이슈 검색 결과
                mock_issues = [
                    {
                        "number": 123,
                        "title": "로그인 오류 발생",
                        "body": "사용자가 로그인할 때 500 에러가 발생합니다. 재현 단계: 1) 로그인 페이지 접속 2) 계정 정보 입력 3) 로그인 버튼 클릭",
                        "state": "open",
                        "labels": ["bug", "high-priority", "authentication"],
                        "url": "https://github.com/example/repo/issues/123",
                        "created_at": "2024-01-15T10:30:00Z",
                        "similarity_score": 0.85
                    },
                    {
                        "number": 124,
                        "title": "UI 개선 요청",
                        "body": "사용자 인터페이스를 더 직관적으로 만들어주세요. 현재 버튼이 너무 작아서 클릭하기 어렵습니다.",
                        "state": "closed",
                        "labels": ["enhancement", "ui", "ux"],
                        "url": "https://github.com/example/repo/issues/124",
                        "created_at": "2024-01-10T14:20:00Z",
                        "similarity_score": 0.72
                    },
                    {
                        "number": 125,
                        "title": "성능 개선 필요",
                        "body": "페이지 로딩 속도가 너무 느립니다. 특히 대용량 데이터를 처리할 때 문제가 발생합니다.",
                        "state": "open",
                        "labels": ["performance", "enhancement"],
                        "url": "https://github.com/example/repo/issues/125",
                        "created_at": "2024-01-12T09:15:00Z",
                        "similarity_score": 0.68
                    }
                ]
                
                # 질문과 관련성이 높은 이슈 필터링
                relevant_issues = [
                    issue for issue in mock_issues 
                    if issue["similarity_score"] > 0.6
                ]
                
                # 이슈 정보를 문자열로 변환
                result = "검색된 유사 이슈:\n\n"
                for i, issue in enumerate(relevant_issues, 1):
                    result += f"이슈 {i}:\n"
                    result += f"번호: #{issue['number']}\n"
                    result += f"제목: {issue['title']}\n"
                    result += f"상태: {issue['state']}\n"
                    result += f"라벨: {', '.join(issue['labels'])}\n"
                    result += f"내용: {issue['body'][:200]}...\n"
                    result += f"URL: {issue['url']}\n"
                    result += f"유사도: {issue['similarity_score']:.2f}\n\n"
                
                return result
                
            except Exception as e:
                return f"이슈 검색 중 오류가 발생했습니다: {str(e)}"
        
        return search_similar_issues
    
    def _create_issue_analysis_tool(self):
        """이슈 분석 도구 생성"""
        @tool
        def analyze_issue(message: str) -> str:
            """사용자 메시지를 분석하여 이슈 정보를 추출합니다."""
            try:
                # 간단한 키워드 기반 분석
                message_lower = message.lower()
                
                # 이슈 타입 분석
                if any(keyword in message_lower for keyword in ["오류", "에러", "버그", "작동하지", "문제가 생겼"]):
                    issue_type = "BUG"
                    priority = "high"
                    labels = ["bug", "high-priority"]
                elif any(keyword in message_lower for keyword in ["개선", "향상", "더 좋게", "업그레이드"]):
                    issue_type = "IMPROVEMENT"
                    priority = "medium"
                    labels = ["enhancement", "improvement"]
                elif any(keyword in message_lower for keyword in ["기능", "추가", "새로운", "요청"]):
                    issue_type = "FEATURE_REQUEST"
                    priority = "medium"
                    labels = ["feature", "enhancement"]
                elif any(keyword in message_lower for keyword in ["불편", "피드백", "의견", "제안"]):
                    issue_type = "COMPLAINT"
                    priority = "low"
                    labels = ["feedback", "complaint"]
                else:
                    issue_type = "QUESTION"
                    priority = "low"
                    labels = ["question"]
                
                # 이슈 제목 생성
                title = f"{issue_type}: {message[:50]}..."
                if len(message) > 50:
                    title += "..."
                
                # 이슈 설명 생성
                description = f"""
사용자 메시지: {message}

이슈 타입: {issue_type}
우선순위: {priority}
제안 라벨: {', '.join(labels)}

추가 정보가 필요합니다:
- 재현 단계 (버그인 경우)
- 예상 동작과 실제 동작 (버그인 경우)
- 사용 환경 정보
- 스크린샷이나 로그 (가능한 경우)
                """
                
                return f"""
이슈 분석 결과:
- 타입: {issue_type}
- 우선순위: {priority}
- 라벨: {', '.join(labels)}
- 제목: {title}
- 설명: {description}
                """
                
            except Exception as e:
                return f"이슈 분석 중 오류가 발생했습니다: {str(e)}"
        
        return analyze_issue
    
    def _create_issue_creation_tool(self):
        """이슈 생성 도구 생성"""
        @tool
        def create_issue_suggestion(title: str, description: str, labels: str, priority: str) -> str:
            """새로운 이슈 생성을 제안합니다."""
            try:
                # 이슈 생성 제안
                suggestion = f"""
새로운 이슈 생성 제안:

제목: {title}
설명: {description}
라벨: {labels}
우선순위: {priority}

이슈를 생성하시겠습니까?
- 예: 이슈를 생성합니다
- 아니오: 이슈 생성을 취소합니다
- 수정: 제목이나 설명을 수정합니다

생성할 경우 다음 정보를 추가로 제공해주세요:
- 재현 단계 (버그인 경우)
- 예상 동작과 실제 동작
- 사용 환경 정보
- 관련 스크린샷이나 로그
                """
                
                return suggestion
                
            except Exception as e:
                return f"이슈 생성 제안 중 오류가 발생했습니다: {str(e)}"
        
        return create_issue_suggestion
    
    def _create_issue_response_tool(self):
        """이슈 응답 도구 생성"""
        @tool
        def generate_issue_response(message: str, analysis: str, similar_issues: str) -> str:
            """이슈에 대한 응답을 생성합니다."""
            try:
                # 유사한 이슈가 있는지 확인
                if "검색된 유사 이슈" in similar_issues and "이슈 1:" in similar_issues:
                    # 유사한 이슈가 있는 경우
                    response = f"""
🔍 유사한 이슈를 찾았습니다!

{similar_issues}

이 이슈들을 확인해보시거나, 추가 정보를 제공해주시면 더 정확한 도움을 드릴 수 있습니다.

새로운 이슈가 필요하시다면 말씀해주세요.
                    """
                else:
                    # 유사한 이슈가 없는 경우
                    response = f"""
📝 새로운 이슈가 필요해 보입니다.

{analysis}

이슈를 생성하시겠습니까? 더 자세한 정보를 제공해주시면 도움을 드리겠습니다.
                    """
                
                return response
                
            except Exception as e:
                return f"이슈 응답 생성 중 오류가 발생했습니다: {str(e)}"
        
        return generate_issue_response
    
    async def process_message(self, message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        메시지 처리 - 이슈 검색 및 관리
        
        Args:
            message: 사용자 메시지
            context: 추가 컨텍스트
            
        Returns:
            Dict[str, Any]: 처리 결과
        """
        if not self.is_initialized:
            return {
                "success": False,
                "error": "Issue Agent가 초기화되지 않았습니다.",
                "answer": "Issue Agent를 초기화하는 중입니다."
            }
        
        try:
            # 채팅 히스토리 가져오기
            chat_history = self._get_chat_history_for_llm()
            
            # 이슈 처리 체인 구성
            issue_chain = (
                {
                    "message": RunnablePassthrough(),
                    "analysis": self._analyze_issue_chain,
                    "similar_issues": self._search_issues_chain,
                    "chat_history": lambda x: chat_history
                }
                | self.issue_response_prompt
                | self.llm
                | StrOutputParser()
            )
            
            # 답변 생성
            answer = await issue_chain.ainvoke(message)
            
            # 이슈 분석 결과 가져오기
            analysis_result = await self._get_issue_analysis(message)
            
            # 유사한 이슈 검색 결과 가져오기
            similar_issues = await self._get_similar_issues(message)
            
            # 이슈 생성 제안 (필요한 경우)
            issue_creation_suggestion = None
            if not similar_issues or len(similar_issues) == 0:
                issue_creation_suggestion = await self._suggest_issue_creation(message, analysis_result)
            
            # 히스토리에 추가
            self.add_to_history(message, answer, {
                "issue_type": analysis_result.get("issue_type", "UNKNOWN"),
                "similar_issues_found": len(similar_issues),
                "issue_creation_suggested": issue_creation_suggestion is not None
            })
            
            return {
                "success": True,
                "answer": answer,
                "similar_issues": similar_issues,
                "issue_creation_suggestion": issue_creation_suggestion,
                "new_issue_created": False,  # 실제 구현에서는 이슈 생성 후 True로 설정
                "issue_type": analysis_result.get("issue_type", "UNKNOWN")
            }
            
        except Exception as e:
            self.logger.error(f"Issue Agent 메시지 처리 중 오류: {e}")
            return {
                "success": False,
                "error": f"이슈 처리 중 오류가 발생했습니다: {e}",
                "answer": "죄송합니다. 이슈를 처리할 수 없습니다. 잠시 후 다시 시도해주세요."
            }
    
    async def _analyze_issue_chain(self, message: str) -> str:
        """이슈 분석 체인"""
        try:
            # 이슈 분석 도구 사용
            analysis_tool = self._create_issue_analysis_tool()
            analysis_result = analysis_tool.invoke({"message": message})
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"이슈 분석 실패: {e}")
            return f"이슈 분석 중 오류가 발생했습니다: {str(e)}"
    
    async def _search_issues_chain(self, message: str) -> str:
        """이슈 검색 체인"""
        try:
            # 이슈 검색 도구 사용
            search_tool = self._create_issue_search_tool()
            search_result = search_tool.invoke({"question": message})
            
            return search_result
            
        except Exception as e:
            self.logger.error(f"이슈 검색 실패: {e}")
            return f"이슈 검색 중 오류가 발생했습니다: {str(e)}"
    
    async def _get_issue_analysis(self, message: str) -> Dict[str, Any]:
        """이슈 분석 결과 가져오기"""
        try:
            # 간단한 분석 결과 반환
            message_lower = message.lower()
            
            if any(keyword in message_lower for keyword in ["오류", "에러", "버그"]):
                return {
                    "issue_type": "BUG",
                    "priority": "high",
                    "labels": ["bug", "high-priority"]
                }
            elif any(keyword in message_lower for keyword in ["개선", "향상"]):
                return {
                    "issue_type": "IMPROVEMENT",
                    "priority": "medium",
                    "labels": ["enhancement"]
                }
            else:
                return {
                    "issue_type": "QUESTION",
                    "priority": "low",
                    "labels": ["question"]
                }
                
        except Exception as e:
            self.logger.error(f"이슈 분석 결과 가져오기 실패: {e}")
            return {
                "issue_type": "UNKNOWN",
                "priority": "low",
                "labels": []
            }
    
    async def _get_similar_issues(self, message: str) -> List[Dict[str, Any]]:
        """유사한 이슈 가져오기"""
        try:
            # 시뮬레이션된 유사 이슈 결과
            mock_issues = [
                {
                    "number": 123,
                    "title": "로그인 오류 발생",
                    "state": "open",
                    "labels": ["bug", "high-priority"],
                    "url": "https://github.com/example/repo/issues/123",
                    "similarity_score": 0.85
                }
            ]
            
            return mock_issues
            
        except Exception as e:
            self.logger.error(f"유사 이슈 가져오기 실패: {e}")
            return []
    
    async def _suggest_issue_creation(self, message: str, analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """새로운 이슈 생성 제안"""
        try:
            # 이슈 생성 제안
            suggestion = {
                "suggested": True,
                "title": f"{analysis.get('issue_type', 'ISSUE')}: {message[:50]}...",
                "labels": analysis.get("labels", []),
                "priority": analysis.get("priority", "low"),
                "description": f"사용자 메시지: {message}\n\n분석 결과: {analysis}",
                "issue_type": analysis.get("issue_type", "UNKNOWN")
            }
            
            return suggestion
            
        except Exception as e:
            self.logger.error(f"이슈 생성 제안 실패: {e}")
            return None
    
    def _get_chat_history_for_llm(self) -> List:
        """LLM용 채팅 히스토리 변환"""
        history = []
        
        for entry in self.conversation_history[-10:]:  # 최근 10개만
            history.append(HumanMessage(content=entry["message"]))
            history.append(AIMessage(content=entry["response"]))
        
        return history
    
    def get_issue_stats(self) -> Dict[str, Any]:
        """이슈 통계 정보 반환"""
        if not self.conversation_history:
            return {"total_issues": 0, "issue_types": {}}
        
        issue_types = {}
        total_issues = len(self.conversation_history)
        
        for entry in self.conversation_history:
            metadata = entry.get("metadata", {})
            issue_type = metadata.get("issue_type", "UNKNOWN")
            issue_types[issue_type] = issue_types.get(issue_type, 0) + 1
        
        return {
            "total_issues": total_issues,
            "issue_types": issue_types
        }