"""
GitHub MCP Client
LangChain MCP Adapters를 사용한 GitHub API 통신 클라이언트
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import asyncio
import json

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage


class GitHubMCPClient:
    """GitHub MCP Client - LangChain MCP Adapters를 사용한 GitHub API 통신"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.is_initialized = False
        
        # GitHub API 설정
        self.token = config.get("github_token")
        self.openai_api_key = config.get("openai_api_key")
        
        # MCP 클라이언트 및 LLM 초기화
        self.mcp_client: Optional[MultiServerMCPClient] = None
        self.llm: Optional[ChatOpenAI] = None
    
    async def initialize(self) -> bool:
        """MCP 클라이언트 초기화"""
        try:
            # LLM 초기화
            if self.openai_api_key:
                self.llm = ChatOpenAI(
                    api_key=self.openai_api_key,
                    model="gpt-4o-mini",
                    temperature=0.1
                )
            
            # MCP 클라이언트 초기화 - GitHub API는 직접 HTTP 요청으로 처리
            # 실제 MCP 서버가 필요한 경우 여기에 추가
            self.mcp_client = None  # GitHub API는 MCP 서버가 아님
            
            # 연결 테스트
            if await self._test_connection():
                self.is_initialized = True
                self.logger.info("GitHub MCP Client 초기화 완료")
                return True
            else:
                self.logger.error("GitHub API 연결 테스트 실패")
                return False
                
        except Exception as e:
            self.logger.error(f"GitHub MCP Client 초기화 실패: {e}")
            return False
    
    async def _test_connection(self) -> bool:
        """GitHub API 연결 테스트"""
        try:
            # GitHub API 직접 연결 테스트
            import aiohttp
            
            headers = {
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "Multi-Agent-Chatbot/1.0"
            }
            
            if self.token:
                headers["Authorization"] = f"token {self.token}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get("https://api.github.com/user", headers=headers) as response:
                    if response.status == 200:
                        self.logger.info("GitHub API 연결 성공")
                        return True
                    else:
                        self.logger.warning(f"GitHub API 연결 실패: {response.status}")
                        return False
        except Exception as e:
            self.logger.error(f"GitHub API 연결 테스트 오류: {e}")
            return False
    
    async def search_repositories(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """저장소 검색"""
        try:
            if not self.is_initialized:
                raise Exception("MCP 클라이언트가 초기화되지 않았습니다.")
            
            # GitHub API 직접 호출
            import aiohttp
            
            headers = {
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "Multi-Agent-Chatbot/1.0"
            }
            
            if self.token:
                headers["Authorization"] = f"token {self.token}"
            
            url = "https://api.github.com/search/repositories"
            params = {
                "q": query,
                "sort": "stars",
                "order": "desc",
                "per_page": limit
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        repositories = []
                        
                        for repo in data.get("items", []):
                            repositories.append({
                                "id": repo["id"],
                                "name": repo["name"],
                                "full_name": repo["full_name"],
                                "description": repo.get("description", ""),
                                "html_url": repo["html_url"],
                                "stars": repo["stargazers_count"],
                                "language": repo.get("language", ""),
                                "created_at": repo["created_at"],
                                "updated_at": repo["updated_at"]
                            })
                        
                        self.logger.info(f"저장소 검색 완료: {len(repositories)}개")
                        return repositories
                    else:
                        self.logger.error(f"저장소 검색 실패: {response.status}")
                        return []
                    
        except Exception as e:
            self.logger.error(f"저장소 검색 오류: {e}")
            return []
    
    async def get_repository_contents(self, owner: str, repo: str, path: str = "") -> List[Dict[str, Any]]:
        """저장소 내용 조회"""
        try:
            if not self.is_initialized:
                raise Exception("MCP 클라이언트가 초기화되지 않았습니다.")
            
            # GitHub API 직접 호출
            import aiohttp
            
            headers = {
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "Multi-Agent-Chatbot/1.0"
            }
            
            if self.token:
                headers["Authorization"] = f"token {self.token}"
            
            url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        contents = []
                        
                        for item in data:
                            contents.append({
                                "name": item["name"],
                                "path": item["path"],
                                "type": item["type"],
                                "size": item.get("size", 0),
                                "download_url": item.get("download_url"),
                                "html_url": item["html_url"]
                            })
                        
                        self.logger.info(f"저장소 내용 조회 완료: {len(contents)}개 항목")
                        return contents
                    else:
                        self.logger.error(f"저장소 내용 조회 실패: {response.status}")
                        return []
                    
        except Exception as e:
            self.logger.error(f"저장소 내용 조회 오류: {e}")
            return []
    
    async def get_file_content(self, owner: str, repo: str, path: str) -> Optional[str]:
        """파일 내용 조회"""
        try:
            if not self.is_initialized:
                raise Exception("MCP 클라이언트가 초기화되지 않았습니다.")
            
            # GitHub API 직접 호출
            import aiohttp
            import base64
            
            headers = {
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "Multi-Agent-Chatbot/1.0"
            }
            
            if self.token:
                headers["Authorization"] = f"token {self.token}"
            
            url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = base64.b64decode(data["content"]).decode("utf-8")
                        self.logger.info(f"파일 내용 조회 완료: {path}")
                        return content
                    else:
                        self.logger.error(f"파일 내용 조회 실패: {response.status}")
                        return None
                    
        except Exception as e:
            self.logger.error(f"파일 내용 조회 오류: {e}")
            return None
    
    async def search_issues(self, owner: str, repo: str, query: str, state: str = "all", limit: int = 10) -> List[Dict[str, Any]]:
        """이슈 검색"""
        try:
            if not self.is_initialized:
                raise Exception("MCP 클라이언트가 초기화되지 않았습니다.")
            
            # GitHub API 직접 호출
            import aiohttp
            
            headers = {
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "Multi-Agent-Chatbot/1.0"
            }
            
            if self.token:
                headers["Authorization"] = f"token {self.token}"
            
            url = "https://api.github.com/search/issues"
            params = {
                "q": f"repo:{owner}/{repo} {query}",
                "sort": "updated",
                "order": "desc",
                "per_page": limit
            }
            
            if state != "all":
                params["q"] += f" state:{state}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        issues = []
                        
                        for issue in data.get("items", []):
                            issues.append({
                                "number": issue["number"],
                                "title": issue["title"],
                                "body": issue.get("body", ""),
                                "state": issue["state"],
                                "labels": [label["name"] for label in issue.get("labels", [])],
                                "html_url": issue["html_url"],
                                "created_at": issue["created_at"],
                                "updated_at": issue["updated_at"],
                                "user": issue["user"]["login"],
                                "assignee": issue.get("assignee", {}).get("login") if issue.get("assignee") else None
                            })
                        
                        self.logger.info(f"이슈 검색 완료: {len(issues)}개")
                        return issues
                    else:
                        self.logger.error(f"이슈 검색 실패: {response.status}")
                        return []
                    
        except Exception as e:
            self.logger.error(f"이슈 검색 오류: {e}")
            return []
    
    async def create_issue(self, owner: str, repo: str, title: str, body: str, labels: List[str] = None) -> Optional[Dict[str, Any]]:
        """이슈 생성"""
        try:
            if not self.is_initialized:
                raise Exception("MCP 클라이언트가 초기화되지 않았습니다.")
            
            # GitHub API 직접 호출
            import aiohttp
            
            headers = {
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "Multi-Agent-Chatbot/1.0"
            }
            
            if self.token:
                headers["Authorization"] = f"token {self.token}"
            
            url = f"https://api.github.com/repos/{owner}/{repo}/issues"
            data = {
                "title": title,
                "body": body
            }
            
            if labels:
                data["labels"] = labels
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=data) as response:
                    if response.status == 201:
                        issue_data = await response.json()
                        new_issue = {
                            "number": issue_data["number"],
                            "title": issue_data["title"],
                            "body": issue_data["body"],
                            "state": issue_data["state"],
                            "labels": [label["name"] for label in issue_data.get("labels", [])],
                            "html_url": issue_data["html_url"],
                            "created_at": issue_data["created_at"],
                            "user": issue_data["user"]["login"]
                        }
                        
                        self.logger.info(f"이슈 생성 완료: #{new_issue['number']}")
                        return new_issue
                    else:
                        self.logger.error(f"이슈 생성 실패: {response.status}")
                        return None
                    
        except Exception as e:
            self.logger.error(f"이슈 생성 오류: {e}")
            return None
    
    async def get_issue(self, owner: str, repo: str, issue_number: int) -> Optional[Dict[str, Any]]:
        """이슈 조회"""
        try:
            if not self.is_initialized:
                raise Exception("MCP 클라이언트가 초기화되지 않았습니다.")
            
            # GitHub API 직접 호출
            import aiohttp
            
            headers = {
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "Multi-Agent-Chatbot/1.0"
            }
            
            if self.token:
                headers["Authorization"] = f"token {self.token}"
            
            url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        issue_data = await response.json()
                        issue = {
                            "number": issue_data["number"],
                            "title": issue_data["title"],
                            "body": issue_data.get("body", ""),
                            "state": issue_data["state"],
                            "labels": [label["name"] for label in issue_data.get("labels", [])],
                            "html_url": issue_data["html_url"],
                            "created_at": issue_data["created_at"],
                            "updated_at": issue_data["updated_at"],
                            "user": issue_data["user"]["login"],
                            "assignee": issue_data.get("assignee", {}).get("login") if issue_data.get("assignee") else None
                        }
                        
                        self.logger.info(f"이슈 조회 완료: #{issue['number']}")
                        return issue
                    else:
                        self.logger.error(f"이슈 조회 실패: {response.status}")
                        return None
                    
        except Exception as e:
            self.logger.error(f"이슈 조회 오류: {e}")
            return None
    
    async def search_code(self, query: str, owner: str = None, repo: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """코드 검색"""
        try:
            if not self.is_initialized:
                raise Exception("MCP 클라이언트가 초기화되지 않았습니다.")
            
            # GitHub API 직접 호출
            import aiohttp
            
            headers = {
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "Multi-Agent-Chatbot/1.0"
            }
            
            if self.token:
                headers["Authorization"] = f"token {self.token}"
            
            url = "https://api.github.com/search/code"
            params = {
                "q": query,
                "sort": "indexed",
                "order": "desc",
                "per_page": limit
            }
            
            if owner and repo:
                params["q"] = f"repo:{owner}/{repo} {query}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        code_results = []
                        
                        for item in data.get("items", []):
                            code_results.append({
                                "name": item["name"],
                                "path": item["path"],
                                "repository": item["repository"]["full_name"],
                                "html_url": item["html_url"],
                                "score": item["score"],
                                "text_matches": item.get("text_matches", [])
                            })
                        
                        self.logger.info(f"코드 검색 완료: {len(code_results)}개")
                        return code_results
                    else:
                        self.logger.error(f"코드 검색 실패: {response.status}")
                        return []
                    
        except Exception as e:
            self.logger.error(f"코드 검색 오류: {e}")
            return []
    
    async def get_repository_info(self, owner: str, repo: str) -> Optional[Dict[str, Any]]:
        """저장소 정보 조회"""
        try:
            if not self.is_initialized:
                raise Exception("MCP 클라이언트가 초기화되지 않았습니다.")
            
            # GitHub API 직접 호출
            import aiohttp
            
            headers = {
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "Multi-Agent-Chatbot/1.0"
            }
            
            if self.token:
                headers["Authorization"] = f"token {self.token}"
            
            url = f"https://api.github.com/repos/{owner}/{repo}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        repo_data = await response.json()
                        repository_info = {
                            "id": repo_data["id"],
                            "name": repo_data["name"],
                            "full_name": repo_data["full_name"],
                            "description": repo_data.get("description", ""),
                            "html_url": repo_data["html_url"],
                            "clone_url": repo_data["clone_url"],
                            "stars": repo_data["stargazers_count"],
                            "forks": repo_data["forks_count"],
                            "language": repo_data.get("language", ""),
                            "topics": repo_data.get("topics", []),
                            "created_at": repo_data["created_at"],
                            "updated_at": repo_data["updated_at"],
                            "pushed_at": repo_data["pushed_at"],
                            "size": repo_data["size"],
                            "default_branch": repo_data["default_branch"]
                        }
                        
                        self.logger.info(f"저장소 정보 조회 완료: {repository_info['full_name']}")
                        return repository_info
                    else:
                        self.logger.error(f"저장소 정보 조회 실패: {response.status}")
                        return None
                    
        except Exception as e:
            self.logger.error(f"저장소 정보 조회 오류: {e}")
            return None
    
    def get_status(self) -> Dict[str, Any]:
        """MCP 클라이언트 상태 정보 반환"""
        return {
            "is_initialized": self.is_initialized,
            "has_token": bool(self.token),
            "has_openai_key": bool(self.openai_api_key),
            "llm_available": self.llm is not None
        }
    
    async def close(self):
        """MCP 클라이언트 종료"""
        # GitHub API는 stateless이므로 특별한 종료 작업이 필요 없음
        self.logger.info("GitHub MCP Client 종료")
    
    async def __aenter__(self):
        """비동기 컨텍스트 매니저 진입"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 매니저 종료"""
        await self.close()
    
    async def process_with_llm(self, query: str, context: str = "") -> str:
        """LLM을 사용한 쿼리 처리"""
        try:
            if not self.llm:
                return "LLM이 초기화되지 않았습니다."
            
            # 시스템 메시지 설정
            system_message = SystemMessage(content="""
            당신은 GitHub API 전문가입니다. 사용자의 질문에 대해 정확하고 유용한 답변을 제공하세요.
            GitHub API를 통해 얻은 정보를 바탕으로 답변하세요.
            """)
            
            # 사용자 메시지 구성
            user_content = f"질문: {query}"
            if context:
                user_content += f"\n\n컨텍스트: {context}"
            
            user_message = HumanMessage(content=user_content)
            
            # LLM 호출
            response = await self.llm.agenerate([[system_message, user_message]])
            return response.generations[0][0].text
            
        except Exception as e:
            self.logger.error(f"LLM 처리 오류: {e}")
            return f"LLM 처리 중 오류가 발생했습니다: {e}"


# 사용 예제
if __name__ == "__main__":
    import asyncio
    
    async def test_github_mcp():
        # 설정
        config = {
            "github_token": "your-github-token",
            "openai_api_key": "your-openai-api-key"
        }
        
        # MCP 클라이언트 테스트
        async with GitHubMCPClient(config) as mcp:
            # 저장소 검색 테스트
            repos = await mcp.search_repositories("python machine learning", limit=5)
            print(f"검색된 저장소: {len(repos)}개")
            
            # 이슈 검색 테스트
            issues = await mcp.search_issues("microsoft", "vscode", "bug", limit=5)
            print(f"검색된 이슈: {len(issues)}개")
            
            # 코드 검색 테스트
            code_results = await mcp.search_code("def main", "microsoft", "vscode", limit=5)
            print(f"검색된 코드: {len(code_results)}개")
            
            # LLM 처리 테스트
            if mcp.llm:
                llm_response = await mcp.process_with_llm(
                    "Python으로 머신러닝 프로젝트를 시작하려면 어떻게 해야 하나요?",
                    "GitHub에서 관련 저장소들을 찾았습니다."
                )
                print(f"LLM 응답: {llm_response}")
    
    # 테스트 실행
    asyncio.run(test_github_mcp())
