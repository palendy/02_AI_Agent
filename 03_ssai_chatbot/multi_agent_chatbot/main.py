#!/usr/bin/env python3
"""
Multi-Agent Chatbot Main Application
Langchain 기반 3-Agent 시스템 메인 애플리케이션
"""

import asyncio
import logging
import sys
from typing import List, Optional

from utils.config import get_config
from core.multi_agent_workflow import MultiAgentWorkflow

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_banner():
    """배너 출력"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                Multi-Agent Chatbot System                   ║
║              Langchain 기반 3-Agent 시스템                  ║
║                                                              ║
║  🤖 ChatAgent  📚 RAG Agent  🔧 Issue Agent                ║
║  🚀 Langchain + LangGraph + GitHub MCP Integration         ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_help():
    """도움말 출력"""
    help_text = """
사용 가능한 명령어:
  help, h          - 이 도움말 표시
  info, i          - 시스템 정보 표시
  status, s        - Agent 상태 확인
  chat, c          - 대화 모드 시작
  history, hist    - 대화 기록 조회
  clear            - 대화 기록 초기화
  reset            - 시스템 초기화
  quit, q, exit    - 프로그램 종료

예제:
  chat
  info
  status
  history 5
    """
    print(help_text)


async def interactive_mode(workflow: MultiAgentWorkflow):
    """대화형 모드"""
    print("\n🤖 대화 모드를 시작합니다. 'quit'를 입력하면 종료됩니다.")
    print("💡 팁: 'help'를 입력하면 도움말을 볼 수 있습니다.\n")
    
    while True:
        try:
            # 사용자 입력 받기
            user_input = input("👤 사용자: ").strip()
            
            if not user_input:
                continue
            
            # 명령어 처리
            if user_input.lower() in ['quit', 'q', 'exit']:
                print("👋 안녕히 가세요!")
                break
            elif user_input.lower() in ['help', 'h']:
                print_help()
                continue
            elif user_input.lower() in ['info', 'i']:
                status = workflow.get_workflow_status()
                print(f"\n📊 시스템 정보:")
                print(f"  - 초기화 상태: {'✅' if status['is_initialized'] else '❌'}")
                print(f"  - Chat Agent: {'✅' if status['agents']['chat_agent']['is_initialized'] else '❌'}")
                print(f"  - RAG Agent: {'✅' if status['agents']['rag_agent']['is_initialized'] else '❌'}")
                print(f"  - Issue Agent: {'✅' if status['agents']['issue_agent']['is_initialized'] else '❌'}")
                print(f"  - GitHub MCP Client: {'✅' if status['mcp_clients']['github_mcp']['is_initialized'] else '❌'}")
                continue
            elif user_input.lower() in ['status', 's']:
                status = workflow.get_workflow_status()
                print(f"\n📊 Agent 상태:")
                for agent_name, agent_status in status['agents'].items():
                    print(f"  - {agent_name}: {agent_status}")
                continue
            elif user_input.lower() in ['history', 'hist']:
                # 각 Agent의 히스토리 조회
                chat_history = workflow.chat_agent.get_conversation_history(5)
                rag_history = workflow.rag_agent.get_conversation_history(5)
                issue_history = workflow.issue_agent.get_conversation_history(5)
                
                print(f"\n📝 최근 대화 기록:")
                print(f"  Chat Agent: {len(chat_history)}개")
                print(f"  RAG Agent: {len(rag_history)}개")
                print(f"  Issue Agent: {len(issue_history)}개")
                continue
            elif user_input.lower() == 'clear':
                await workflow.reset()
                print("🗑️ 대화 기록이 초기화되었습니다.")
                continue
            elif user_input.lower() == 'reset':
                await workflow.reset()
                print("🔄 시스템이 초기화되었습니다.")
                continue
            
            # 질문 처리
            print("🤔 생각 중...")
            result = await workflow.process_message(user_input)
            
            # 답변 출력
            if result['success']:
                print(f"\n🤖 {result['processing_agent'].upper()} Agent: {result['answer']}")
                
                # 추가 정보 출력
                print(f"   📊 메시지 타입: {result['message_type']}")
                print(f"   📈 신뢰도: {result['confidence']:.3f}")
                print(f"   ⏱️ 처리 시간: {result['processing_time']:.2f}초")
                
                if result['processing_agent'] == 'rag':
                    rag_info = result.get('rag_info', {})
                    print(f"   📚 사용된 문서: {rag_info.get('documents_used', 0)}개")
                    print(f"   📈 관련성 점수: {rag_info.get('relevance_score', 0.0):.3f}")
                
                elif result['processing_agent'] == 'issue':
                    issue_info = result.get('issue_info', {})
                    print(f"   🔧 유사한 이슈: {issue_info.get('similar_issues', 0)}개")
                    if issue_info.get('issue_creation_suggestion'):
                        print(f"   💡 이슈 생성 제안: 있음")
                
            else:
                print(f"\n❌ 오류: {result.get('error', '알 수 없는 오류')}")
            
            print()  # 빈 줄 추가
            
        except KeyboardInterrupt:
            print("\n\n👋 안녕히 가세요!")
            break
        except Exception as e:
            logger.error(f"대화 중 오류 발생: {e}")
            print(f"❌ 오류가 발생했습니다: {e}")


async def batch_mode(workflow: MultiAgentWorkflow, messages: List[str]):
    """배치 모드 (메시지 목록 처리)"""
    print(f"\n📝 배치 모드: {len(messages)}개 메시지 처리")
    
    for i, message in enumerate(messages, 1):
        print(f"\n[{i}/{len(messages)}] 메시지: {message}")
        
        result = await workflow.process_message(message)
        if result['success']:
            print(f"답변: {result['answer']}")
            print(f"처리 Agent: {result['processing_agent']}")
        else:
            print(f"오류: {result.get('error', '알 수 없는 오류')}")


async def main():
    """메인 함수"""
    try:
        # 배너 출력
        print_banner()
        
        # 설정 확인
        config = get_config()
        if not config.validate():
            print("❌ 설정 검증에 실패했습니다.")
            print("   .env 파일을 확인하고 필요한 API 키를 설정하세요.")
            sys.exit(1)
        
        print("✅ 설정 검증 완료")
        
        # 워크플로우 초기화
        print("🤖 Multi-Agent Workflow 초기화 중...")
        workflow = MultiAgentWorkflow(config.get_agent_config())
        
        if await workflow.initialize():
            print("✅ Multi-Agent Workflow 초기화 완료")
        else:
            print("❌ Multi-Agent Workflow 초기화 실패")
            sys.exit(1)
        
        # 시스템 정보 출력
        status = workflow.get_workflow_status()
        print(f"📊 시스템 상태: {'✅ 정상' if status['is_initialized'] else '❌ 오류'}")
        
        # 명령행 인수 확인
        if len(sys.argv) > 1:
            command = sys.argv[1].lower()
            
            if command in ['help', 'h']:
                print_help()
            elif command in ['info', 'i']:
                status = workflow.get_workflow_status()
                print(f"\n📊 시스템 정보: {status}")
            elif command in ['status', 's']:
                status = workflow.get_workflow_status()
                print(f"\n📊 Agent 상태:")
                for agent_name, agent_status in status['agents'].items():
                    print(f"  - {agent_name}: {agent_status}")
            elif command in ['chat', 'c']:
                await interactive_mode(workflow)
            elif command in ['history', 'hist']:
                # 각 Agent의 히스토리 조회
                chat_history = workflow.chat_agent.get_conversation_history(10)
                rag_history = workflow.rag_agent.get_conversation_history(10)
                issue_history = workflow.issue_agent.get_conversation_history(10)
                
                print(f"\n📝 최근 대화 기록:")
                print(f"  Chat Agent: {len(chat_history)}개")
                for entry in chat_history[-3:]:
                    print(f"    - {entry['message'][:50]}...")
                print(f"  RAG Agent: {len(rag_history)}개")
                for entry in rag_history[-3:]:
                    print(f"    - {entry['message'][:50]}...")
                print(f"  Issue Agent: {len(issue_history)}개")
                for entry in issue_history[-3:]:
                    print(f"    - {entry['message'][:50]}...")
            elif command == 'clear':
                await workflow.reset()
                print("🗑️ 대화 기록이 초기화되었습니다.")
            elif command == 'reset':
                await workflow.reset()
                print("🔄 시스템이 초기화되었습니다.")
            else:
                print(f"❌ 알 수 없는 명령어: {command}")
                print_help()
        else:
            # 대화형 모드 시작
            await interactive_mode(workflow)
    
    except KeyboardInterrupt:
        print("\n\n👋 프로그램이 중단되었습니다.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"프로그램 실행 중 오류 발생: {e}")
        print(f"❌ 오류가 발생했습니다: {e}")
        sys.exit(1)
    finally:
        # 워크플로우 종료
        try:
            await workflow.close()
        except:
            pass


if __name__ == "__main__":
    # 비동기 메인 함수 실행
    asyncio.run(main())
