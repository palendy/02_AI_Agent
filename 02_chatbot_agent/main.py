"""
AI Agent Chatbot Main Application
GitHub 문서를 기반으로 한 지능형 챗봇 메인 애플리케이션
"""

import logging
import sys
from typing import List, Optional

from config import get_config
from model import AIChatbot

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
║                    AI Agent Chatbot                         ║
║              GitHub 문서 기반 지능형 챗봇                    ║
║                                                              ║
║  🚀 Corrective RAG + LangGraph + GitHub Integration        ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_help():
    """도움말 출력"""
    help_text = """
사용 가능한 명령어:
  help, h          - 이 도움말 표시
  info, i          - 시스템 정보 표시
  load, l          - 설정된 repository 로드
  add <url>        - repository 추가
  chat, c          - 대화 모드 시작
  history, hist    - 대화 기록 조회
  clear            - 대화 기록 초기화
  reset            - 시스템 초기화
  quit, q, exit    - 프로그램 종료

예제:
  add https://github.com/owner/repo
  chat
  history 5
    """
    print(help_text)


def interactive_mode(chatbot: AIChatbot):
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
                info = chatbot.get_system_info()
                print(f"\n📊 시스템 정보:")
                print(f"  - 벡터 스토어: {info.get('vector_store', {}).get('document_count', 0)}개 문서")
                print(f"  - 모델: {info.get('config', {}).get('model_name', 'Unknown')}")
                print(f"  - 임베딩: {info.get('config', {}).get('embedding_model', 'Unknown')}")
                print(f"  - 대화 수: {info.get('conversation_count', 0)}")
                print(f"  - 초기화 상태: {'✅' if info.get('initialized') else '❌'}")
                continue
            elif user_input.lower() in ['history', 'hist']:
                history = chatbot.get_conversation_history(10)
                print(f"\n📝 최근 대화 기록 ({len(history)}개):")
                for i, entry in enumerate(history, 1):
                    print(f"  {i}. {entry['question']}")
                    print(f"     → {entry['answer'][:100]}...")
                continue
            elif user_input.lower() == 'clear':
                chatbot.clear_conversation_history()
                print("🗑️ 대화 기록이 초기화되었습니다.")
                continue
            elif user_input.lower() == 'reset':
                if chatbot.reset_system():
                    print("🔄 시스템이 초기화되었습니다.")
                else:
                    print("❌ 시스템 초기화에 실패했습니다.")
                continue
            
            # 질문 처리
            print("🤔 생각 중...")
            result = chatbot.chat(user_input)
            
            # 답변 출력
            print(f"\n🤖 챗봇: {result['answer']}")
            
            # 추가 정보 출력 (디버그 모드)
            if result.get('search_source') != 'error':
                print(f"   📊 검색 소스: {result['search_source']}")
                print(f"   📈 관련성 점수: {result['relevance_score']:.3f}")
                print(f"   🔄 재시도 횟수: {result['retry_count']}")
                print(f"   📄 사용된 문서: {result['documents_used']}개")
            
            if result.get('error_message'):
                print(f"   ⚠️ 오류: {result['error_message']}")
            
            print()  # 빈 줄 추가
            
        except KeyboardInterrupt:
            print("\n\n👋 안녕히 가세요!")
            break
        except Exception as e:
            logger.error(f"대화 중 오류 발생: {e}")
            print(f"❌ 오류가 발생했습니다: {e}")


def batch_mode(chatbot: AIChatbot, questions: List[str]):
    """배치 모드 (질문 목록 처리)"""
    print(f"\n📝 배치 모드: {len(questions)}개 질문 처리")
    
    for i, question in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] 질문: {question}")
        
        result = chatbot.chat(question)
        print(f"답변: {result['answer']}")
        
        if result.get('search_source') != 'error':
            print(f"검색 소스: {result['search_source']}, 관련성: {result['relevance_score']:.3f}")


def main():
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
        
        # 챗봇 초기화
        print("🤖 AI Chatbot 초기화 중...")
        chatbot = AIChatbot()
        print("✅ AI Chatbot 초기화 완료")
        
        # 시스템 정보 출력
        info = chatbot.get_system_info()
        print(f"📊 시스템 상태: {'✅ 정상' if info.get('initialized') else '❌ 오류'}")
        print(f"📚 벡터 스토어: {info.get('vector_store', {}).get('document_count', 0)}개 문서")
        
        # 설정된 repository 로드
        print("\n📁 설정된 repository 로드 중...")
        load_result = chatbot.load_configured_repositories()
        
        if load_result['success']:
            print(f"✅ {load_result['success_count']}개 repository 로드 완료")
            print(f"📄 총 {load_result['total_documents']}개 문서 추가")
        else:
            print(f"⚠️ Repository 로드 실패: {load_result['message']}")
            print("   수동으로 repository를 추가하거나 대화를 시작할 수 있습니다.")
        
        # 명령행 인수 확인
        if len(sys.argv) > 1:
            command = sys.argv[1].lower()
            
            if command in ['help', 'h']:
                print_help()
            elif command in ['info', 'i']:
                info = chatbot.get_system_info()
                print(f"\n📊 시스템 정보: {info}")
            elif command in ['load', 'l']:
                load_result = chatbot.load_configured_repositories()
                print(f"Repository 로드 결과: {load_result}")
            elif command.startswith('add '):
                url = sys.argv[1][4:].strip()
                if url:
                    result = chatbot.add_github_repository(url)
                    print(f"Repository 추가 결과: {result}")
                else:
                    print("❌ Repository URL을 입력하세요.")
            elif command in ['chat', 'c']:
                interactive_mode(chatbot)
            elif command in ['history', 'hist']:
                limit = int(sys.argv[2]) if len(sys.argv) > 2 else 10
                history = chatbot.get_conversation_history(limit)
                print(f"\n📝 최근 대화 기록 ({len(history)}개):")
                for i, entry in enumerate(history, 1):
                    print(f"  {i}. {entry['question']}")
                    print(f"     → {entry['answer'][:100]}...")
            elif command == 'clear':
                chatbot.clear_conversation_history()
                print("🗑️ 대화 기록이 초기화되었습니다.")
            elif command == 'reset':
                if chatbot.reset_system():
                    print("🔄 시스템이 초기화되었습니다.")
                else:
                    print("❌ 시스템 초기화에 실패했습니다.")
            else:
                print(f"❌ 알 수 없는 명령어: {command}")
                print_help()
        else:
            # 대화형 모드 시작
            interactive_mode(chatbot)
    
    except KeyboardInterrupt:
        print("\n\n👋 프로그램이 중단되었습니다.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"프로그램 실행 중 오류 발생: {e}")
        print(f"❌ 오류가 발생했습니다: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
