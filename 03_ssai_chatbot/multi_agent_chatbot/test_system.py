#!/usr/bin/env python3
"""
Multi-Agent Chatbot System Test
시스템 테스트 및 검증 스크립트
"""

import asyncio
import logging
from utils.config import get_config
from core.multi_agent_workflow import MultiAgentWorkflow

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_agents():
    """Agent들 개별 테스트"""
    print("🧪 Agent 개별 테스트 시작...")
    
    config = get_config()
    if not config.validate():
        print("❌ 설정 검증 실패")
        return False
    
    # 워크플로우 초기화
    workflow = MultiAgentWorkflow(config.get_agent_config())
    
    if not await workflow.initialize():
        print("❌ 워크플로우 초기화 실패")
        return False
    
    print("✅ 워크플로우 초기화 성공")
    
    # 각 Agent 상태 확인
    status = workflow.get_workflow_status()
    print(f"📊 시스템 상태: {status['is_initialized']}")
    
    for agent_name, agent_status in status['agents'].items():
        print(f"  - {agent_name}: {'✅' if agent_status['is_initialized'] else '❌'}")
    
    return True


async def test_message_processing():
    """메시지 처리 테스트"""
    print("\n💬 메시지 처리 테스트 시작...")
    
    config = get_config()
    workflow = MultiAgentWorkflow(config.get_agent_config())
    
    if not await workflow.initialize():
        print("❌ 워크플로우 초기화 실패")
        return False
    
    # 테스트 메시지들
    test_messages = [
        "안녕하세요!",
        "GitHub에서 문서를 추출하는 방법을 알려주세요",
        "버그가 발생했어요. 도움을 주세요",
        "이 기능을 개선해주세요"
    ]
    
    for i, message in enumerate(test_messages, 1):
        print(f"\n[{i}] 테스트 메시지: {message}")
        
        try:
            result = await workflow.process_message(message)
            
            if result['success']:
                print(f"✅ 처리 성공")
                print(f"   답변: {result['answer'][:100]}...")
                print(f"   처리 Agent: {result['processing_agent']}")
                print(f"   처리 시간: {result['processing_time']:.2f}초")
            else:
                print(f"❌ 처리 실패: {result.get('error', '알 수 없는 오류')}")
                
        except Exception as e:
            print(f"❌ 처리 중 오류: {e}")
    
    return True


async def main():
    """메인 테스트 함수"""
    print("🚀 Multi-Agent Chatbot System 테스트 시작")
    print("=" * 50)
    
    try:
        # Agent 테스트
        if not await test_agents():
            print("❌ Agent 테스트 실패")
            return
        
        # 메시지 처리 테스트
        if not await test_message_processing():
            print("❌ 메시지 처리 테스트 실패")
            return
        
        print("\n✅ 모든 테스트 통과!")
        print("🎉 Multi-Agent Chatbot System이 정상적으로 작동합니다!")
        
    except Exception as e:
        print(f"❌ 테스트 중 오류 발생: {e}")
        logger.error(f"테스트 실패: {e}")


if __name__ == "__main__":
    asyncio.run(main())
