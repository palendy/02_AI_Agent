#!/usr/bin/env python3
"""
SRS Generation Agent 실행 스크립트
================================

이 스크립트는 SRS Generation Agent를 쉽게 실행할 수 있도록 도와줍니다.
README 파일의 가이드에 따라 간단한 인터페이스를 제공합니다.

사용법:
    python run.py [spec_file1] [spec_file2] ...
    
예시:
    python run.py spec1.txt spec2.pdf
    python run.py requirements.txt
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

# .env 파일 로드 (python-dotenv가 설치되어 있다면)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# 02_SRS_Agent 디렉토리를 Python 경로에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '02_SRS_Agent'))

try:
    from srs_generation_agent import SRSGenerationAgent
    from config import AgentConfig, ConfigProfiles
except ImportError as e:
    print(f"❌ 모듈 import 오류: {e}")
    print("📋 requirements.txt에 따라 패키지를 설치했는지 확인하세요:")
    print("   pip install -r 02_SRS_Agent/requirements.txt")
    sys.exit(1)


def check_api_key():
    """OpenAI API 키 확인"""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        print("📋 다음 중 하나의 방법으로 API 키를 설정하세요:")
        print("   1. 환경변수로 설정:")
        print("      export OPENAI_API_KEY='your-api-key-here'")
        print("   2. .env 파일 생성:")
        print("      echo 'OPENAI_API_KEY=your-api-key-here' > .env")
        print("   3. 현재 세션에서 직접 설정:")
        print("      OPENAI_API_KEY='your-api-key-here' python run.py [files]")
        print("\n💡 OpenAI API 키를 얻으려면: https://platform.openai.com/api-keys")
        return False
    
    # API 키가 설정되었는지 확인 (길이 체크)
    if len(api_key) < 20:
        print("❌ OPENAI_API_KEY가 너무 짧습니다. 올바른 API 키인지 확인하세요.")
        return False
    
    print(f"✅ OpenAI API 키가 설정되었습니다 (길이: {len(api_key)})")
    return True


def validate_files(file_paths):
    """입력 파일들의 유효성 검사"""
    valid_files = []
    invalid_files = []
    
    for file_path in file_paths:
        if os.path.exists(file_path):
            # 지원되는 파일 형식 확인
            if file_path.lower().endswith(('.txt', '.md', '.pdf', '.docx')):
                valid_files.append(file_path)
            else:
                print(f"⚠️  지원되지 않는 파일 형식: {file_path}")
                invalid_files.append(file_path)
        else:
            print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
            invalid_files.append(file_path)
    
    return valid_files, invalid_files


def create_env_file():
    """환경 변수 파일 생성"""
    env_file = ".env"
    if os.path.exists(env_file):
        print(f"📄 .env 파일이 이미 존재합니다: {env_file}")
        return env_file
    
    print("📄 .env 파일을 생성합니다...")
    print("🔑 OpenAI API 키를 입력하세요 (입력한 키는 .env 파일에 저장됩니다):")
    
    api_key = input("API Key: ").strip()
    if not api_key:
        print("❌ API 키가 입력되지 않았습니다.")
        return None
    
    try:
        with open(env_file, 'w', encoding='utf-8') as f:
            f.write(f"OPENAI_API_KEY={api_key}\n")
        print(f"✅ .env 파일이 생성되었습니다: {env_file}")
        return env_file
    except Exception as e:
        print(f"❌ .env 파일 생성 실패: {e}")
        return None


def create_sample_spec():
    """샘플 스펙 파일 생성"""
    sample_content = """
# 시스템 요구사항 명세서 샘플

## 프로젝트 개요
이 시스템은 사용자 관리 및 인증 기능을 제공하는 웹 애플리케이션입니다.

## 주요 기능
1. 사용자 등록 및 로그인
2. 프로필 관리
3. 비밀번호 재설정
4. 관리자 대시보드

## 기술적 요구사항
- 웹 기반 인터페이스
- RESTful API 제공
- 데이터베이스 연동
- 보안 인증 시스템

## 성능 요구사항
- 동시 사용자 1000명 지원
- 응답 시간 2초 이내
- 99.9% 가용성 보장

## 보안 요구사항
- HTTPS 통신 필수
- 비밀번호 암호화 저장
- 세션 관리 및 타임아웃
"""
    
    sample_file = "sample_specification.txt"
    with open(sample_file, 'w', encoding='utf-8') as f:
        f.write(sample_content)
    
    print(f"📄 샘플 스펙 파일이 생성되었습니다: {sample_file}")
    return sample_file


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(
        description="SRS Generation Agent 실행 스크립트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python run.py spec1.txt spec2.pdf          # 여러 파일로 SRS 생성
  python run.py --sample                     # 샘플 파일로 테스트
  python run.py --model gpt-4o spec.txt     # 특정 모델 사용
  python run.py --config production spec.txt # 프로덕션 설정 사용
        """
    )
    
    parser.add_argument('files', nargs='*', help='스펙 파일 경로들')
    parser.add_argument('--model', default='gpt-4o-mini', 
                       choices=['gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo'],
                       help='사용할 OpenAI 모델 (기본값: gpt-4o-mini)')
    parser.add_argument('--config', choices=['development', 'production', 'high_quality', 'fast_processing'],
                       help='사전 정의된 설정 프로필 사용')
    parser.add_argument('--sample', action='store_true',
                       help='샘플 스펙 파일로 테스트 실행')
    parser.add_argument('--output', '-o', default='generated_srs.md',
                       help='출력 파일 경로 (기본값: generated_srs.md)')
    parser.add_argument('--temperature', type=float, default=0.1,
                       help='모델 온도 설정 (기본값: 0.1)')
    parser.add_argument('--setup', action='store_true',
                       help='API 키 설정 도우미 실행')
    
    args = parser.parse_args()
    
    print("🚀 SRS Generation Agent 시작")
    print("=" * 50)
    
    # API 키 확인
    if not check_api_key():
        if args.setup:
            print("\n🔧 API 키 설정 도우미를 실행합니다...")
            create_env_file()
            # .env 파일을 다시 로드
            try:
                from dotenv import load_dotenv
                load_dotenv()
            except ImportError:
                pass
            # 다시 API 키 확인
            if not check_api_key():
                sys.exit(1)
        else:
            print("\n💡 API 키를 설정하려면 --setup 옵션을 사용하세요:")
            print("   python run.py --setup")
            sys.exit(1)
    
    # 파일 처리
    if args.sample:
        print("📄 샘플 스펙 파일을 생성합니다...")
        spec_files = [create_sample_spec()]
    elif args.files:
        valid_files, invalid_files = validate_files(args.files)
        if not valid_files:
            print("❌ 유효한 스펙 파일이 없습니다.")
            sys.exit(1)
        if invalid_files:
            print(f"⚠️  {len(invalid_files)}개의 파일이 무시됩니다.")
        spec_files = valid_files
    else:
        print("❌ 스펙 파일을 제공하거나 --sample 옵션을 사용하세요.")
        print("📋 도움말: python run.py --help")
        sys.exit(1)
    
    print(f"📁 처리할 파일들: {', '.join(spec_files)}")
    
    # 에이전트 설정
    try:
        if args.config:
            print(f"⚙️  설정 프로필 사용: {args.config}")
            config = AgentConfig(getattr(ConfigProfiles, args.config)())
            agent = SRSGenerationAgent(
                model_name=config.model.name,
                temperature=config.model.temperature
            )
        else:
            print(f"🤖 모델: {args.model}, 온도: {args.temperature}")
            agent = SRSGenerationAgent(
                model_name=args.model,
                temperature=args.temperature
            )
    except Exception as e:
        print(f"❌ 에이전트 초기화 오류: {e}")
        sys.exit(1)
    
    # SRS 생성 실행
    print("\n🔄 SRS 문서 생성 중...")
    print("-" * 30)
    
    start_time = datetime.now()
    thread_id = f"srs_generation_{start_time.strftime('%Y%m%d_%H%M%S')}"
    
    try:
        result = agent.generate_srs(spec_files, thread_id=thread_id)
        
        if result["success"]:
            print("✅ SRS 생성 완료!")
            print(f"📊 생성된 요구사항:")
            print(f"   - 기능 요구사항: {len(result['functional_requirements'])}개")
            print(f"   - 비기능 요구사항: {len(result['non_functional_requirements'])}개")
            print(f"   - 시스템 인터페이스: {len(result['system_interfaces'])}개")
            print(f"   - 데이터 요구사항: {len(result['data_requirements'])}개")
            print(f"   - 성능 요구사항: {len(result['performance_requirements'])}개")
            
            # 파일 저장
            if agent.save_srs_document(result["srs_document"], args.output):
                print(f"💾 SRS 문서가 저장되었습니다: {args.output}")
            else:
                print("❌ 파일 저장에 실패했습니다.")
            
            # 처리 시간
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            print(f"⏱️  처리 시간: {duration:.2f}초")
            
            # 오류가 있다면 표시
            if result["errors"]:
                print(f"\n⚠️  처리 중 발생한 오류들:")
                for error in result["errors"]:
                    print(f"   - {error}")
            
        else:
            print(f"❌ SRS 생성 실패: {result['error']}")
            if result["errors"]:
                print("📋 상세 오류:")
                for error in result["errors"]:
                    print(f"   - {error}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️  사용자에 의해 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")
        sys.exit(1)
    
    print("\n🎉 작업이 완료되었습니다!")


if __name__ == "__main__":
    main()
