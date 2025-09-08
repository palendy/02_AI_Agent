#!/usr/bin/env python3
"""
하이브리드 SRS Generation Agent 실행 스크립트
==========================================

기존의 풍부한 요구사항 추출 + 사실 검증을 결합한 하이브리드 접근법

사용법:
    python hybrid_run.py [spec_file1] [spec_file2] ...
    
예시:
    python hybrid_run.py ../99_Texts/JCREspecCLASSIC-3_2.pdf
"""

import os
import sys
import argparse
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from srs_generation_agent import HybridSRSGenerationAgent
except ImportError as e:
    print(f"❌ SRS Generation Agent를 찾을 수 없습니다: {e}")
    print("📋 srs_generation_agent.py 파일이 존재하는지 확인하세요.")
    sys.exit(1)


def check_api_key():
    """OpenAI API 키 확인"""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        print("📋 다음 중 하나의 방법으로 API 키를 설정하세요:")
        print("   1. 환경변수로 설정: export OPENAI_API_KEY='your-api-key'")
        print("   2. .env 파일 생성: echo 'OPENAI_API_KEY=your-api-key' > .env")
        return False
    
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
            if file_path.lower().endswith(('.txt', '.md', '.pdf', '.docx')):
                valid_files.append(file_path)
            else:
                print(f"⚠️  지원되지 않는 파일 형식: {file_path}")
                invalid_files.append(file_path)
        else:
            print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
            invalid_files.append(file_path)
    
    return valid_files, invalid_files


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(
        description="하이브리드 SRS Generation Agent (풍부한 추출 + 사실 검증)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python hybrid_run.py spec.pdf                              # 기본 실행
  python hybrid_run.py --model gpt-4o spec.pdf             # GPT-4o 사용
  python hybrid_run.py --temperature 0.05 spec.pdf         # 낮은 temperature
  python hybrid_run.py --output hybrid_srs.md spec.pdf     # 출력 파일 지정
  python hybrid_run.py --verbose spec.pdf                  # 상세 출력
        """
    )
    
    parser.add_argument('files', nargs='*', help='스펙 파일 경로들')
    parser.add_argument('--model', default='gpt-4o-mini',
                       choices=['gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo'],
                       help='사용할 OpenAI 모델 (기본값: gpt-4o-mini)')
    parser.add_argument('--output', '-o', default='generated_srs.md',
                       help='출력 파일 경로 (기본값: generated_srs.md)')
    parser.add_argument('--temperature', type=float, default=0.1,
                       help='모델 온도 설정 (기본값: 0.1)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='상세한 출력 표시')
    
    args = parser.parse_args()
    
    print("🚀 SRS Generation Agent 시작 (Hybrid Edition)")
    print("=" * 50)
    print("🎯 접근법: 풍부한 추출 + 사실 검증")
    print("=" * 50)
    
    # API 키 확인
    if not check_api_key():
        sys.exit(1)
    
    # 파일 처리
    if not args.files:
        print("❌ 스펙 파일을 제공해주세요.")
        print("📋 도움말: python hybrid_run.py --help")
        sys.exit(1)
    
    valid_files, invalid_files = validate_files(args.files)
    if not valid_files:
        print("❌ 유효한 스펙 파일이 없습니다.")
        sys.exit(1)
    
    if invalid_files:
        print(f"⚠️  {len(invalid_files)}개의 파일이 무시됩니다.")
    
    print(f"📁 처리할 파일들: {', '.join(valid_files)}")
    
    # 하이브리드 설정 정보
    print(f"\n🔧 하이브리드 설정:")
    print(f"   - 모델: {args.model}")
    print(f"   - 온도: {args.temperature}")
    print(f"   - 풍부한 추출: ✅ 활성화")
    print(f"   - 사실 검증: ✅ 활성화")
    print(f"   - Hallucination 탐지: ✅ 활성화")
    
    # 에이전트 초기화
    try:
        agent = HybridSRSGenerationAgent(
            model_name=args.model,
            temperature=args.temperature
        )
        print(f"✅ SRS Generation Agent 초기화 완료 (Hybrid Edition)")
    except Exception as e:
        print(f"❌ 에이전트 초기화 오류: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    
    # SRS 생성 실행
    print(f"\n🔄 하이브리드 SRS 문서 생성 중...")
    print("   1️⃣ 풍부한 요구사항 추출")
    print("   2️⃣ 사실 검증 및 필터링")
    print("   3️⃣ Hallucination 제거")
    print("   4️⃣ 검증된 SRS 생성")
    print("-" * 50)
    
    start_time = datetime.now()
    thread_id = f"hybrid_srs_{start_time.strftime('%Y%m%d_%H%M%S')}"
    
    try:
        result = agent.generate_srs(valid_files, thread_id=thread_id)
        
        if result["success"]:
            print("✅ 하이브리드 SRS 생성 완료!")
            
            # 기본 요구사항 통계
            print(f"\n📊 생성된 요구사항 (검증 후):")
            func_count = len(result.get('functional_requirements', []))
            nonfunc_count = len(result.get('non_functional_requirements', []))
            interface_count = len(result.get('system_interfaces', []))
            data_count = len(result.get('data_requirements', []))
            perf_count = len(result.get('performance_requirements', []))
            total_count = func_count + nonfunc_count + interface_count + data_count + perf_count
            
            print(f"   - 기능 요구사항: {func_count}개")
            print(f"   - 비기능 요구사항: {nonfunc_count}개")
            print(f"   - 시스템 인터페이스: {interface_count}개")
            print(f"   - 데이터 요구사항: {data_count}개")
            print(f"   - 성능 요구사항: {perf_count}개")
            print(f"   - 📝 총 요구사항: {total_count}개")
            
            # 검증 결과 표시
            if 'validation_summary' in result:
                validation = result['validation_summary']
                print(f"\n🛡️  하이브리드 검증 결과:")
                print(f"   - ✅ 검증 통과: {validation.get('total_validated', 0)}개")
                print(f"   - ❌ 검증 실패: {validation.get('total_rejected', 0)}개")
                print(f"   - 📈 검증 성공률: {validation.get('validation_rate', 0):.1%}")
                
                # 거부된 요구사항 정보
                rejected_count = validation.get('total_rejected', 0)
                if rejected_count > 0:
                    reasons = validation.get('rejection_reasons', [])
                    print(f"   - 🚫 주요 거부 이유: {', '.join(reasons[:3])}")
                    
                    if args.verbose and 'rejected_requirements' in result:
                        print(f"\n🚫 거부된 요구사항들 (처음 5개):")
                        for i, req in enumerate(result['rejected_requirements'][:5]):
                            print(f"      {i+1}. {req.get('requirement', 'N/A')[:80]}...")
                            print(f"         이유: {req.get('reason', 'N/A')}")
                            print(f"         신뢰도: {req.get('confidence', 0):.2f}")
                            print()
            
            # 비교 정보 (원본 대비)
            if rejected_count > 0:
                original_total = validation.get('total_validated', 0) + validation.get('total_rejected', 0)
                print(f"\n📊 원본 대비 개선:")
                print(f"   - 원본 추출량: {original_total}개 (검증 전)")
                print(f"   - 최종 검증량: {validation.get('total_validated', 0)}개 (검증 후)")
                print(f"   - 🛡️ Hallucination 제거: {rejected_count}개")
                print(f"   - 📈 신뢰도: 대폭 향상")
            
            # 파일 저장
            if agent.save_srs_document(result["srs_document"], args.output):
                print(f"\n💾 하이브리드 SRS 문서가 저장되었습니다: {args.output}")
            else:
                print("❌ 파일 저장에 실패했습니다.")
            
            # 처리 시간
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            print(f"\n⏱️  총 처리 시간: {duration:.2f}초")
            
            # 경고 및 오류
            if result.get("errors"):
                print(f"\n⚠️  처리 중 발생한 오류들:")
                for error in result["errors"][:5]:  # 최대 5개만 표시
                    print(f"   - {error}")
                if len(result["errors"]) > 5:
                    print(f"   ... 및 {len(result['errors']) - 5}개 추가 오류")
            
        else:
            print(f"❌ 하이브리드 SRS 생성 실패: {result.get('error', '알 수 없는 오류')}")
            if result.get("errors"):
                print("📋 상세 오류:")
                for error in result["errors"]:
                    print(f"   - {error}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️  사용자에 의해 중단되었습니다.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    
    print(f"\n🎉 하이브리드 SRS 생성이 완료되었습니다!")
    print(f"📄 출력 파일: {args.output}")
    print(f"🎯 특징: 풍부한 추출 + 엄격한 사실 검증")


if __name__ == "__main__":
    main()