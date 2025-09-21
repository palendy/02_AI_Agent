#!/usr/bin/env python3
"""
Streamlit App Launcher
AI Agent Chatbot Streamlit 웹 인터페이스 실행 스크립트
"""

import os
import sys
import subprocess
from pathlib import Path


def check_requirements():
    """필요한 패키지 설치 확인"""
    try:
        import streamlit
        print("✅ Streamlit이 설치되어 있습니다.")
        return True
    except ImportError:
        print("❌ Streamlit이 설치되지 않았습니다.")
        print("다음 명령어로 설치하세요: pip install -r requirements.txt")
        return False


def check_env_file():
    """환경 변수 파일 확인"""
    env_file = Path(".env")
    if env_file.exists():
        print("✅ .env 파일이 존재합니다.")
        return True
    else:
        print("⚠️ .env 파일이 없습니다.")
        print("다음 환경 변수들을 설정하세요:")
        print("- OPENAI_API_KEY")
        print("- TAVILY_API_KEY")
        print("- GITHUB_TOKEN (선택사항)")
        print("- GITHUB_REPOSITORIES (선택사항)")
        return False


def launch_streamlit():
    """Streamlit 앱 실행"""
    try:
        # 앱 파일 경로
        app_file = Path("view/app.py")
        
        if not app_file.exists():
            print(f"❌ 앱 파일을 찾을 수 없습니다: {app_file}")
            return False
        
        print("🚀 Streamlit 앱을 시작합니다...")
        print("📱 브라우저에서 http://localhost:8501 을 열어주세요.")
        print("⏹️  종료하려면 Ctrl+C를 누르세요.")
        print("-" * 50)
        
        # Streamlit 실행
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            str(app_file),
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--browser.gatherUsageStats", "false"
        ]
        
        subprocess.run(cmd)
        return True
        
    except KeyboardInterrupt:
        print("\n👋 앱이 종료되었습니다.")
        return True
    except Exception as e:
        print(f"❌ 앱 실행 실패: {e}")
        return False


def main():
    """메인 함수"""
    print("🤖 AI Agent Chatbot Streamlit Launcher")
    print("=" * 50)
    
    # 요구사항 확인
    if not check_requirements():
        sys.exit(1)
    
    # 환경 변수 파일 확인
    check_env_file()
    
    # Streamlit 앱 실행
    launch_streamlit()


if __name__ == "__main__":
    main()
