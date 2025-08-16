#!/bin/bash

# 컨테이너 내부에서 실행할 설정 스크립트
# 사용법: 컨테이너 내부에서 ./setup_container.sh 실행


echo "📦 Python 의존성을 설치합니다..."
pip install -r requirements.txt

echo "🔗 패키지를 개발 모드로 설치합니다..."
pip install -e .

echo "✅ 설정이 완료되었습니다!"
echo "🚀 이제 benchmarks 디렉토리에서 테스트를 실행할 수 있습니다."
echo "   cd benchmarks"
echo "   ./test_single.sh"
