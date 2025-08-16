#!/bin/bash

# Docker 컨테이너 실행 스크립트
# 사용법: ./run_docker.sh [container_name]

# 컨테이너 이름 설정 (기본값: autotp-container)
if [ -z "$1" ]; then
    echo "컨테이너 이름을 입력해주세요."
    exit 1
fi
CONTAINER_NAME=${1:-autotp-container}

echo "🚀 AutoTP Docker 컨테이너를 시작합니다..."
echo "📦 컨테이너 이름: $CONTAINER_NAME"
echo "💾 볼륨: ../ -> /workspace/"

# 기존 컨테이너가 있으면 제거
if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
    echo "기존 컨테이너가 존재합니다."
    exit 1
fi

if [ ! -d "../cache" ]; then
    mkdir -p ../cache
fi

# 새 컨테이너 실행
docker run -it \
    --gpus all \
    --ipc=host \
    --net=host \
    --name=$CONTAINER_NAME \
    -v ../:/workspace \
    nvcr.io/nvidia/pytorch:23.10-py3 \
    bash

echo "✅ 컨테이너가 종료되었습니다."
