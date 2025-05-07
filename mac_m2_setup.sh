#!/bin/bash
echo "🔧 安装 Python 环境并配置依赖..."
conda create -n poker_rl python=3.9 -y
conda activate poker_rl
pip install -r requirements.txt
echo "✅ 环境安装完成"
