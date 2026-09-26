# tsforge — Docker 镜像（干净环境一键复现）
# 构建： docker build -t tsforge .
# 运行： docker run --rm tsforge python -m tsforge.cli demo
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 先装依赖层（利用缓存）
COPY requirements.lock.txt /app/requirements.lock.txt
RUN pip install --no-cache-dir -r requirements.lock.txt

# 再拷源码
COPY . /app

# 默认运行 demo（可用 `python -m tsforge.cli version` 等覆盖）
CMD ["python", "-m", "tsforge.cli", "demo"]
