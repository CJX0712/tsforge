# tsforge — Makefile（一键复现）
# 作者：晨星

PYTHON ?= python3
VENV  ?= .venv

.PHONY: help setup test demo clean lock

help:
	@echo "tsforge — 时间序列预测系统（作者: 晨星）"
	@echo "  make setup   创建 venv 并安装锁定依赖"
	@echo "  make test    运行 pytest 单元测试"
	@echo "  make demo    运行端到端基准 -> benchmark.json"
	@echo "  make lock    重新生成 requirements.lock.txt"
	@echo "  make clean   清理产物与 venv"

setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install -U pip
	$(VENV)/bin/pip install -r requirements.lock.txt

test:
	$(VENV)/bin/python -m pytest tests/ -q -W ignore::UserWarning

demo:
	$(VENV)/bin/python -m tsforge.cli demo

lock:
	$(VENV)/bin/pip freeze > requirements.lock.txt

clean:
	rm -rf $(VENV) __pycache__ .pytest_cache benchmark.json demo_run.log
	find . -name '*.pyc' -delete
