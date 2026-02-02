# Repository Guidelines

## Project Structure & Module Organization
本仓库以 Python 与脚本为主：`src/` 放核心代码，`src/evaluation/` 为评测任务脚本，`src/online_exploration/` 为在线探索相关模块。`scripts/` 提供训练与评测入口脚本，`data/` 预期放数据集与测试数据（默认未随仓库发布），`benchmark/` 放基准相关资源，`assets/` 为 README 图片。依赖在 `requirements.txt`，打包信息在 `setup.py`。

## Build, Test, and Development Commands
常用命令（按需调整路径与环境变量）：
- 安装依赖：`pip install -r requirements.txt`
- 开发安装（含格式化/检查工具）：`pip install -e .[dev]`
- 训练入口：`./scripts/train.sh stage1|stage2|stage3`（支持 `-m/-d/-o/-c` 覆盖路径）
- Web-CogBench 评测：`./scripts/evaluate_web_cogbench.sh --model-endpoint http://localhost:8080/v1`
- 其他评测：`./scripts/evaluate_visualwebbench.sh`、`./scripts/evaluate_webvoyager.sh`、`./scripts/evaluate_mind2web.sh`

## Coding Style & Naming Conventions
Python 代码遵循 4 空格缩进与 PEP8 风格，模块与函数优先 `snake_case`。脚本命名保持描述性（如 `evaluate_web_cogbench.sh`）。若启用格式化与静态检查，建议顺序：`isort src` → `black src` → `flake8 src`（工具来自 `.[dev]` 依赖）。

## Testing Guidelines
仓库未提供单元测试目录；当前主要通过评测脚本验证功能与性能。评测日志默认写入 `results/` 下的任务子目录。建议在提交前至少跑一次与改动相关的评测脚本，并附上关键日志或指标。

## Commit & Pull Request Guidelines
近期提交历史以 `Update README.md` 为主，偶见 `docs:` 前缀（如 `docs: Update README.md`）。建议新提交采用清晰的前缀与主题（例如 `feat: add stage3 config`、`docs: update training notes`）。PR 请包含：改动说明、关联 issue（如有）、必要的评测命令与结果摘要，模型输出或指标变化请附截图或日志片段。

## Configuration & Security Notes
训练脚本默认使用 Conda 环境 `KG-WebVoyager`，并读取 `WANDB_API_KEY` 进行 W&B 登录；评测脚本可通过 `MODEL_ENDPOINT`、`MODEL_NAME` 等环境变量配置服务端点。涉及私有模型路径或数据集时，请避免提交敏感路径与密钥。
