#!/bin/bash
python /code/Web-CogReasoner/src/evaluation/online/auto_eval_mind2web_cross_task.py \
    --provider gemini \
    --api_model gemini-2.5-pro \
    --gemini_api_key YOUR_GEMINI_API_KEY \
    --process_dir /code/Web-CogReasoner/results_mind2web_task \
    --output_report_path /code/Web-CogReasoner/results_mind2web/eval_summary_mind2web.json \
    --mind2web_json /code/Web-CogReasoner/benchmark/mind2web_test_cross_task.jsonl \
    --max_attached_imgs 15