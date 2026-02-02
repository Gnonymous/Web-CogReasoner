#!/bin/bash
python /code/Web-CogReasoner/src/evaluation/online/auto_eval_webvoyager.py \
    --provider gemini \
    --api_model gemini-2.5-pro \
    --gemini_api_key YOUR_GEMINI_KEY \
    --process_dir /code/Web-CogReasoner/results_webvoyager \
    --output_report_path /code/Web-CogReasoner/results_webvoyager/evaluation_summary.json \
    --max_attached_imgs 15