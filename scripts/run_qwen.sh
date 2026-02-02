python -u /code/Web-CogReasoner/src/online_exploration/run_qwen.py \
    --test_file /code/Web-CogReasoner/benchmark/WebVoyager.jsonl \
    --headless \
    --max_iter 15 \
    --max_attached_imgs 3 \
    --save_accessibility_tree \
    --output_dir /code/Web-CogReasoner/results_qwen_webvoyager \
    --api_localhost http://localhost:8080/v1 > /code/Web-CogReasoner/results_qwen_webvoyager.log