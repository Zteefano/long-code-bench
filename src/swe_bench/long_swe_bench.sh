#!/bin/bash

python swebench/collect/get_tasks_pipeline.py \
--repos $(cat ../../repo_1M.txt) \
--path_prs "../../swe_bench_1M" \
--path_tasks "../../swe_bench_1M"