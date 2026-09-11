.PHONY: install test data golden-candidates split dev-ablation eval calibrate smoke all

install:
	pip install -r requirements.txt

# Runs with no API key and no dataset required — this is the fastest way
# for a grader to confirm the logic (escalation, retrieval, baselines,
# stats, hallucination checks) is actually correct before spending time on
# API keys or the 350MB Kaggle download.
test:
	python -m pytest tests/ -v

data:
	python src/data_prep.py

golden-candidates:
	python eval/golden_set_builder.py --n 200

# Run AFTER hand-labeling data/golden_set.csv per LABELING_GUIDE.md
label-check:
	python eval/label_quality_check.py

# Run AFTER label-check passes
split:
	python eval/split_golden_set.py

# Diagnostic only (dev split) — justifies the retrieval-grounding design
# choice with numbers, does not touch test
dev-ablation:
	python eval/ablation.py

# Headline numbers for REPORT.md — run exactly once, against test split
eval:
	python eval/run_eval.py

calibrate-sample:
	python eval/judge_calibration.py --step sample --n 35

calibrate-compare:
	python eval/judge_calibration.py --step compare

# Full reproduction path assuming data/twcs.csv and GOOGLE_API_KEY are
# already in place, and golden_set.csv is already hand-labeled.
all: data split dev-ablation eval
