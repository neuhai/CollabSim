# Experiments

Run manifests, outputs, and reproducibility artifacts will live here.

## Example run
- Counter task: `python -m src.cli configs/counter_example.yml --run-id counter_v1 --output-dir experiments/counter_v1`
- Accumulator task: `python -m src.cli configs/accumulator_example.yml --run-id accumulator_v1 --output-dir experiments/accumulator_v1`
- Hidden-profile task: `python -m src.cli configs/hidden_profile_example.yml --run-id hidden_profile_v1 --output-dir experiments/hidden_profile_v1`
- Hidden-profile demo: `python -m src.cli configs/hidden_profile_demo.yml --run-id hidden_profile_demo --output-dir experiments/hidden_profile_demo`
- Outputs: `events.jsonl`, `probes.jsonl`, `metrics.json`, `run_manifest.json`
