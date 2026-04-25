# CodeRepo_Agent-Collaboration-Simulation

## Setup

```bash
pip install -r requirements.txt
```

## Experiments (4)

### 1) ShapeFactory

```bash
python -m src.cli configs/shapefactory_time_mode_example.yml --print-actions
```

### 2) MapTask

Formal run:

```bash
python -m src.cli configs/maptask_example.yml --print-actions
```

Debug run:

```bash
python -m src.cli configs/maptask_example.yml --max-steps 3 --dry-run --print-actions
```

### 3) DayTrader

```bash
python -m src.cli configs/daytrader_example.yml --print-actions
```

### 4) Hidden Profile

```bash
python -m src.cli configs/hidden_profile_example.yml --print-actions
```