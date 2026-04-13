# Next Step: LSTM/GRU baseline

## Files created

- `src/models/sequential/data.py`
- `src/models/sequential/model.py`
- `src/models/sequential/train.py`

## What this baseline does

1. Builds transaction sequences per client from `data/raw/transactions.csv`.
2. Aligns each credit to a client sequence and target `en_defaut` from `data/raw/credits.csv`.
3. Trains two recurrent models:
   - LSTM baseline
   - GRU baseline
4. Saves artifacts in `models/`:
   - `sequential_lstm.pt`
   - `sequential_gru.pt`
   - `sequential_metadata.json`

## Install dependency

PyTorch is required:

```bash
pip install torch
```

## Run training

```bash
python -m src.models.sequential.train
```

