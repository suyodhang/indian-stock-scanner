# AI-Based Indian Stock Market Scanner

## Project Structure
- `config/`: settings and stock universe
- `data_collection/`: NSE, BSE, Yahoo and live feed fetchers
- `analysis/`: technical, pattern, volume, fundamental analysis
- `ai_models/`: prediction, breakout, sentiment, anomaly, training pipeline
- `scanners/`: momentum, breakout, reversal, volume, custom scans
- `alerts/`: telegram, email, webhook alerts
- `database/`: SQLAlchemy models and DB manager
- `dashboard/`: Streamlit app
- `main.py`: orchestrator

## Quick Start
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py --mode once
```

## Dashboard
```bash
streamlit run dashboard/app.py
```

## Scheduled Mode
```bash
python main.py --mode scheduled
```

## Docker
```bash
docker-compose up -d
```
