# 🎯 Kriterion Regime Filter

Dashboard Streamlit per il filtraggio e il dimensionamento dei Trading System
algoritmici in base al regime di mercato (Entropia + Ergodicità sull'S&P 500).

---

## Cosa fa

Per ogni Trading System (equity esportate da TradeStation/MultiCharts):

1. **Calcola il regime SPX** combinando due misure:
   - **Shannon Entropy** (rolling 63g) → 3 stati: Bassa / Media / Alta
   - **Ergodicità SEM** (rolling 252g) → 2 stati: Ergodico / Non Ergodico
   - → **6 regimi totali** (es. `Bassa|Ergodico`, `Alta|Non Ergodico`, ...)

2. **Ottimizza le regole di esposizione** per ogni TS via backtest storico:
   | Condizione | Moltiplicatore | Esposizione |
   |-----------|---------------|-------------|
   | Mean PnL regime ≥ 1.4× media globale | ×1.5 | 🟢 BOOST |
   | Mean PnL regime ≥ 0.3× media globale | ×1.0 | 🟡 STANDARD |
   | Mean PnL regime ≥ $0 | ×0.5 | 🟠 RIDOTTO |
   | Mean PnL regime < $0 | ×0.0 | 🔴 INIBITO |

3. **Mostra le equity curve** baseline vs adjusted e lo stato corrente.

4. **Invia un report Telegram** giornaliero alle 16:30 ET (dopo chiusura SPX).

---

## Struttura del repository

```
kriterion-regime-filter/
├── app.py                       # Dashboard Streamlit principale
├── notify.py                    # Notificatore Telegram standalone
├── requirements.txt
├── .env.example                 # Template variabili d'ambiente (notify.py)
├── .streamlit/
│   ├── config.toml              # Tema dark
│   └── secrets.toml.example     # Template secrets (app.py)
└── src/
    ├── equity_loader.py         # Parsing equity TradeStation + download Drive
    ├── spx_data.py              # Fetch SPX da EODHD
    ├── entropy_calc.py          # Shannon + Permutation Entropy
    ├── ergodicity_calc.py       # Ergodicità SEM
    ├── regime_engine.py         # Classificazione regime composita
    ├── optimizer.py             # Ottimizzazione soglie esposizione per TS
    ├── exposure_engine.py       # Equity curve adjusted
    ├── charts.py                # Grafici Plotly
    └── telegram_bot.py          # Formattazione e invio messaggi Telegram
```

---

## Setup e Deploy

### 1. Clone del repository

```bash
git clone https://github.com/tuousername/kriterion-regime-filter.git
cd kriterion-regime-filter
pip install -r requirements.txt
```

### 2. Configurazione Secrets (per Streamlit Cloud / locale)

Crea `.streamlit/secrets.toml` (NON committare su GitHub):

```toml
EODHD_API_KEY      = "la-tua-chiave-eodhd"
TELEGRAM_BOT_TOKEN = "token-del-bot-telegram"
TELEGRAM_CHAT_ID   = "il-tuo-chat-id"
```

Su **Streamlit Cloud**: vai in *Settings → Secrets* e incolla il contenuto.

### 3. Run locale

```bash
streamlit run app.py
```

### 4. Deploy su Streamlit Cloud

1. Push del repo su GitHub
2. [streamlit.io/cloud](https://streamlit.io/cloud) → "New app" → scegli il repo
3. Imposta i Secrets in *Advanced Settings*
4. Clicca **Deploy**

---

## Setup Telegram Bot

### Creazione Bot

1. Apri Telegram e cerca **@BotFather**
2. Invia `/newbot` e segui le istruzioni
3. Salva il **token** ricevuto (formato: `1234567890:ABCdefGHIjklMNOpqrSTUvwxYZ`)

### Ottenere il Chat ID

1. Avvia una chat con il tuo bot (invia `/start`)
2. Visita: `https://api.telegram.org/bot<TOKEN>/getUpdates`
3. Nel JSON trova `"chat": {"id": XXXXXXXXX}` → questo è il tuo `TELEGRAM_CHAT_ID`

Per un **gruppo**: aggiungi il bot al gruppo, invia un messaggio, poi usa getUpdates.
Il chat_id di un gruppo inizia con `-` (es. `-1001234567890`).

---

## Setup Notifica Giornaliera (notify.py)

Il mercato US chiude alle **16:00 ET**. Il report viene inviato alle **16:30 ET**
(con buffer di 30 min per garantire che i dati EODHD siano aggiornati).

### Configurazione .env

Copia `.env.example` come `.env` e inserisci le chiavi:

```bash
cp .env.example .env
# Edita .env con le tue chiavi API
```

### Linux / macOS (cron)

```bash
crontab -e
```

Aggiungi questa riga (esegue alle 21:30 UTC = 16:30 ET estate / 17:30 ET inverno):

```cron
30 21 * * 1-5 cd /path/to/kriterion-regime-filter && /usr/bin/python3 notify.py >> /tmp/kriterion_notify.log 2>&1
```

**Nota DST:** d'estate (DST USA attivo, UTC-4) 21:30 UTC = 17:30 ET.
Per coprire sempre le 16:30 ET:
- Estate (marzo-novembre): cron a `30 20 * * 1-5` (20:30 UTC = 16:30 ET DST)
- Inverno (novembre-marzo): cron a `30 21 * * 1-5` (21:30 UTC = 16:30 ET)

Oppure usa un'unica regola a **21:30 UTC** (17:30 ET estate) come buffer sicuro.

### Windows (Task Scheduler)

1. Apri **Task Scheduler** → "Create Basic Task"
2. Nome: `Kriterion Notify`
3. Trigger: **Daily** → 23:30 CEST (estate) o 22:30 CET (inverno)
4. Azione: **Start a program**
   - Program: `python.exe` (o path completo)
   - Arguments: `C:\path\to\kriterion-regime-filter\notify.py`
   - Start in: `C:\path\to\kriterion-regime-filter`
5. Assicurati che il Task venga eseguito anche se l'utente non è loggato

### Test manuale

```bash
python notify.py
```

Oppure usa il pulsante **"📲 Test Telegram"** nella sidebar della dashboard.

---

## Aggiornamento Equity Files

Per aggiornare i file equity:
1. Esporta le nuove equity da TradeStation/MultiCharts
2. Carica i file `.txt` nella cartella Google Drive
3. Nella dashboard, clicca **"🔄 Ricarica Equity da Drive"**

Il sistema si adatta automaticamente ai nuovi file (aggiunta/rimozione TS).

---

## Formato File Equity TradeStation

I file `.txt` devono essere nel formato standard di esportazione TradeStation:

```
ENTRY_DATE|TIME;EXIT_DATE|TIME;direction;contracts;pnl_total;...
1230501|1600;1230502|1600;buy;1;700.00;0;700.00;...
```

- **Data format**: YYYMMDD dove YYY = anni dal 1900
  - Es: `1230501` → 1900+123=2023, mese=05, giorno=01 → 2023-05-01
- I file devono terminare con `_USD.txt`
- Sistemi short: il nome deve contenere `short` o `copertura`

---

## Dipendenze

| Libreria | Versione | Uso |
|---------|---------|-----|
| streamlit | ≥1.32 | Dashboard web |
| pandas | ≥2.0 | Manipolazione dati |
| numpy | ≥1.26 | Calcoli numerici |
| plotly | ≥5.20 | Grafici interattivi |
| requests | ≥2.31 | API EODHD + Telegram |
| scipy | ≥1.12 | Shannon Entropy |
| gdown | ≥4.7.3 | Download Google Drive |
| python-dotenv | ≥1.0 | Variabili .env per notify.py |

---

## Kriterion Quant

Progetto sviluppato per [kriterionquant.com](https://kriterionquant.com)

Strategia e logica quantitativa: Luca Guidoni
Implementazione: Kriterion Quant + Claude AI
