# Selene — Backend API

**Lunar Mission Planning Explorer · NASA International Space Apps Challenge 2026 · Bhopal**

The FastAPI service behind Selene. It computes lunar environmental conditions (Sun and Earth geometry, illumination, temperature), forecasts and Excel exports, requirements-based mission-operations screening, NASA dataset analysis, LangGraph mission workflows and the optional Groq-powered assistant.

- **Frontend:** [khalahuja/nsac_frontend](https://github.com/khalahuja/nsac_frontend), live at https://selenenasa.netlify.app
- **Interactive API docs:** `/docs` on any running instance

> Selene is a research and educational prototype for preliminary planning. Results are labelled with their data source and fidelity. It is not a flight-certified mission-design or landing-safety system.

## Tech stack

| Area | Libraries |
|---|---|
| API | FastAPI, Uvicorn, Pydantic, pydantic-settings |
| Numerics and export | NumPy, SciPy, pandas, openpyxl |
| Mission workflows | LangGraph |
| Optional science | SpiceyPy (SPICE kernels), rasterio and xarray (terrain DEMs) |
| Assistant | Groq API (optional, server-side key) |

## Project structure

```text
app/
  main.py             App factory and router registration
  config.py           Settings loaded from environment variables / .env
  http_app.py         CORS, trusted hosts, body limits, middleware
  rate_limit.py       Per-IP rate limiting
  bot_guard.py        Origin checks and crawler blocking
  chat_security.py    Optional Cloudflare Turnstile verification
  security_headers.py Response security headers
  api/                Routers: science (routes.py), nasa, mission, chat
  models/             Request/response schemas
  services/           Ephemeris, illumination, thermal, terrain, forecasts, mission planning, exports
  datasets/README.md  How to obtain and prepare the optional NASA datasets
data/moon_craters.json  Bundled lunar feature catalogue
scripts/              NASA data preparation (fetch_nasa_data.py)
tests/                pytest suite
DEPLOYMENT.md         Production hosting, rate limits and security configuration
Dockerfile            Lightweight production image
```

## Run locally

**Prerequisites:** Python 3.12 (the version the Docker image uses).

The app resolves its `.env` file and the optional `data/` folder relative to the directory **above** this repository. Clone it into a folder named `backend`:

```bash
mkdir nsac && cd nsac
git clone https://github.com/khalahuja/nsac_backend.git backend
cd backend

python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate

pip install -r requirements-runtime.txt
```

Create `backend/.env`, using the options in [Configuration](#configuration), then start the server:

```bash
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

| URL | Purpose |
|---|---|
| http://127.0.0.1:8000/docs | Interactive API documentation |
| http://127.0.0.1:8000/api/health | Liveness check |
| http://127.0.0.1:8000/api/status | Which science providers are active |

**Dependency sets**

- `requirements-runtime.txt` is the lightweight set used in production. The app falls back to its analytic models.
- `requirements.txt` is the full set: it adds SciPy, pandas, SpiceyPy, rasterio, xarray and pytest. Install it to run the tests or to use the real NASA datasets.

## Configuration

Settings come from environment variables or from `backend/.env`. These are the most important ones:

| Variable | Default | Purpose |
|---|---|---|
| `CORS_ORIGINS` | localhost dev origins | Comma-separated browser origins allowed to call the API, e.g. your Netlify URL |
| `TRUSTED_HOSTS` | `localhost,127.0.0.1,testserver,backend` | Hostnames the API answers to, e.g. `your-service.onrender.com` |
| `GROQ_API_KEY` | *(empty)* | Enables the chatbot. Keep it server-side only |
| `GROQ_MODEL` / `GROQ_FALLBACK_MODELS` | — | Primary model, plus fallbacks used when quota is reached |
| `CHAT_RATE_LIMIT_PER_MINUTE` / `_PER_HOUR` | `12` / `60` | Per-IP chat limits (the minute limit is capped at 12) |
| `API_RATE_LIMIT_PER_MINUTE` | `300` | Per-IP limit for the whole API (`0` disables it) |
| `TRUST_PROXY_HEADERS` / `TRUSTED_PROXY_IPS` | `false` / `127.0.0.1,::1` | Read the client IP from `X-Forwarded-For`, but only from the listed proxy IPs |
| `TURNSTILE_SITE_KEY` / `TURNSTILE_SECRET_KEY` | *(empty)* | Optional Cloudflare Turnstile bot check for chat |
| `USE_SPICE` | `false` | Use JPL/NAIF SPICE kernels instead of the analytic ephemeris |
| `USE_TERRAIN_DEM` / `USE_TERRAIN_SHADOWS` | `false` | Use a LOLA DEM for elevation, slope and horizon shadows |
| `USE_DIVINER` | `false` | Use LRO Diviner temperature climatologies |
| `SPICE_DATA_DIR`, `TERRAIN_DATA_DIR`, `DIVINER_DATA_DIR`, `NASA_DATA_DIR` | `../data/...` | Dataset locations |

### Optional NASA datasets

This repository doesn't include the datasets because together they total about 300 MB. Every provider is off by default, and the API works without them, returning clearly labelled approximate results. To enable them, place the files in a `data/` folder next to this one:

```text
nsac/
  backend/          ← this repository
  data/
    spice/          naif0012.tls, de440s.bsp, pck00011.tpc, moon_pa_de440_200625.bpc, moon_de440_250416.tf
    terrain/        ldem_80s_80m.lbl + .img (LOLA, 80°S to pole, 80 m/px)
    diviner/        prepared Diviner climatology (.npz + manifest)
    nasa/           regional LOLA products, hydrogen and water context
```

Then set the matching `USE_*` flags to `true`. Confirm with `GET /api/status`, because a provider that fails to load falls back to its approximation with a logged warning. [app/datasets/README.md](app/datasets/README.md) covers sources and preparation, and `scripts/fetch_nasa_data.py` handles the import.

## API overview

| Method and path | Purpose |
|---|---|
| `GET /api/health` | Liveness |
| `GET /api/status` | Active providers and fidelity |
| `GET /api/moon/features` | Lunar feature catalogue |
| `POST /api/environment` | Conditions at one coordinate and instant |
| `POST /api/timeline` | Sampled conditions over a time range |
| `POST /api/forecast` | Environmental forecast time series |
| `POST /api/export/xlsx` | Forecast as an Excel workbook |
| `POST /api/surface/screen` | DEM skyline, moving shadow and landing-footprint screen |
| `POST /api/surface/skyline` | Terrain horizon around a site |
| `POST /api/surface/horizons` | Reference Sun and Earth angles from JPL Horizons |
| `POST /api/mission/operations` | Rank landing dates against surface-stay requirements |
| `POST /api/mission/simulate` | Structured mission stages (LangGraph) |
| `POST /api/mission/simulate/stream` | The same, streamed stage by stage |
| `POST /api/mission/windows` | Mission window search |
| `POST /api/mission/site-ranking` | Candidate site ranking |
| `POST /api/mission/interpret` | Natural-language request to typed mission inputs |
| `GET /api/mission/config` | Mission planner configuration |
| `GET /api/nasa/catalog` | Installed NASA products and contextual layers |
| `GET /api/nasa/imagery/{layer_id}` | Moon Trek imagery layer |
| `POST /api/nasa/terrain` | Regional terrain analysis |
| `POST /api/nasa/thermal` | Historical temperature comparison |
| `POST /api/nasa/water` | Water evidence and hydrogen context |
| `POST /api/nasa/validate` | Reference comparison report |
| `GET /api/chat/config` | Assistant availability and verification settings |
| `POST /api/chat` | Assistant chat (rate-limited) |

Example request:

```bash
curl -X POST http://127.0.0.1:8000/api/environment \
  -H "Content-Type: application/json" \
  -d '{"latitude": -89.54, "longitude": 0.0, "datetime_utc": "2026-11-15T12:00:00Z", "include_sun_events": true}'
```

`/docs` has the complete schemas and constraints.

## Deploy

### Docker

```bash
docker build -t selene-backend .
docker run -p 8000:8000 \
  -e CORS_ORIGINS=https://your-frontend.example \
  -e TRUSTED_HOSTS=your-api.example \
  -e GROQ_API_KEY=... \
  selene-backend
```

The image installs `requirements-runtime.txt`, runs as a non-root user, and starts a single Uvicorn worker on port 8000 with a built-in health check. Use one worker or replica: the rate-limit counters live in memory.

### Render

This repository includes a [render.yaml](render.yaml) Blueprint, which defines a free Docker web service with all of its environment variables.

1. In the Render dashboard, go to **New → Blueprint** and select this repository.
2. When Render prompts for `GROQ_API_KEY`, enter it. Leave it empty to disable the chatbot.
3. Click **Apply**. When the deploy finishes, open `https://<your-service>.onrender.com/api/health` to check it's running.
4. Put `https://<your-service>.onrender.com` into the frontend's `VITE_API_BASE_URL` and rebuild the frontend.

To serve a different frontend, change `CORS_ORIGINS` in `render.yaml` or in the dashboard.

On Render's free plan the service sleeps after about 15 minutes idle, so the first request afterwards takes 30–60 seconds.

[DEPLOYMENT.md](DEPLOYMENT.md) covers rate limits, proxy trust, security headers and hardening in detail.

## Tests

```bash
pip install -r requirements.txt
python -m pytest
```

Run a quicker security and API subset with:

```bash
python -m pytest tests/test_app_integration.py tests/test_rate_limit.py tests/test_chat_security.py tests/test_bot_guard.py tests/test_api.py
```
