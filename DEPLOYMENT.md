# Public app deployment

Selene opens directly without login or signup. One FastAPI process on port
8000 serves science, chat, exports, and mission planning. The running app does
not register account endpoints, initialize an account database, or require a
SQLCipher key. Existing account files are left untouched; legacy account
modules and migration tools remain available for offline compatibility.

## Run

After reviewing dependencies according to AGENTS.md, use
requirements-runtime.txt for the lightweight container profile. Optional
SPICE/terrain/Diviner dependencies and datasets require separate preparation.

~~~sh
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1 --limit-concurrency 8 --no-proxy-headers --no-server-header
~~~

Run this from backend. Use the hosting platform's assigned port when applicable.
Do not use --reload in production. Routes include /api/health, /api/status, and
/docs. Optional chatbot credentials stay on the backend.

## Browser and attack protection

~~~dotenv
GROQ_API_KEY=<optional server-side key>
CORS_ORIGINS=https://your-site.example
TRUSTED_HOSTS=your-site.example,localhost,127.0.0.1,backend
CHAT_RATE_LIMIT_PER_MINUTE=12
CHAT_RATE_LIMIT_PER_HOUR=60
API_RATE_LIMIT_PER_MINUTE=300
API_BODY_TIMEOUT_SECONDS=15
BLOCK_BOT_USER_AGENTS=true
TRUST_PROXY_HEADERS=false
TRUSTED_PROXY_IPS=127.0.0.1,::1
~~~

Chat and /api/mission/interpret share a per-IP rolling window: the first 12
requests in 60 seconds are allowed and the next receives HTTP 429 with
Retry-After. The minute limit cannot be disabled or raised above 12. The
additional hourly cap is 60 by default (0 disables only that hourly cap).
The client displays a cooldown when rate-limited. Refreshing the browser does
not reset the server counter.

Counters are per process and reset on server restarts. Use one worker/replica
for this deployment; multiple replicas require a shared gateway/edge limiter.
Users sharing a public IP also share the allowance. These application controls
do not guarantee protection against distributed denial-of-service attacks.

The API checks allowed website origins, hosts and methods, bounds URL/body
sizes, times out slow bodies, rejects compressed bodies and ambiguous lengths,
and attaches security headers including on rejected requests. User-agent
blocking is supplemental; callers can spoof browser user-agents and origins.
Keep credentials server-side. Optional Cloudflare Turnstile verification uses
TURNSTILE_SITE_KEY and TURNSTILE_SECRET_KEY; configure both to enable it.

## Proxy and Docker

~~~sh
docker compose up -d --build
~~~

No SQLCipher key or account volume is needed. Existing account volumes are not
deleted. The backend port is internal to Docker; nginx forwards /api and serves
the frontend on port 5173. Keep VITE_API_BASE_URL empty for this same-origin
setup. Set CORS_ORIGINS and TRUSTED_HOSTS for your public domain.

Compose trusts forwarding headers only from the configured Docker proxy subnet
(default 172.16.0.0/12). Set TRUSTED_PROXY_IPS to match your actual Docker network
or the exact immediate ingress IP/CIDR. Never use arbitrary-client trust.
nginx overwrites X-Forwarded-For/X-Real-IP; Uvicorn's own proxy rewriting is off
so the application can validate the immediate peer. If another proxy/CDN is in
front of nginx, configure its verified real-IP handling before relying on
per-visitor limits: otherwise all visitors can share the proxy's bucket.

Terminate HTTPS at a trusted reverse proxy or hosting ingress. Restrict public
traffic to that ingress. Add edge DDoS/WAF controls for an internet deployment.
For separate frontend hosting set VITE_API_BASE_URL before building, explicitly
allow that origin in the backend and update frontend CSP connect-src; same-origin
hosting avoids that extra configuration.

The lightweight Docker profile disables optional scientific datasets and caps
backend memory at 512 MB. Monitor memory under real workloads.

## Verification

From backend:

~~~sh
python -m pytest tests/test_app_integration.py tests/test_rate_limit.py tests/test_chat_security.py tests/test_bot_guard.py tests/test_api.py
~~~

From the repository root:

~~~sh
npm --prefix frontend run build
~~~
