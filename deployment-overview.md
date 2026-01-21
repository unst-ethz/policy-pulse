# How Our Deployment Works

## The Flow

```
You push to main
       ↓
GitHub Actions builds a Docker image (~1-2 min)
       ↓
Watchtower (running on our VM) notices the new image
       ↓
Old container is removed, new one starts (Site goes down)
       ↓
App downloads and preprocesses data (~5 min)
       ↓
Site is up again
```

---

## Key Components

**GitHub Actions** — Automatically builds a Docker image whenever we push to `main` using the `Dockerfile`. The image contains our code and dependencies.

**Watchtower** — A service running on our VM that checks every 60 seconds if there's a new image. When it finds one, it pulls it and restarts the app.

**Nginx** — A reverse proxy that sits in front of our app. It handles incoming web requests and forwards them to our Dash app. Can use Nginx to add SSL later.

**Gunicorn** — The server that runs our Python app. We use the `--preload` flag, which loads the resolution data once, processes it and then spawns worker processes that share the processed data.

---

## What Happens on Deploy

When we push to `main`:
1. The current container is removed
2. A new container starts with the updated code
3. The app redownloads and preprocesses all data from scratch
4. **Site is unavailable for ~5 minutes** because of refetching and processing data

---

## Checking Status

Everything runs with Docker. To see what's happening on the VM:

```bash
cd /opt/dash-app

# View logs for the app
docker compose logs -f app

# View logs for everything
docker compose logs -f
```

---

## Code Requirement

Your Dash app must expose the server object for Gunicorn:
```python
app = dash.Dash(__name__)
server = app.server  # ← Required for deployment
```

Without this line, the deployment will fail.

---

## File Locations

All Docker configuration lives on the VM at:
```
/opt/dash-app/
```

---

## Next Steps

- [ ] Add SSL/HTTPS for security
- [ ] Configure firewall
