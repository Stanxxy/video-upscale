# Service Runbook

## Pre-flight Checks

Before starting any service:
1. Verify `.env` exists in `bjj-vision-backend/` (copy from `.env-tmplt` if missing)
2. Verify `.venv` exists: `ls bjj-vision-backend/.venv/bin/python`
3. Verify poetry deps installed: `cd bjj-vision-backend && poetry install`

## Starting Services

### Single Service
```bash
cd bjj-vision-backend/app/services/{service_name}
../../.venv/bin/python -m uvicorn src.app:app --host 0.0.0.0 --port {PORT} --reload
```

### Multiple Services
Run each in a separate terminal/background process. Start in dependency order:
1. User Authentication (8001) — no deps
2. Video Management (8003) — depends on auth
3. Other services as needed

## Health Checks

```bash
# Quick check
curl -sf http://localhost:{PORT}/health && echo "OK" || echo "FAIL"

# Detailed check
curl -s http://localhost:{PORT}/health | python -m json.tool
```

## Common Issues

### Port Already in Use
```bash
lsof -i :{PORT}
# Note the PID, then:
kill {PID}
```

### ModuleNotFoundError
```bash
cd bjj-vision-backend
poetry install
# If shared_lib specifically:
poetry install  # It's a path dependency, reinstall picks it up
```

### Supabase Connection Refused
- Check `SUPABASE_URL` in `.env`
- Verify network connectivity to Supabase cloud
- Check if API key is expired

### LocalStack Unreachable
- Verify Tailscale VPN is connected
- Test: `curl http://100.79.167.101:4566/_localstack/health`
- Check DynamoDB tables: `aws --endpoint-url=http://100.79.167.101:4566 dynamodb list-tables --region us-east-1`

### Service Crashes on Startup
1. Check the traceback for missing env vars
2. Check for syntax errors in recent changes
3. Try running with `--no-reload` to see if it's a reload loop
4. Check if required infrastructure is available

## Logs
- Services log to stdout by default
- For persistent logging, redirect: `... 2>&1 | tee service.log`
- Check recent errors: look for `ERROR` or `Traceback` in output
