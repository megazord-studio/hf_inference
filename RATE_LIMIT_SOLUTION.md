# Quick Setup: Avoid HF Rate Limits

**Your Scenario**: Laptop without HF token → Remote server with HF token

## TL;DR

```bash
# On your laptop
export REMOTE_INFERENCE_ENABLED=true
export REMOTE_INFERENCE_TOKEN=SUPERSECRET

# Restart server
.venv/bin/python -m uvicorn app.main:app --reload

# Open browser, check the "Use remote inference" box
# → No more 429 errors! 🎉
```

## Why This Works

```
Your Laptop (no token)
    ↓
Remote Server (has HF_TOKEN=hf_xxx)
    ↓
HuggingFace API
    ↓
✅ No rate limit!
```

## Usage Patterns

### Pattern 1: Always Use Remote (Recommended)
```bash
# Add to ~/.bashrc
export REMOTE_INFERENCE_ENABLED=true
export REMOTE_INFERENCE_TOKEN=SUPERSECRET

# Check the box in UI by default
# Or set automatic forwarding in code
```

### Pattern 2: Selective Remote
```bash
# Only enable when needed
export REMOTE_INFERENCE_ENABLED=false

# Check "Use remote" box only for heavy models
# or when you hit rate limit
```

### Pattern 3: Model Discovery
```bash
# Fetch models from remote (has HF token, no rate limit)
curl "http://127.0.0.1:8000/models?task=text-generation&include_remote=true"
```

## Benefits for Your Workflow

### ✅ Before (Rate Limited)
```
Browse models → 429 Error after 10 requests
Try to run inference → "Rate limit exceeded"
Wait 1 hour → Try again → Still limited
```

### ✅ After (Remote Forwarding)
```
Browse models → Include remote = no rate limit
Run inference → Check remote box = works!
Test 100 models → No issues! 🚀
```

## Best Practices

### 1. Default to Remote
Since remote has the HF token, use it by default:

```bash
# In your shell rc file
export REMOTE_INFERENCE_ENABLED=true
export REMOTE_INFERENCE_TOKEN=SUPERSECRET
```

### 2. Use Local for UI Development Only
When working on CSS/JS/HTML:
```bash
# Temporarily disable remote
export REMOTE_INFERENCE_ENABLED=false

# Work on UI, no actual inference needed
# When ready to test real models, re-enable
```

### 3. Fetch Models from Remote
```bash
# Always use include_remote=true to avoid rate limits
curl "http://127.0.0.1:8000/models?task=anything&include_remote=true"
```

## Troubleshooting

### Still Getting 429?

Check environment:
```bash
env | grep REMOTE
# Should show:
# REMOTE_INFERENCE_ENABLED=true
# REMOTE_INFERENCE_TOKEN=SUPERSECRET
```

Check server logs for "X-Inference-Source: remote"

### Slow Response?

Network latency is normal. Remote adds ~100-300ms for roundtrip.

### Can't Connect?

Test connectivity:
```bash
curl https://inference.kube.megazord.studio/healthz
# Should return: {"status":"ok",...}
```

## Team Setup

Share these settings with teammates:

```bash
# Everyone adds to their ~/.bashrc
export REMOTE_INFERENCE_ENABLED=true
export REMOTE_INFERENCE_TOKEN=SUPERSECRET
export REMOTE_INFERENCE_ENDPOINT=https://inference.kube.megazord.studio/inference
```

Now everyone can develop locally without HF tokens! 🎉

## Summary

**Problem**: Laptop gets rate-limited by HuggingFace  
**Solution**: Forward to remote server that has HF_TOKEN  
**Result**: Unlimited inference, no rate limits  

**Total setup time**: 30 seconds  
**Configuration**: 2 environment variables  
**Impact**: Eliminates 429 errors completely  

---

**You're all set!** Just check that box and enjoy rate-limit-free inference! ✨
