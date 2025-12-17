# Bug: inference endpoint crashes on binary files

## what breaks
trying to generate images/audio returns 500:
```
TypeError: Object of type bytes is not JSON serializable
```

## how to reproduce
```bash
curl -X POST http://localhost:8000/inference \
  -F 'spec={"model_id":"Intel/dpt-swinv2-tiny-256","task":"depth-estimation","payload":{}}' \
  -F 'image=@test.jpg' --output depth.png
```

**expected:** get PNG back  
**actual:** server crashes

### screenshot 1: the error
<img width="996" height="154" alt="Image" src="https://github.com/user-attachments/assets/7c777fd2-c953-41b7-8bba-3cc2a78ab39e" />

### screenshot 2: pdb showing the data
<img width="2090" height="468" alt="Image" src="https://github.com/user-attachments/assets/9ed2534c-a048-4e77-84c5-c2a9609d7d6b" />

### screenshot 3: frontend doesn't display the 500 error
<img width="XXX" height="XXX" alt="frontend with no error message shown" src="PLACEHOLDER" />

**note:** the UI doesn't show any error when this happens. user just sees loading spinner forever or form silently fails with no feedback. need proper error handling in the frontend.

## why it breaks

line 91 in `app/routes/inference.py`:
```python
return JSONResponse(content=result)  # trying to JSON-encode bytes!
```

but runners return this:
```python
{"file_data": b'\x89PNG...', "file_name": "depth.png", "content_type": "image/png"}
```

json.dumps() can't handle bytes so it crashes.

## how to fix

**option 1:** detect binary and stream it (recommended)
```python
# check if we got a file
if isinstance(result, dict) and "file_data" in result:
    import io
    from fastapi.responses import StreamingResponse
    return StreamingResponse(
        io.BytesIO(result["file_data"]),
        media_type=result.get("content_type", "application/octet-stream"),
        headers={"Content-Disposition": f'inline; filename="{result.get("file_name", "output.bin")}"'}
    )

# otherwise return json
return JSONResponse(
    content=result if isinstance(result, dict) else {"result": result}
)
```

**option 2:** base64 encode it
```python
import base64

if isinstance(result, dict) and "file_data" in result:
    result["file_data"] = base64.b64encode(result["file_data"]).decode()
    
return JSONResponse(content=result)
```

i'd go with option 1 since it's cleaner and doesn't bloat the response.

## broken tasks
- depth-estimation
- text-to-image  
- text-to-speech
- image-to-image
- text-to-audio

---
branch: `mo/add-run-forms`
