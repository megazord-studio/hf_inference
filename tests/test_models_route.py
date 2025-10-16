import requests


def _http_error(status: int, message: str = "") -> requests.HTTPError:
    response = requests.Response()
    response.status_code = status
    reason = message or f"{status} Error"
    return requests.HTTPError(reason, response=response)


def test_models_returns_stale_cache_on_http_error(client, monkeypatch):
    task = "text-generation"
    stale_data = [
        {"id": "dummy/model", "likes": 42, "trendingScore": 0, "downloads": 100, "gated": "false"}
    ]

    def fake_get_cached_min(requested_task: str, *, allow_stale: bool = False):
        assert requested_task == task
        if allow_stale:
            return stale_data
        return None

    def fake_set_cached_min(requested_task, data):  # pragma: no cover - no-op
        return None

    def fake_fetch_all_by_task(*args, **kwargs):
        raise _http_error(429, "429 Client Error: Too Many Requests")

    monkeypatch.setattr("app.routes.hf_models.get_cached_min", fake_get_cached_min)
    monkeypatch.setattr("app.routes.hf_models.set_cached_min", fake_set_cached_min)
    monkeypatch.setattr("app.routes.hf_models.fetch_all_by_task", fake_fetch_all_by_task)

    response = client.get(f"/models?task={task}")

    assert response.status_code == 200
    assert response.headers.get("X-HF-Cache") == "stale"
    assert response.json() == stale_data


def test_models_returns_hint_on_rate_limit_without_cache(client, monkeypatch):
    task = "summarization"

    def fake_get_cached_min(requested_task: str, *, allow_stale: bool = False):
        assert requested_task == task
        return None

    def fake_set_cached_min(requested_task, data):  # pragma: no cover - no-op
        return None

    def fake_fetch_all_by_task(*args, **kwargs):
        raise _http_error(429, "429 Client Error: Too Many Requests")

    monkeypatch.setattr("app.routes.hf_models.get_cached_min", fake_get_cached_min)
    monkeypatch.setattr("app.routes.hf_models.set_cached_min", fake_set_cached_min)
    monkeypatch.setattr("app.routes.hf_models.fetch_all_by_task", fake_fetch_all_by_task)

    response = client.get(f"/models?task={task}")

    assert response.status_code == 429
    payload = response.json()
    assert payload.get("error") == "hf_api_failed"
    assert "Too Many Requests" in payload.get("reason", "")
    assert "hint" in payload
