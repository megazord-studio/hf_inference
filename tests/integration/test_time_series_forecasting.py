def test_time_series_forecasting_basic(client):
    model_id = "ibm-granite/granite-timeseries-ttm-r1"  # pipeline tag resolves to our runner
    series = [1.0, 1.5, 2.0, 2.5]
    payload = {
        "model_id": model_id,
        "intent_id": None,
        "input_type": "time-series",
        "inputs": {"series": series},
        "task": "time-series-forecasting",
        "options": {"horizon": 3, "method": "persistence"},
    }
    resp = client.post("/api/inference", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()["result"]
    assert data.get("task") == "time-series-forecasting"
    out = data.get("task_output", {})
    forecast = out.get("forecast")
    assert isinstance(forecast, list) and len(forecast) == 3
    assert all(isinstance(x, (int, float)) for x in forecast)
    # Persistence: last value repeated
    assert forecast == [series[-1]] * 3
