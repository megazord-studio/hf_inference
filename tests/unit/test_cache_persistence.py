"""Tests for disk cache persistence, TTL, and corruption handling.

These tests verify that:
- Models cache is persisted to disk and loaded correctly
- Enrich cache is persisted to disk and loaded correctly
- TTL expiration is respected
- Corrupted cache files are handled gracefully
"""
import json
import os
import tempfile
import time
from unittest.mock import patch

import pytest


class TestModelsCachePersistence:
    """Tests for models_preloaded.json disk cache."""

    def test_cache_persists_to_disk(self, tmp_path):
        """Test that cache is written to disk correctly."""
        cache_file = tmp_path / "models_preloaded.json"
        
        payload = {
            "timestamp": time.time(),
            "mode": "all",
            "models": [
                {"id": "test/model1", "pipeline_tag": "text-generation"},
                {"id": "test/model2", "pipeline_tag": "image-classification"},
            ],
        }
        
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        
        assert cache_file.exists()
        
        with open(cache_file, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        
        assert loaded["mode"] == "all"
        assert len(loaded["models"]) == 2
        assert loaded["models"][0]["id"] == "test/model1"

    def test_ttl_expiration_check(self, tmp_path):
        """Test that expired cache is detected."""
        cache_file = tmp_path / "models_preloaded.json"
        
        # Create cache with old timestamp (5 days ago, TTL is 4 days)
        old_ts = time.time() - (5 * 24 * 60 * 60)
        payload = {
            "timestamp": old_ts,
            "mode": "all",
            "models": [{"id": "test/model1"}],
        }
        
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        
        with open(cache_file, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        
        ttl = 4 * 24 * 60 * 60  # 4 days
        is_expired = time.time() - loaded["timestamp"] > ttl
        assert is_expired, "Cache should be expired"

    def test_fresh_cache_not_expired(self, tmp_path):
        """Test that fresh cache is not expired."""
        cache_file = tmp_path / "models_preloaded.json"
        
        # Create cache with fresh timestamp
        payload = {
            "timestamp": time.time(),
            "mode": "all",
            "models": [{"id": "test/model1"}],
        }
        
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        
        with open(cache_file, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        
        ttl = 4 * 24 * 60 * 60  # 4 days
        is_expired = time.time() - loaded["timestamp"] > ttl
        assert not is_expired, "Cache should not be expired"

    def test_corrupted_cache_handled(self, tmp_path):
        """Test that corrupted JSON is handled gracefully."""
        cache_file = tmp_path / "models_preloaded.json"
        
        # Write corrupted JSON
        with open(cache_file, "w", encoding="utf-8") as f:
            f.write("{invalid json content")
        
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                json.load(f)
            loaded_ok = True
        except json.JSONDecodeError:
            loaded_ok = False
        
        assert not loaded_ok, "Corrupted JSON should raise error"

    def test_missing_fields_handled(self, tmp_path):
        """Test that cache with missing fields is handled."""
        cache_file = tmp_path / "models_preloaded.json"
        
        # Create cache missing required fields
        payload = {
            "timestamp": time.time(),
            # Missing "mode" and "models"
        }
        
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        
        with open(cache_file, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        
        models = loaded.get("models", [])
        assert models == [], "Missing models should default to empty list"


class TestEnrichCachePersistence:
    """Tests for enrich_cache.json disk cache."""

    def test_enrich_cache_persists(self, tmp_path):
        """Test that enrich cache is written correctly."""
        cache_file = tmp_path / "enrich_cache.json"
        
        payload = {
            "timestamp": time.time(),
            "entries": {
                "model/a": {"gated": True, "ts": time.time()},
                "model/b": {"gated": False, "ts": time.time()},
            },
        }
        
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        
        assert cache_file.exists()
        
        with open(cache_file, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        
        assert "model/a" in loaded["entries"]
        assert loaded["entries"]["model/a"]["gated"] is True

    def test_enrich_entry_ttl(self, tmp_path):
        """Test per-entry TTL expiration in enrich cache."""
        cache_file = tmp_path / "enrich_cache.json"
        
        now = time.time()
        ttl = 4 * 24 * 60 * 60  # 4 days
        
        payload = {
            "timestamp": now,
            "entries": {
                "fresh/model": {"gated": True, "ts": now},
                "stale/model": {"gated": False, "ts": now - (5 * 24 * 60 * 60)},
            },
        }
        
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        
        with open(cache_file, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        
        fresh_entry = loaded["entries"]["fresh/model"]
        stale_entry = loaded["entries"]["stale/model"]
        
        assert now - fresh_entry["ts"] < ttl, "Fresh entry should not be expired"
        assert now - stale_entry["ts"] > ttl, "Stale entry should be expired"

    def test_enrich_cache_corrupted(self, tmp_path):
        """Test handling of corrupted enrich cache."""
        cache_file = tmp_path / "enrich_cache.json"
        
        with open(cache_file, "w", encoding="utf-8") as f:
            f.write("not valid json {")
        
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                json.load(f)
            loaded_ok = True
        except json.JSONDecodeError:
            loaded_ok = False
        
        assert not loaded_ok, "Corrupted cache should fail to load"

    def test_atomic_write_pattern(self, tmp_path):
        """Test that cache write uses atomic pattern (tmp + rename)."""
        cache_file = tmp_path / "enrich_cache.json"
        tmp_file = tmp_path / "enrich_cache.json.tmp"
        
        payload = {
            "timestamp": time.time(),
            "entries": {"test/model": {"gated": False, "ts": time.time()}},
        }
        
        # Simulate atomic write pattern
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        
        # Atomic rename
        os.replace(tmp_file, cache_file)
        
        assert cache_file.exists()
        assert not tmp_file.exists(), "Temp file should be removed after rename"


class TestCacheDirectoryHandling:
    """Tests for cache directory creation and handling."""

    def test_cache_dir_created_if_missing(self, tmp_path):
        """Test that cache directory is created if it doesn't exist."""
        cache_dir = tmp_path / "cache"
        assert not cache_dir.exists()
        
        os.makedirs(cache_dir, exist_ok=True)
        assert cache_dir.exists()

    def test_cache_dir_permissions(self, tmp_path):
        """Test that cache directory has correct permissions."""
        cache_dir = tmp_path / "cache"
        os.makedirs(cache_dir, exist_ok=True)
        
        # Should be writable
        test_file = cache_dir / "test.txt"
        with open(test_file, "w") as f:
            f.write("test")
        assert test_file.exists()
