"""Run without ComfyUI: python -m unittest discover -s tests."""
import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


class HashingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.models = self.root / "models"
        (self.models / "checkpoints").mkdir(parents=True)
        folders = types.SimpleNamespace(
            models_dir=str(self.models),
            folder_names_and_paths={"checkpoints": []},
            get_filename_list=lambda kind: [p.name for p in (self.models / kind).glob("*.safetensors")],
            get_full_path=lambda kind, name: str(self.models / kind / name),
        )
        spec = importlib.util.spec_from_file_location("hashing_under_test", Path(__file__).parents[1] / "hashing.py")
        self.h = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"folder_paths": folders}):
            spec.loader.exec_module(self.h)
        self.h.INDEX_DIR = str(self.root / "index")
        self.model = self.models / "checkpoints" / "test.safetensors"
        self.key = "checkpoints/test.safetensors"

    def test_download_after_initial_scan_is_discovered(self):
        self.h.build_index()
        self.assertEqual(self.h.get_snapshot()[1], [])
        self.model.write_bytes(b"completed model")
        self.h.build_index()
        self.assertEqual(self.h.get_snapshot()[1][0]["hash"], hashlib.sha256(b"completed model").hexdigest())

    def test_aria2_partial_is_skipped_then_hashed(self):
        self.model.write_bytes(b"partial")
        marker = Path(str(self.model) + ".aria2")
        marker.touch()
        self.h.build_index()
        self.assertEqual(self.h.get_snapshot()[1], [])
        self.model.write_bytes(b"completed")
        marker.unlink()
        self.h.build_index()
        self.assertEqual(len(self.h.get_snapshot()[1]), 1)

    def test_replacement_rehashes_but_unchanged_file_does_not(self):
        self.model.write_bytes(b"first")
        self.h.build_index()
        version = self.h.get_snapshot()[0]
        with patch.object(self.h, "calculate_sha256", side_effect=AssertionError("unnecessary hash")):
            self.h.build_index()
        self.assertEqual(self.h.get_snapshot()[0], version)
        self.model.write_bytes(b"replacement model")
        self.h.build_index()
        self.assertNotEqual(self.h.get_snapshot()[0], version)

    def test_legacy_unvalidated_hash_is_repaired(self):
        self.model.write_bytes(b"complete")
        self.h.save_index(self.key, {"key": self.key, "hash": "old partial hash"})
        self.h.build_index()
        self.assertEqual(self.h.get_snapshot()[1][0]["hash"], hashlib.sha256(b"complete").hexdigest())

    def test_file_changing_during_hash_is_not_published(self):
        self.model.write_bytes(b"start")
        def changed(_):
            self.model.write_bytes(b"still downloading")
            return "invalid"
        with patch.object(self.h, "calculate_sha256", side_effect=changed):
            self.h.build_index()
        self.assertEqual(self.h.get_snapshot()[1], [])
        self.assertIsNone(self.h.load_index(self.key))
        self.h.build_index()
        self.assertEqual(len(self.h.get_snapshot()[1]), 1)

    def test_watch_rescans_after_download(self):
        def wait(_):
            if not self.model.exists():
                self.model.write_bytes(b"late download")
            else:
                raise KeyboardInterrupt
        with patch.object(self.h.time, "sleep", side_effect=wait):
            with self.assertRaises(KeyboardInterrupt):
                self.h.watch_models()
        self.assertEqual(len(self.h.get_snapshot()[1]), 1)


if __name__ == "__main__":
    unittest.main()
