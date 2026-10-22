"""Run without ComfyUI: python -m unittest discover -s tests."""
import hashlib
import importlib.util
import json
import shutil
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


class _Paths(dict):
    """folder_names_and_paths as ComfyUI has it: name -> (paths, extensions), live over `dirs`."""
    def __init__(self, dirs):
        super().__init__()
        self.dirs = dirs

    def __contains__(self, name):
        return name in self.dirs

    def __iter__(self):
        return iter(list(self.dirs))

    def get(self, name, default=None):
        return (self.dirs[name], {".safetensors"}) if name in self.dirs else default


OLD_ROOT = "C/Users/donny/Documents/Github/ComfyUI/models"


class HashingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.models = self.root / "models"
        (self.models / "checkpoints").mkdir(parents=True)
        # A fake ComfyUI folder_paths: each type -> ([dirs], extensions), files listed recursively.
        self.dirs = {"checkpoints": [str(self.models / "checkpoints")]}

        def names(kind):
            out = []
            for d in self.dirs.get(kind, []):
                out += [str(p.relative_to(d)).replace(os.sep, "/") for p in Path(d).rglob("*.safetensors")]
            return sorted(set(out))

        def full(kind, name):
            for d in self.dirs.get(kind, []):
                if (Path(d) / name).is_file():
                    return os.path.join(d, *name.split("/"))
            return None

        folders = types.SimpleNamespace(
            models_dir=str(self.models),
            folder_names_and_paths={},
            get_filename_list=names,
            get_full_path=full,
        )
        folders.folder_names_and_paths = _Paths(self.dirs)
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


    # -- startup preload, key migration, pending list --------------------------------------

    def _sidecar(self, key, path, digest="cafe"):
        stat = os.stat(path)
        self.h.save_index(key, {"key": key, "filename": os.path.basename(path), "hash": digest,
                                "hashed_at": 1.0, "file": {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}})

    def test_preload_serves_every_sidecar_before_the_scan(self):
        self.model.write_bytes(b"model")
        self._sidecar(self.key, self.model, "aa")
        self._sidecar(OLD_ROOT + "/loras/gone.safetensors", self.model, "bb")
        with patch.object(self.h, "calculate_sha256", side_effect=AssertionError("no hashing")), \
             patch.object(self.h, "_file_signature", side_effect=AssertionError("no model file touched")), \
             patch.object(self.h.folder_paths, "get_filename_list", side_effect=AssertionError("no listing")):
            self.assertEqual(self.h.preload(), 2)
        version, models, pending, progress = self.h.get_listing()
        self.assertEqual({m["key"]: m["hash"] for m in models},
                         {self.key: "aa", OLD_ROOT + "/loras/gone.safetensors": "bb"})
        self.assertFalse(progress["ready"])
        self.assertNotEqual(version, "empty")

    def test_scan_drops_keys_of_files_that_are_gone_but_keeps_their_sidecars(self):
        self.model.write_bytes(b"model")
        self._sidecar(self.key, self.model, "aa")
        self._sidecar("checkpoints/gone.safetensors", self.model, "bb")
        self.h.preload()
        with patch.object(self.h, "calculate_sha256", side_effect=AssertionError("no hashing")):
            self.h.build_index()
        _, models, pending, progress = self.h.get_listing()
        self.assertEqual([m["key"] for m in models], [self.key])
        self.assertEqual(progress, {"hashed": 1, "total": 1, "pending": 0, "ready": True})
        self.assertIsNotNone(self.h.load_index("checkpoints/gone.safetensors"))

    def test_old_extra_path_key_is_carried_over_without_hashing(self):
        """extra_model_paths.yaml -> symlink: every key changed; same name + size + mtime_ns reuse the hash."""
        self.model.write_bytes(b"model")
        old_key = OLD_ROOT + "/checkpoints/test.safetensors"
        self._sidecar(old_key, self.model, "oldhash")
        self.h.preload()
        with patch.object(self.h, "calculate_sha256", side_effect=AssertionError("no hashing")):
            self.h.build_index()
        _, models, pending, _ = self.h.get_listing()
        self.assertEqual([(m["key"], m["hash"]) for m in models], [(self.key, "oldhash")])
        self.assertEqual(pending, [])
        self.assertEqual(self.h.load_index(self.key)["key"], self.key)
        self.assertIsNotNone(self.h.load_index(old_key), "old sidecars are never deleted")
        # Idempotent: the next scan reads the new sidecar and changes nothing.
        version = self.h.get_snapshot()[0]
        with patch.object(self.h, "calculate_sha256", side_effect=AssertionError("no hashing")):
            self.h.build_index()
        self.assertEqual(self.h.get_snapshot()[0], version)

    def test_new_sidecar_from_the_migration_script_wins(self):
        self.model.write_bytes(b"model")
        self._sidecar(OLD_ROOT + "/checkpoints/test.safetensors", self.model, "old")
        self._sidecar(self.key, self.model, "new")
        with patch.object(self.h, "calculate_sha256", side_effect=AssertionError("no hashing")):
            self.h.build_index()
        self.assertEqual([m["hash"] for m in self.h.get_snapshot()[1]], ["new"])

    def test_sidecar_copied_with_its_old_key_field_is_filed_under_its_path(self):
        self.model.write_bytes(b"model")
        self._sidecar(self.key, self.model, "aa")
        path = Path(self.h.INDEX_DIR) / "checkpoints" / "test.safetensors.index"
        data = json.loads(path.read_text())
        data["key"] = OLD_ROOT + "/checkpoints/test.safetensors"
        path.write_text(json.dumps(data))
        self.h.preload()
        self.assertEqual([m["key"] for m in self.h.get_snapshot()[1]], [self.key])

    def test_moved_file_keeps_its_hash(self):
        self.model.write_bytes(b"model")
        self.h.build_index()
        digest = self.h.get_snapshot()[1][0]["hash"]
        (self.models / "checkpoints" / "SDXL").mkdir()
        moved = self.models / "checkpoints" / "SDXL" / "test.safetensors"
        os.replace(self.model, moved)
        with patch.object(self.h, "calculate_sha256", side_effect=AssertionError("no hashing")):
            self.h.build_index()
        self.assertEqual([(m["key"], m["hash"]) for m in self.h.get_snapshot()[1]],
                         [("checkpoints/SDXL/test.safetensors", digest)])

    def test_same_name_different_size_is_hashed_again(self):
        self.model.write_bytes(b"model")
        self._sidecar(OLD_ROOT + "/checkpoints/test.safetensors", self.model, "old")
        self.model.write_bytes(b"another model")
        self.h.build_index()
        self.assertEqual(self.h.get_snapshot()[1][0]["hash"], hashlib.sha256(b"another model").hexdigest())

    def test_unhashed_models_are_pending_and_small_ones_go_first(self):
        big = self.models / "checkpoints" / "big.safetensors"
        big.write_bytes(b"x" * 1000)
        self.model.write_bytes(b"small")
        order, seen = [], []
        real = self.h.calculate_sha256

        def spy(path):
            order.append(os.path.basename(path))
            seen.append({p["filename"]: p["state"] for p in self.h.get_listing()[2]})
            return real(path)

        with patch.object(self.h, "calculate_sha256", side_effect=spy):
            self.h.build_index()
        self.assertEqual(order, ["test.safetensors", "big.safetensors"])
        self.assertEqual(seen[0], {"test.safetensors": "hashing", "big.safetensors": "queued"})
        self.assertEqual(seen[1], {"big.safetensors": "hashing"})
        _, models, pending, progress = self.h.get_listing()
        self.assertEqual((len(models), pending, progress["hashed"], progress["total"]), (2, [], 2, 2))
        self.assertIsNone(self.h.get_status()["current"])

    def test_pending_changes_the_version(self):
        self.model.write_bytes(b"model")
        versions = []
        with patch.object(self.h, "calculate_sha256", side_effect=lambda p: versions.append(self.h.get_snapshot()[0]) or "x"):
            self.h.build_index()
        self.assertNotEqual(versions[0], "empty")
        self.assertNotEqual(versions[0], self.h.get_snapshot()[0])

    def test_failed_hash_stays_listed_as_error(self):
        self.model.write_bytes(b"model")
        with patch.object(self.h, "calculate_sha256", side_effect=OSError("share gone")):
            self.h.build_index()
        pending = self.h.get_listing()[2]
        self.assertEqual((pending[0]["key"], pending[0]["state"]), (self.key, "error"))

    def _beside(self, path):
        return Path(path).parent / ".hash-index" / (Path(path).name + ".index")

    def test_hash_is_written_beside_the_model(self):
        (self.models / "checkpoints" / "SDXL").mkdir()
        model = self.models / "checkpoints" / "SDXL" / "a.safetensors"
        model.write_bytes(b"model")
        self.h.build_index()
        data = json.loads(self._beside(model).read_text())
        self.assertEqual((data["key"], data["hash"]),
                         ("checkpoints/SDXL/a.safetensors", hashlib.sha256(b"model").hexdigest()))

    def test_another_machines_hash_beside_the_model_is_used(self):
        """A fresh machine on the same share: no local index, the sidecar beside the model is enough."""
        self.model.write_bytes(b"model")
        self.h.build_index()
        shutil.rmtree(self.h.INDEX_DIR)
        self.h._BESIDE_SEEN.clear()
        self.h._INDEX.clear()
        with patch.object(self.h, "calculate_sha256", side_effect=AssertionError("no hashing")):
            self.h.build_index()
        self.assertEqual(self.h.get_snapshot()[1][0]["hash"], hashlib.sha256(b"model").hexdigest())
        self.assertIsNotNone(self.h.load_index(self.key), "a local copy is kept")

    def test_existing_local_hash_is_shared_beside_the_model(self):
        self.model.write_bytes(b"model")
        self._sidecar(self.key, self.model, "aa")
        with patch.object(self.h, "calculate_sha256", side_effect=AssertionError("no hashing")):
            self.h.build_index()
        self.assertEqual(json.loads(self._beside(self.model).read_text())["hash"], "aa")

    def test_mtime_rounded_by_windows_still_matches(self):
        self.model.write_bytes(b"model")
        stat = os.stat(self.model)
        rounded = stat.st_mtime_ns - stat.st_mtime_ns % 100 - 100  # 100 ns ticks, a tick off at most
        self.h.save_index(self.key, {"key": self.key, "hash": "win", "file": {"size": stat.st_size, "mtime_ns": rounded}})
        with patch.object(self.h, "calculate_sha256", side_effect=AssertionError("no hashing")):
            self.h.build_index()
        self.assertEqual(self.h.get_snapshot()[1][0]["hash"], "win")

    def test_replaced_file_drops_the_stale_hash_beside_it(self):
        self.model.write_bytes(b"first")
        self.h.build_index()
        self.model.write_bytes(b"a different model")
        os.utime(self.model, ns=(1, 1))
        self.h.build_index()
        self.assertEqual(json.loads(self._beside(self.model).read_text())["hash"],
                         hashlib.sha256(b"a different model").hexdigest())

    def test_forget_removes_the_hash_beside_the_model(self):
        self.model.write_bytes(b"model")
        self.h.build_index()
        self.h.forget(str(self.model))
        self.assertFalse(self._beside(self.model).exists())

    def test_read_only_share_still_hashes(self):
        self.model.write_bytes(b"model")
        real = self.h._write_json
        def refuse(path, data):
            if ".hash-index" in path:
                raise PermissionError("read-only")
            real(path, data)
        with patch.object(self.h, "_write_json", side_effect=refuse):
            self.h.build_index()
            self.h.build_index()
        self.assertEqual(self.h.get_snapshot()[1][0]["hash"], hashlib.sha256(b"model").hexdigest())
        self.assertIsNotNone(self.h.load_index(self.key))

    def test_extra_path_and_resolved_symlink_get_type_relative_keys(self):
        extra = self.root / "share" / "ckpts"
        (extra / "SDXL").mkdir(parents=True)
        (extra / "SDXL" / "a.safetensors").write_bytes(b"a")
        self.dirs["checkpoints"].append(str(extra))
        self.assertEqual(self.h.cache_key(str(extra / "SDXL" / "a.safetensors")), "checkpoints/SDXL/a.safetensors")
        # models/loras is a symlink to the share; a path that comes back resolved keys the same.
        target = self.root / "share" / "loras"
        target.mkdir()
        (target / "l.safetensors").write_bytes(b"l")
        try:
            (self.models / "loras").symlink_to(target, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("no symlinks here")
        self.dirs["loras"] = [str(self.models / "loras")]
        self.assertEqual(self.h.cache_key(str(self.models / "loras" / "l.safetensors")), "loras/l.safetensors")
        self.assertEqual(self.h.cache_key(str(target / "l.safetensors")), "loras/l.safetensors")
        self.h.build_index()
        self.assertEqual(sorted(m["key"] for m in self.h.get_snapshot()[1]),
                         ["checkpoints/SDXL/a.safetensors", "loras/l.safetensors"])

    EXTRA_FOLDERS = ["text_encoders", "ipadapter", "instantid", "pulid", "style_models", "background_removal"]

    def test_extra_folders_are_scanned(self):
        for name in ["clip"] + self.EXTRA_FOLDERS:
            self.assertIn(name, self.h.MODEL_TYPES)

    def test_every_extra_folder_is_hashed_with_its_sidecar_beside_it(self):
        expected = []
        for name in self.EXTRA_FOLDERS:
            folder = self.models / name
            folder.mkdir()
            (folder / f"{name}.safetensors").write_bytes(name.encode())
            self.dirs[name] = [str(folder)]
            expected.append(f"{name}/{name}.safetensors")
        self.h.build_index()
        snapshot = {m["key"]: m["hash"] for m in self.h.get_snapshot()[1]}
        self.assertEqual(sorted(snapshot), sorted(expected))
        for name in self.EXTRA_FOLDERS:
            model = self.models / name / f"{name}.safetensors"
            data = json.loads(self._beside(model).read_text())
            self.assertEqual((data["key"], data["hash"]),
                             (f"{name}/{name}.safetensors", hashlib.sha256(name.encode()).hexdigest()))

    def test_old_clip_name_still_scanned_when_text_encoders_is_unknown(self):
        folder = self.models / "clip"
        folder.mkdir()
        (folder / "t5.safetensors").write_bytes(b"t5")
        self.dirs["clip"] = [str(folder)]
        self.h.build_index()
        self.assertEqual([m["key"] for m in self.h.get_snapshot()[1]], ["clip/t5.safetensors"])

    def test_directories_and_sidecars_listed_by_an_unfiltered_folder_are_skipped(self):
        """A pack that registers no extensions gets every entry listed: a model directory, our own sidecars."""
        folder = self.models / "instantid"
        (folder / "antelopev2").mkdir(parents=True)
        (folder / "ip-adapter.bin").write_bytes(b"instantid")
        self.dirs["instantid"] = [str(folder)]
        listed = ["antelopev2", "ip-adapter.bin", ".hash-index/ip-adapter.bin.index"]
        real = self.h.folder_paths.get_filename_list
        real_full = self.h.folder_paths.get_full_path

        def names(kind):
            return listed if kind == "instantid" else real(kind)

        def full(kind, name):
            if kind == "instantid":
                path = folder / name
                return str(path) if path.exists() else None
            return real_full(kind, name)

        with patch.object(self.h.folder_paths, "get_filename_list", names), \
                patch.object(self.h.folder_paths, "get_full_path", full):
            self.h.build_index()
            self.assertTrue(self._beside(folder / "ip-adapter.bin").is_file())
            self.h.build_index()  # now the sidecar exists and is listed too
        keys = [m["key"] for m in self.h.get_snapshot()[1]]
        self.assertEqual(keys, ["instantid/ip-adapter.bin"])
        self.assertEqual(self.h.get_listing()[2], [])


if __name__ == "__main__":
    unittest.main()
