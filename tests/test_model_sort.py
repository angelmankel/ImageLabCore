"""Run without ComfyUI: python -m unittest discover -s tests.

The download test needs aiohttp (ComfyUI has it) and is skipped without it.
"""
import asyncio
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).parents[1]


def safetensors(path: Path, keys, payload: bytes = b"") -> None:
    header = json.dumps({"__metadata__": {}, **{k: {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]} for k in keys}}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + payload)


def load_package(folders):
    """Load hashing, model_sort (and downloads when aiohttp exists) as a package, with a fake
    ComfyUI `folder_paths`."""
    pkg = types.ModuleType("ilc")
    pkg.__path__ = [str(ROOT)]
    mods = {"folder_paths": folders, "ilc": pkg}
    with patch.dict(sys.modules, mods):
        loaded = {}
        for name in ("hashing", "model_sort") + (("downloads",) if importlib.util.find_spec("aiohttp") else ()):
            spec = importlib.util.spec_from_file_location(f"ilc.{name}", ROOT / f"{name}.py")
            mod = importlib.util.module_from_spec(spec)
            sys.modules[f"ilc.{name}"] = mod
            spec.loader.exec_module(mod)
            setattr(pkg, name, mod)
            loaded[name] = mod
        return loaded


class ModelSortTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.models = Path(self.temp.name) / "models"
        for d in ("checkpoints", "diffusion_models", "text_encoders"):
            (self.models / d).mkdir(parents=True)
        folders = types.SimpleNamespace(
            models_dir=str(self.models),
            folder_names_and_paths={d: ([str(self.models / d)], set()) for d in ("checkpoints", "diffusion_models", "text_encoders")},
            get_filename_list=lambda kind: sorted(p.name for p in (self.models / kind).glob("*.safetensors")),
            get_full_path=lambda kind, name: str(self.models / kind / name),
        )
        self.m = load_package(folders)
        self.m["hashing"].INDEX_DIR = str(Path(self.temp.name) / "index")

    def test_classifies_by_header(self):
        ms = self.m["model_sort"]
        sdxl = self.models / "checkpoints" / "sdxl.safetensors"
        safetensors(sdxl, ["model.diffusion_model.x", "first_stage_model.y", "conditioner.z"])
        aio = self.models / "checkpoints" / "aio.safetensors"
        safetensors(aio, ["model.diffusion_model.x", "vae.y", "text_encoders.z"])
        wan = self.models / "checkpoints" / "wan.safetensors"
        safetensors(wan, ["blocks.0.self_attn.q.weight", "head.head.weight"])
        self.assertFalse(ms.is_diffusion_only(ms.safetensors_keys(str(sdxl))))
        self.assertFalse(ms.is_diffusion_only(ms.safetensors_keys(str(aio))))
        self.assertTrue(ms.is_diffusion_only(ms.safetensors_keys(str(wan))))
        # A diffusion model with its VAE bundled still has no text encoder: not a checkpoint.
        wan_vae = self.models / "checkpoints" / "wan_vae.safetensors"
        safetensors(wan_vae, ["model.diffusion_model.blocks.0.w", "vae.decoder.w"])
        self.assertTrue(ms.is_diffusion_only(ms.safetensors_keys(str(wan_vae))))
        info = ms.describe(str(wan_vae))
        self.assertEqual((info["has_vae"], info["has_text_encoder"], info["tensors"]), (True, False, 2))
        self.assertIsNone(ms.safetensors_keys(str(self.models / "checkpoints" / "x.ckpt")))

    def test_sort_moves_only_diffusion_models_and_keeps_the_hash(self):
        ms, h = self.m["model_sort"], self.m["hashing"]
        safetensors(self.models / "checkpoints" / "sdxl.safetensors", ["first_stage_model.a", "conditioner.b"])
        wan = self.models / "checkpoints" / "wan.safetensors"
        safetensors(wan, ["blocks.0.w"], b"weights")
        digest = h.hash_model(str(wan))["hash"]
        with patch.object(h, "calculate_sha256", side_effect=AssertionError("moved file was rehashed")):
            result = ms.sort_checkpoints()
        self.assertEqual(result, {"moved": ["wan.safetensors"]})
        self.assertTrue((self.models / "diffusion_models" / "wan.safetensors").exists())
        self.assertTrue((self.models / "checkpoints" / "sdxl.safetensors").exists())
        entries = {e["key"]: e for e in h.get_snapshot()[1]}
        self.assertEqual(entries["diffusion_models/wan.safetensors"]["hash"], digest)
        self.assertNotIn("checkpoints/wan.safetensors", entries)

    def test_never_overwrites(self):
        ms = self.m["model_sort"]
        safetensors(self.models / "checkpoints" / "wan.safetensors", ["blocks.0.w"])
        (self.models / "diffusion_models" / "wan.safetensors").write_bytes(b"other")
        self.assertEqual(ms.sort_checkpoints(), {"moved": []})
        self.assertEqual((self.models / "diffusion_models" / "wan.safetensors").read_bytes(), b"other")

    def test_gguf_moves_under_its_own_name(self):
        ms = self.m["model_sort"]
        # An older downloader named CivitAI's x.gguf "x.gguf.safetensors"; checkpoints/ does not
        # even list a real .gguf. Both are diffusion models for UnetLoaderGGUF.
        (self.models / "checkpoints" / "wanQ8H.gguf.safetensors").write_bytes(b"GGUF\x03\x00\x00\x00rest")
        (self.models / "checkpoints" / "wanQ8L.gguf").write_bytes(b"GGUF\x03\x00\x00\x00rest")
        result = ms.sort_checkpoints()
        self.assertEqual(sorted(result["moved"]), ["wanQ8H.gguf.safetensors", "wanQ8L.gguf"])
        self.assertEqual(sorted(p.name for p in (self.models / "diffusion_models").iterdir()), ["wanQ8H.gguf", "wanQ8L.gguf"])
        self.assertEqual(list((self.models / "checkpoints").iterdir()), [])
        self.assertEqual(ms.gguf_name("a.safetensors"), "a.gguf")

    @unittest.skipUnless(importlib.util.find_spec("aiohttp"), "aiohttp not installed")
    def test_download_in_parts_and_in_one_stream(self):
        from aiohttp import web
        d = self.m["downloads"]
        data = os.urandom(3 * 1024 * 1024 + 17)
        src = Path(self.temp.name) / "blob.bin"
        src.write_bytes(data)
        seen = []

        async def ranged(request):
            seen.append(request.headers.get("Range"))
            return web.FileResponse(src)          # FileResponse serves Range requests

        async def plain(request):
            seen.append(request.headers.get("Range"))
            return web.Response(body=data)        # ignores Range: a 200 with the whole body

        async def go():
            app = web.Application()
            app.router.add_get("/ranged/blob.safetensors", ranged)
            app.router.add_get("/plain/blob.safetensors", plain)
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            port = site._server.sockets[0].getsockname()[1]
            try:
                d._SPLIT_MIN = 1024 * 1024
                for kind in ("ranged", "plain"):
                    target = self.models / "text_encoders" / f"{kind}.safetensors"
                    progress = d.DownloadProgress(version_id=d.url_key(kind), model_id=0, folder="text_encoders",
                                                  filename=target.name, downloaded_bytes=0, total_bytes=0, status="downloading")
                    ok = await d._stream(f"http://127.0.0.1:{port}/{kind}/blob.safetensors", {}, str(target), progress)
                    self.assertTrue(ok)
                    self.assertEqual(hashlib.sha256(target.read_bytes()).digest(), hashlib.sha256(data).digest(), kind)
                    self.assertEqual(progress.total_bytes, len(data))
                    self.assertFalse(Path(str(target) + ".downloading").exists())
            finally:
                await runner.cleanup()

        asyncio.run(go())
        ranged_parts = [r for r in seen if r and r != "bytes=0-0"]
        self.assertEqual(len(ranged_parts), 8 + 0, seen)   # the ranged server was split in 8 parts


if __name__ == "__main__":
    unittest.main()
