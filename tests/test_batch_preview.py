"""batch_preview: a batch of N image latents becomes one N-wide strip; single / video latents pass through."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest


class FakeImage:
    def __init__(self, size, tag=None):
        self.size = size
        self.tag = tag
        self.pasted = []

    def paste(self, img, at):
        self.pasted.append((img.tag, at))

    def resize(self, size):
        return FakeImage(size, self.tag)


class FakeLatent:
    def __init__(self, n, ndim=4, start=0):
        self.shape = (n,)
        self.ndim = ndim
        self.start = start

    def __getitem__(self, s):
        return FakeLatent(s.stop - s.start, self.ndim, self.start + s.start)


def load():
    pil = types.ModuleType("PIL")
    pil.Image = types.SimpleNamespace(new=lambda mode, size: FakeImage(size, "strip"))
    sys.modules["PIL"] = pil

    lp = types.ModuleType("latent_preview")
    lp.MAX_PREVIEW_RESOLUTION = 512

    class LatentPreviewer:
        def decode_latent_to_preview_image(self, preview_format, x0):
            return ("JPEG", self.decode_latent_to_preview(x0), lp.MAX_PREVIEW_RESOLUTION)

    class Fake(LatentPreviewer):
        def decode_latent_to_preview(self, x0):
            return FakeImage((64, 96), x0.start)

    lp.LatentPreviewer = LatentPreviewer
    lp.Fake = Fake
    sys.modules["latent_preview"] = lp
    spec = importlib.util.spec_from_file_location("batch_preview", Path(__file__).resolve().parent.parent / "batch_preview.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, lp


class BatchPreviewTest(unittest.TestCase):
    def test_strip_and_passthrough(self):
        mod, lp = load()
        self.assertTrue(mod.install())
        self.assertTrue(mod.install())  # idempotent
        fmt, img, cap = lp.Fake().decode_latent_to_preview_image("JPEG", FakeLatent(3))
        self.assertEqual(img.size, (192, 96))
        self.assertEqual(img.pasted, [(0, (0, 0)), (1, (64, 0)), (2, (128, 0))])
        self.assertEqual(cap, 512 * 3)
        _, single, cap1 = lp.Fake().decode_latent_to_preview_image("JPEG", FakeLatent(1))
        self.assertEqual(single.size, (64, 96))
        self.assertEqual(cap1, 512)
        _, video, _ = lp.Fake().decode_latent_to_preview_image("JPEG", FakeLatent(2, ndim=5))
        self.assertEqual(video.size, (64, 96))


if __name__ == "__main__":
    unittest.main()
