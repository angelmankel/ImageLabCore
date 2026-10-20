"""
Live previews for every picture of a batch.

ComfyUI's previewers decode only the first latent of a batch (`x0[:1]` / `x0[0]` in
`latent_preview`), so with batch size 4 the browser sees one picture move and three appear at the
end. This wraps `LatentPreviewer.decode_latent_to_preview_image` (the one method every previewer
inherits) so a batch of N image latents comes out as ONE frame: the N decoded pictures side by side,
left to right, each at the preview size. The ImageLab web app (next/ `jobs/liveGridLogic.ts`)
recognises a frame N pictures wide and cuts it into N live tiles; an old app just shows the strip.

Video latents (5-D) and single pictures go through untouched. Anything that fails falls back to
ComfyUI's own preview, so a preview can never break a render. `IMAGELAB_BATCH_PREVIEW=0` turns it off.
"""
import os

MAX_TILES = 16


def _enabled() -> bool:
    return os.environ.get("IMAGELAB_BATCH_PREVIEW", "1").strip().lower() not in ("0", "false", "no", "off")


def install() -> bool:
    """Patch the previewer base class once. Returns whether the patch is in place."""
    if not _enabled():
        return False
    try:
        import latent_preview
        from PIL import Image
    except Exception as e:  # noqa: BLE001 — not inside ComfyUI
        print(f"[ImageLab] batch preview not installed: {e}")
        return False

    base = latent_preview.LatentPreviewer
    if getattr(base, "_imagelab_batch", False):
        return True
    original = base.decode_latent_to_preview_image

    def decode_batch(self, preview_format, x0):
        try:
            n = int(x0.shape[0]) if getattr(x0, "ndim", 0) == 4 else 1
        except Exception:  # noqa: BLE001
            n = 1
        if n <= 1 or n > MAX_TILES:
            return original(self, preview_format, x0)
        try:
            tiles = [self.decode_latent_to_preview(x0[i:i + 1]) for i in range(n)]
            w, h = tiles[0].size
            strip = Image.new("RGB", (w * n, h))
            for i, tile in enumerate(tiles):
                if tile.size != (w, h):
                    tile = tile.resize((w, h))
                strip.paste(tile, (i * w, 0))
            # The cap applies to the long side; give each picture the usual preview size.
            return ("JPEG", strip, int(latent_preview.MAX_PREVIEW_RESOLUTION) * n)
        except Exception as e:  # noqa: BLE001 — never break a render over a preview
            print(f"[ImageLab] batch preview fell back: {e}")
            return original(self, preview_format, x0)

    base.decode_latent_to_preview_image = decode_batch
    base._imagelab_batch = True
    return True
