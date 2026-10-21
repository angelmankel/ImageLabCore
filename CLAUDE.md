# ImageLabCore — the ComfyUI custom node

**Every change Donny asks for ships at once, without asking:** push it to the running pod, then
commit + push here, then move the pin in ImageLabDocker. Full rule: `../ImageLabDocker/CLAUDE.md`,
rule 6.

Adds an HTTP API to ComfyUI and no nodes at all. See `README.md` for the route table.

**Changing it needs no image rebuild.** It lives on the pod's volume, symlinked into `custom_nodes`
by `ImageLabDocker/start.sh`:

```sh
cd ../ImageLabDocker && scripts/push-node.sh <ip:port>     # ~15 seconds
```

That pushes every `.py`, then POSTs `/manager/reboot`, which ends in `os.execv` — ComfyUI re-execs
in place, so the container does not restart and the pod's port does not change. Because it is a
real restart, **adding a route works the same as changing one**; aiohttp's frozen router stops
mattering.

Check what is actually running: `GET /imagelab/api/version` reports the directory it loaded from and
the mtime of `api.py`.

## Things to know

- **Only `hashing`, `api`, `downloads`, `favorites`, `model_sort`, `preview` and `batch_preview` are imported.** `batch_preview` patches ComfyUI's previewer so a batch's live preview is all N pictures side by side in one frame (the web app cuts it into tiles; `IMAGELAB_BATCH_PREVIEW=0` turns it off). `routes.py`, `workflows.py`,
  `imagelab_nodes.py`, `resources.py`, `civitai_routes.py`, `parameters.py` and
  `schema_generator.py` are reference only and deliberately not
  loaded. Editing one of those and seeing nothing change is an easy hour to lose.
- **The CivitAI token is `CIVITAI_API_KEY`**, the same name the image and `.env` use. One name
  for one secret — do not add an alias.
- **No third-party dependencies.** It imports `aiohttp` (ComfyUI has it) plus `folder_paths` and
  `server`, which are ComfyUI itself. There is no `requirements.txt` and adding one means the image
  build starts installing things.
- **Every list endpoint is ETag'd.** Clients poll; 304 keeps that nearly free. Keep it that way.
- **CivitAI calls every base model a "Checkpoint".** Wan / Flux / Z-Image / Qwen / Anima files are
  usually the diffusion model alone and only `UNETLoader` (diffusion_models/) can load them.
  `model_sort` reads the safetensors header: no text-encoder weights → moved to diffusion_models/
  (after each download, at startup, `POST /models/sort`), hash carried over (`hashing.relocate`).
- **Downloads use up to 8 Range connections** (`downloads._stream`), each part retried. One
  connection from CivitAI ran ~1 MB/s; split, a 16 GB file took about two minutes on a pod.
  `{url, folder}` downloads any https file (Hugging Face); `HF_TOKEN` is sent to huggingface.co.
- **A pod may run two ComfyUIs on one `models/` and one copy of this node** (the second is for
  video, at `/video-comfy/`). The second gets `IMAGELAB_CORE_SECONDARY=1`: it scans nothing,
  sorts nothing, downloads nothing, and serves only favorites + version (favorites read *its*
  temp/). nginx sends its other `/imagelab/api/*` requests to the first. `/imagelab/api/version`
  says `"secondary": true|false`.
- **Hashes travel with the models.** Every sidecar is written twice: `model_index/` (this machine's
  copy, read by `preload()`) and `<model folder>/.hash-index/<file>.index` beside the model, so a
  machine mounting the same share (gpu-server, winold, a pod) reuses the others' hashes. A scan
  takes the local copy, then the one beside, and fills in the missing one; a read-only share only
  logs once. Mtimes within 1 µs match (Windows/SMB keep 100 ns ticks).
- **Hashing never hides a known model.** `hashing.preload()` serves every `model_index/` sidecar
  before sorting or scanning; a scan then reconciles files with sidecars (a key that changed takes
  over the sidecar with the same name + size + mtime_ns, copied to the new key, old one kept) and
  hashes the rest smallest first. `/hashes` lists those as `pending`, with `progress`. Keys are
  relative to the models dir, or `<type>/…` for an extra path, never the resolved symlink target.
- Tests: `python -m unittest discover -s tests` (the download test needs aiohttp).

**The project rules — pods, money, pulling images before a terminate — live in
`../ImageLabDocker/CLAUDE.md`. Read that first.**
