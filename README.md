# ImageLabCore

The ComfyUI custom node ImageLab talks to. It adds an HTTP API to ComfyUI and no nodes at all —
everything here exists so the front end can ask a ComfyUI server three things it cannot answer on
its own:

- **What models are installed, and what are they?** `/imagelab/api/hashes` returns a SHA-256 index of
  every model file, scanned in the background at startup and every 30 seconds after each scan.
  Incomplete aria2 downloads are skipped; changed files are rehashed. Hashes make a local file
  identifiable on CivitAI, which is where the previews, trigger words and base-model badges come
  from.
- **Fetch this model.** `/imagelab/api/downloads` starts a CivitAI download, reports progress, and
  cancels. `/imagelab/api/models/…` deletes one again.
- **Keep this picture.** `/imagelab/api/favorites` copies a generation out of ComfyUI's `temp/`, which
  is wiped on restart, into a dated tree that survives.

Every list endpoint is ETag'd, so a client that polls costs almost nothing.

## Routes

| Method | Path | |
|---|---|---|
| GET | `/imagelab/api/hashes` | model hash index; send `If-None-Match` to get a 304 |
| GET | `/imagelab/api/downloads` | every tracked download, active or finished |
| POST | `/imagelab/api/downloads` | start one — `{version_id, folder?, filename?}` from CivitAI, or `{url, folder, filename?}` from any https URL (Hugging Face) |
| DELETE | `/imagelab/api/downloads/{version_id}` | cancel, or dismiss a finished row |
| POST | `/imagelab/api/downloads/{version_id}/retry` | start a failed / cancelled row again with the version or URL, folder and filename it had (rows carry `url` for URL downloads) |
| DELETE | `/imagelab/api/models/{type}/{filename}` | delete a local model, drop it from the index |
| GET | `/imagelab/api/models/inspect/{type}/{filename}` | what a safetensors file holds: name prefixes, VAE / text encoder inside |
| POST | `/imagelab/api/models/sort` | move diffusion-model-only files out of `checkpoints/` (also runs at startup and after each download) |
| GET | `/imagelab/api/favorites` | list; ETag'd |
| POST | `/imagelab/api/favorites` | save `{filename, subfolder?, type?}` out of temp/ |
| DELETE | `/imagelab/api/favorites/{date}/{filename}` | remove one |
| GET | `/imagelab/api/favorites/view` | the image bytes for one favorite |
| GET | `/imagelab/api/preview-size` | the cap on sampler previews (long side, px); `unlimited: true` = render size |
| POST | `/imagelab/api/preview-size` | `{size}`: `64..8192`, or `0` / `"max"` for no cap. Takes effect at once, no restart |

## Install

Clone into ComfyUI's `custom_nodes/` and restart ComfyUI. Routes are registered at import time, so
a running server will not pick them up — aiohttp freezes its router once the app has started.

## Configuration

Read from the process environment, or from a `.env` beside this file (the environment wins):

| | |
|---|---|
| `CIVITAI_API_KEY` | CivitAI token. Downloads 401 without it. |
| `IMAGELAB_AUTO_DOWNLOAD` | model ids to fetch on startup |
| `IMAGELAB_PREVIEW_SIZE` | Sampler preview cap at start. Default: 1024, or ComfyUI's `--preview-size` if that is larger. `0` = no cap. |
| `IMAGELAB_BATCH_PREVIEW` | `0` turns off the batch live preview (a batch of N sends its N pictures side by side in one preview frame). On by default. |
| `IMAGELAB_CORE_SECONDARY` | `1` on a second ComfyUI that shares `models/` with the first: no hash scan, no checkpoint sort, no startup downloads, and only the favorites and version routes. One process must own the index and the downloads. |

`CIVITAI_API_KEY` is the same name the pod image and the project `.env` use.

## Modules

`__init__.py` loads only `hashing`, `api` and `downloads` (`api` pulls in `favorites`). The other
modules — `routes.py`, `workflows.py`, `imagelab_nodes.py`, `resources.py`, `civitai_routes.py`,
`parameters.py`, `schema_generator.py` — are not imported. They are reference only.
