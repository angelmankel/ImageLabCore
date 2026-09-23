# Agent instructions — ImageLabCore (the ComfyUI custom node)

**Read `CLAUDE.md` in this repo first**, then `../ImageLabDocker/CLAUDE.md`, which holds the project
rules (pods, money, pulling images before a terminate) and is the entry point for all three repos.

## The short version

- A ComfyUI custom node that adds an HTTP API and **no nodes**: model hash index, CivitAI downloads,
  persistent favorites. Routes live under `/imagelab/api/*` — `/imagelab/` itself is where the pod
  serves ImageLab, so the prefix is not free.
- Only `hashing.py`, `api.py`, `downloads.py` and `favorites.py` are imported. The other modules are
  reference only and deliberately not loaded; editing one and seeing nothing change is an easy hour
  to lose.
- **No third-party dependencies.** `aiohttp`, `folder_paths` and `server` come from ComfyUI itself.
  Adding a requirement means the image build starts installing things.
- Every list endpoint is ETag'd; clients poll. Keep it that way.
- `CIVITAI_API_KEY` is the one name for that secret in every repo. Do not add an alias.

## Changing it on a running pod

No image rebuild: `cd ../ImageLabDocker && scripts/push-node.sh <ip:port>`. It pushes every `.py`,
then POSTs `/manager/reboot`, which re-execs ComfyUI in place — about 15 seconds, same port, Traefik
untouched. Adding a route works the same as changing one. Check with
`GET /imagelab/api/version`, which reports the directory it loaded from and the mtime of `api.py`.

## Known rough edge

Downloads run on one connection, about 1 MB/s from CivitAI — a 7 GB checkpoint takes an hour.
`ImageLabDocker/download-models.sh` gets 3-7 MB/s with aria2 (eight connections) for the same file.
Moving `downloads.py` onto aria2 with a curl fallback is the obvious fix and has not been done.
