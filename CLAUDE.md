# ImageLabCore — the ComfyUI custom node

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

Check what is actually running: `GET /imagelab/version` reports the directory it loaded from and
the mtime of `api.py`.

## Things to know

- **Only `hashing`, `api`, `downloads` and `favorites` are imported.** `routes.py`, `workflows.py`,
  `imagelab_nodes.py`, `resources.py`, `civitai_routes.py`, `parameters.py` and
  `schema_generator.py` are reference only and deliberately not
  loaded. Editing one of those and seeing nothing change is an easy hour to lose.
- **The CivitAI token is `CIVITAI_API_KEY`**, the same name the image and `.env` use. One name
  for one secret — do not add an alias.
- **No third-party dependencies.** It imports `aiohttp` (ComfyUI has it) plus `folder_paths` and
  `server`, which are ComfyUI itself. There is no `requirements.txt` and adding one means the image
  build starts installing things.
- **Every list endpoint is ETag'd.** Clients poll; 304 keeps that nearly free. Keep it that way.

**The project rules — pods, money, pulling images before a terminate — live in
`../ImageLabDocker/CLAUDE.md`. Read that first.**
