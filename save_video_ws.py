"""SaveVideoWebsocket: a finished video goes to the client over ComfyUI's own websocket, never to disk.

ComfyUI's `SaveImageWebsocket` sends images through the preview path; there is no such node for
video, and `SaveVideo` writes into `output/`. This node encodes the `VIDEO` in memory (the same
`VideoInput.save_to` call `SaveVideo` makes, with a BytesIO instead of a path, so no file appears in
`output/` or `temp/`) and sends the bytes as binary frames of its own event type on the running
`PromptServer`.

Why the PromptServer and not ImageLabCore's own route: the job's client (the ImageLab agent) is
already connected to ComfyUI's `/ws?clientId=<id>` for this prompt, and `server.client_id` is that
id while the prompt runs. `PromptServer.instance.send_sync(event, bytes, sid)` queues the frame on
the same socket and the same ordered message queue as `executing` / `progress` / `executed`, so the
chunks arrive after the node starts and before its `executed`, exactly like SaveImageWebsocket's
images. (Without a client_id, ComfyUI broadcasts, as it does for previews.)

Wire format (every number big-endian; ComfyUI's `encode_bytes` writes the first field):

    [4 bytes: event type VIDEO_EVENT = 0x494C5643 "ILVC"]
    [4 bytes: n][n bytes: UTF-8 JSON header][chunk bytes]

    header = { "video_id": hex, "prompt_id", "node_id", "display_node_id",
               "mime": "video/mp4" | "video/webm", "format", "codec",
               "size": total bytes, "sha256": hex of the whole clip,
               "index": 0-based chunk, "total": chunk count, "offset": byte offset }

Every frame is at most 4 MiB in all (`MAX_FRAME_BYTES`); a client concatenates the chunks of one
`video_id` in `index` order (they are sent in order) and checks the result against `size` / `sha256`.

Inputs match `SaveVideo` minus `filename_prefix` (imagelab-server `packages/graphs` builds them):
`video` (VIDEO), `format` mp4 | webm, `codec` h264 | av1 (webm takes AV1 only, as in SaveVideo).
Optional `crf` (INT 0..63; H.264 stops at 51): when the prompt carries it, the clip is re-encoded with
that CRF, the way SaveVideo's `codec.encoding = re-encode` does (`save_to(..., crf=)`). When it is
absent the call is exactly the old one, so the encoder keeps its own default (H.264 23, AV1 30).
"""
import hashlib
import io
import json
import struct
import time
import uuid

VIDEO_EVENT = 0x494C5643  # "ILVC" — far from ComfyUI's own BinaryEventTypes (1..4)
MAX_FRAME_BYTES = 4 * 1024 * 1024
# Room for the 4-byte event type, the 4-byte length and the JSON header inside MAX_FRAME_BYTES.
HEADER_RESERVE = 16 * 1024
CHUNK_BYTES = MAX_FRAME_BYTES - HEADER_RESERVE

MIME = {"mp4": "video/mp4", "webm": "video/webm"}
# Highest CRF each codec takes (SaveVideo's re-encode options: H.264 0..51, AV1 0..63).
MAX_CRF = {"h264": 51, "av1": 63}


def chunk_payloads(data: bytes, header: dict, chunk_bytes: int = CHUNK_BYTES) -> list:
    """The payloads (everything after the 4-byte event type) that carry `data`, in order.

    `header` is copied into every chunk with `size`, `index`, `total` and `offset` filled in.
    An empty `data` still makes one (empty) chunk, so the client always sees the video end.
    """
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    size = len(data)
    total = max(1, -(-size // chunk_bytes))
    view = memoryview(data)
    out = []
    for index in range(total):
        offset = index * chunk_bytes
        head = dict(header, size=size, index=index, total=total, offset=offset)
        raw = json.dumps(head, separators=(",", ":")).encode("utf-8")
        if len(raw) + 8 > HEADER_RESERVE:
            raise ValueError(f"video frame header is {len(raw)} bytes, max {HEADER_RESERVE - 8}")
        out.append(struct.pack(">I", len(raw)) + raw + bytes(view[offset:offset + chunk_bytes]))
    return out


def parse_payload(payload: bytes):
    """(header, chunk bytes) of one payload; the inverse of `chunk_payloads`. Raises ValueError."""
    if len(payload) < 4:
        raise ValueError("video frame shorter than its length field")
    (n,) = struct.unpack(">I", payload[:4])
    if len(payload) < 4 + n:
        raise ValueError("video frame shorter than its header")
    return json.loads(payload[4:4 + n].decode("utf-8")), payload[4 + n:]


def check_crf(codec_name: str, crf):
    """`crf` as an int, or None when unset. Raises ValueError outside 0..MAX_CRF[codec]."""
    if crf is None:
        return None
    if isinstance(crf, bool) or not isinstance(crf, (int, float)) or crf != int(crf):
        raise ValueError(f"crf must be a whole number, got {crf!r}")
    top = MAX_CRF.get(codec_name, 63)
    if not 0 <= crf <= top:
        raise ValueError(f"crf for {codec_name} must be 0..{top}, got {crf}")
    return int(crf)


def encode_video(video, format_name: str, codec_name: str, crf=None) -> bytes:
    """The whole clip, encoded in memory with ComfyUI's own `save_to` (no file is written).

    `crf` set re-encodes with it, as SaveVideo's `re-encode` option does; None makes the old call.
    """
    from comfy_api.latest import Types

    buf = io.BytesIO()
    kwargs = {} if crf is None else {"crf": crf}
    video.save_to(buf, format=Types.VideoContainer(format_name), codec=Types.VideoCodec(codec_name), **kwargs)
    return buf.getvalue()


class SaveVideoWebsocket:
    """Encode a VIDEO to mp4 (H.264 / AV1) or webm (AV1) in memory and send it over the websocket.

    See the module docstring for why it goes through ComfyUI's PromptServer to the prompt's
    `client_id`, and for the frame header. Nothing is written to `output/` or `temp/`.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO", {"tooltip": "The video to send."}),
                "format": (["mp4", "webm"], {"default": "mp4", "tooltip": "Container. WebM takes AV1 only."}),
                "codec": (["h264", "av1"], {"default": "h264"}),
            },
            "optional": {
                "crf": ("INT", {
                    "default": 23, "min": 0, "max": 63, "step": 1,
                    "tooltip": "Re-encode with this CRF: lower is higher quality and a bigger file. "
                               "H.264 takes 0..51, AV1 0..63. Left out of the prompt, the encoder default is kept.",
                }),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ()
    FUNCTION = "send_video"
    OUTPUT_NODE = True
    CATEGORY = "imagelab"
    DESCRIPTION = "Sends the video to the websocket client as binary frames; writes no file."

    def send_video(self, video, format="mp4", codec="h264", crf=None, unique_id=None):
        if format not in MIME:
            raise ValueError(f"format must be one of {sorted(MIME)}, got {format!r}")
        if format == "webm" and codec != "av1":
            raise ValueError("WebM output requires the AV1 codec")
        crf = check_crf(codec, crf)
        data = encode_video(video, format, codec) if crf is None else encode_video(video, format, codec, crf)

        from server import PromptServer

        server = PromptServer.instance
        prompt_id, node_id = getattr(server, "last_prompt_id", None), unique_id
        try:
            from comfy_execution.utils import get_executing_context

            ctx = get_executing_context()
            if ctx is not None:
                prompt_id, node_id = ctx.prompt_id, ctx.node_id
        except ImportError:
            pass
        header = {
            "video_id": uuid.uuid4().hex,
            "prompt_id": prompt_id,
            "node_id": None if node_id is None else str(node_id),
            "display_node_id": None if unique_id is None else str(unique_id),
            "mime": MIME[format],
            "format": format,
            "codec": codec,
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        sid = server.client_id
        for payload in chunk_payloads(data, header):
            server.send_sync(VIDEO_EVENT, payload, sid)
        print(f"[ImageLab] SaveVideoWebsocket: {len(data)} bytes {MIME[format]} sent to {sid or 'all clients'}")
        return {}

    @classmethod
    def IS_CHANGED(cls, **_kwargs):
        # Always run: a cached output node sends nothing, and the client would wait for a video.
        return time.time()


NODE_CLASS_MAPPINGS = {"SaveVideoWebsocket": SaveVideoWebsocket}
NODE_DISPLAY_NAME_MAPPINGS = {"SaveVideoWebsocket": "Save Video (Websocket)"}
