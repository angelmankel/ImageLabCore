"""save_video_ws: the video frame header, 4 MiB chunking, and the node's send path (no ComfyUI needed)."""
import hashlib
import importlib.util
from pathlib import Path
import struct
import sys
import types
import unittest

spec = importlib.util.spec_from_file_location("save_video_ws", Path(__file__).resolve().parent.parent / "save_video_ws.py")
svw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(svw)


def reassemble(payloads):
    heads, parts = [], []
    for p in payloads:
        h, chunk = svw.parse_payload(p)
        heads.append(h)
        parts.append(chunk)
    return heads, b"".join(parts)


class FrameTest(unittest.TestCase):
    def test_small_clip_is_one_frame_with_the_header(self):
        data = b"\x00\x00\x00\x18ftypisom" + bytes(range(200))
        payloads = svw.chunk_payloads(data, {"video_id": "v", "prompt_id": "p", "node_id": "9", "mime": "video/mp4"})
        self.assertEqual(len(payloads), 1)
        (n,) = struct.unpack(">I", payloads[0][:4])
        head, chunk = svw.parse_payload(payloads[0])
        self.assertEqual(len(payloads[0]), 4 + n + len(data))
        self.assertEqual(head, {"video_id": "v", "prompt_id": "p", "node_id": "9", "mime": "video/mp4",
                                "size": len(data), "index": 0, "total": 1, "offset": 0})
        self.assertEqual(chunk, data)

    def test_over_4mb_splits_in_order_under_4mib_and_reassembles_byte_identical(self):
        data = bytes((i * 7 + 3) % 251 for i in range(9 * 1024 * 1024 + 123))
        payloads = svw.chunk_payloads(data, {"video_id": "v", "mime": "video/mp4"})
        self.assertEqual(len(payloads), 3)
        for p in payloads:
            self.assertLessEqual(4 + len(p), svw.MAX_FRAME_BYTES)  # + the event type ComfyUI prepends
        heads, joined = reassemble(payloads)
        self.assertEqual([h["index"] for h in heads], [0, 1, 2])
        self.assertTrue(all(h["total"] == 3 and h["size"] == len(data) for h in heads))
        self.assertEqual([h["offset"] for h in heads], [0, svw.CHUNK_BYTES, 2 * svw.CHUNK_BYTES])
        self.assertEqual(joined, data)

    def test_exact_multiple_and_empty(self):
        self.assertEqual(len(svw.chunk_payloads(b"x" * 20, {}, chunk_bytes=10)), 2)
        heads, joined = reassemble(svw.chunk_payloads(b"", {}))
        self.assertEqual((heads[0]["total"], heads[0]["size"], joined), (1, 0, b""))

    def test_truncated_payload_is_rejected(self):
        p = svw.chunk_payloads(b"abc", {"video_id": "v"})[0]
        with self.assertRaises(ValueError):
            svw.parse_payload(p[:2])
        with self.assertRaises(ValueError):
            svw.parse_payload(p[:6])


class _NodeCase(unittest.TestCase):
    """A fake PromptServer that records frames, and an encoder that returns ~5 MiB of bytes."""

    def setUp(self):
        self.sent = []
        sent = self.sent

        class FakeServer:
            client_id = "agent-1"
            last_prompt_id = "prompt-1"

            def send_sync(self, event, data, sid=None):
                sent.append((event, data, sid))

        server_mod = types.ModuleType("server")
        server_mod.PromptServer = type("PromptServer", (), {"instance": FakeServer()})
        self._saved = {k: sys.modules.get(k) for k in ("server", "comfy_execution", "comfy_execution.utils")}
        sys.modules["server"] = server_mod
        sys.modules.pop("comfy_execution", None)
        sys.modules.pop("comfy_execution.utils", None)
        self.encoded = bytes(range(256)) * (5 * 4096 + 1)  # ~5.0 MiB
        self._encode = svw.encode_video
        svw.encode_video = lambda video, f, c: self.encoded

    def tearDown(self):
        svw.encode_video = self._encode
        for k, v in self._saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


class NodeTest(_NodeCase):
    def test_sends_ordered_chunks_to_the_prompt_client(self):
        self.assertEqual(svw.SaveVideoWebsocket().send_video(object(), "mp4", "h264", unique_id="12"), {})
        self.assertEqual(len(self.sent), 2)
        self.assertTrue(all(e == svw.VIDEO_EVENT and sid == "agent-1" for e, _, sid in self.sent))
        heads, joined = reassemble([d for _, d, _ in self.sent])
        self.assertEqual(joined, self.encoded)
        self.assertEqual(heads[0]["prompt_id"], "prompt-1")
        self.assertEqual(heads[0]["node_id"], "12")
        self.assertEqual(heads[0]["mime"], "video/mp4")
        self.assertEqual(heads[0]["video_id"], heads[1]["video_id"])
        self.assertEqual(heads[1]["sha256"], hashlib.sha256(joined).hexdigest())

    def test_webm_needs_av1(self):
        with self.assertRaises(ValueError):
            svw.SaveVideoWebsocket().send_video(object(), "webm", "h264")
        self.assertEqual(self.sent, [])

    def test_declares_an_output_node_without_outputs(self):
        it = svw.SaveVideoWebsocket.INPUT_TYPES()["required"]
        self.assertEqual(it["video"][0], "VIDEO")
        self.assertEqual(it["format"][0], ["mp4", "webm"])
        self.assertEqual(it["codec"][0], ["h264", "av1"])
        self.assertTrue(svw.SaveVideoWebsocket.OUTPUT_NODE)
        self.assertEqual(svw.SaveVideoWebsocket.RETURN_TYPES, ())
        self.assertEqual(svw.SaveVideoWebsocket.CATEGORY, "imagelab")

    def test_declares_an_optional_int_crf(self):
        types_ = svw.SaveVideoWebsocket.INPUT_TYPES()
        self.assertNotIn("crf", types_["required"])
        kind, opts = types_["optional"]["crf"]
        self.assertEqual((kind, opts["min"], opts["max"]), ("INT", 0, 63))


class CrfTest(_NodeCase):
    """The optional crf: passed to the encoder only when set, refused outside the codec's range."""

    def setUp(self):
        super().setUp()
        self.calls = []
        calls, encoded = self.calls, self.encoded

        def encode(video, f, c, *rest):
            calls.append((f, c) + rest)
            return encoded

        svw.encode_video = encode

    def test_unset_crf_makes_the_old_three_argument_call(self):
        svw.SaveVideoWebsocket().send_video(object(), "mp4", "h264", unique_id="1")
        self.assertEqual(self.calls, [("mp4", "h264")])

    def test_crf_reaches_the_encoder(self):
        svw.SaveVideoWebsocket().send_video(object(), "mp4", "h264", crf=16, unique_id="1")
        svw.SaveVideoWebsocket().send_video(object(), "webm", "av1", crf=63)
        self.assertEqual(self.calls, [("mp4", "h264", 16), ("webm", "av1", 63)])
        self.assertEqual(len(self.sent), 4)

    def test_crf_out_of_range_is_refused_before_encoding(self):
        for codec, crf in (("h264", 52), ("h264", -1), ("av1", 64), ("h264", 20.5), ("h264", "20")):
            with self.assertRaises(ValueError, msg=f"{codec} {crf!r}"):
                svw.SaveVideoWebsocket().send_video(object(), "mp4", codec, crf=crf)
        self.assertEqual((self.calls, self.sent), ([], []))

    def test_h264_takes_51_and_a_whole_float(self):
        svw.SaveVideoWebsocket().send_video(object(), "mp4", "h264", crf=51)
        svw.SaveVideoWebsocket().send_video(object(), "mp4", "av1", crf=30.0)
        self.assertEqual(self.calls, [("mp4", "h264", 51), ("mp4", "av1", 30)])


class EncodeTest(unittest.TestCase):
    """encode_video calls VideoInput.save_to like SaveVideo does, with crf only when set."""

    def setUp(self):
        latest = types.ModuleType("comfy_api.latest")
        latest.Types = types.SimpleNamespace(VideoContainer=lambda v: ("container", v), VideoCodec=lambda v: ("codec", v))
        pkg = types.ModuleType("comfy_api")
        pkg.latest = latest
        self._saved = {k: sys.modules.get(k) for k in ("comfy_api", "comfy_api.latest")}
        sys.modules.update({"comfy_api": pkg, "comfy_api.latest": latest})
        self.calls = []
        calls = self.calls

        class FakeVideo:
            def save_to(self, path, **kwargs):
                calls.append(kwargs)
                path.write(b"clip")

        self.video = FakeVideo()

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    def test_without_crf_save_to_gets_no_crf(self):
        self.assertEqual(svw.encode_video(self.video, "mp4", "h264"), b"clip")
        self.assertEqual(self.calls, [{"format": ("container", "mp4"), "codec": ("codec", "h264")}])

    def test_with_crf_save_to_re_encodes(self):
        svw.encode_video(self.video, "webm", "av1", 40)
        self.assertEqual(self.calls, [{"format": ("container", "webm"), "codec": ("codec", "av1"), "crf": 40}])


if __name__ == "__main__":
    unittest.main()
