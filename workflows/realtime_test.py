import os
from comfy_script.runtime import *

# ComfyUI server URL - configurable via environment variable
COMFYUI_SERVER_URL = os.environ.get('COMFYUI_SERVER_URL', 'http://127.0.0.1:8188/')
load(COMFYUI_SERVER_URL)

from comfy_script.runtime.nodes import *

def f(wf):
    seed = 0
    
    pos = 'an ancient tree in the jungle, highly detailed, intricate, digital art'
    neg = 'blurry, lowres, bad anatomy, worst quality'
    
    model, clip, vae = CheckpointLoaderSimple(Checkpoints.aniversePonyXL_v60)
    model, clip = LoraLoader(model, clip, Loras.LCMV2_PONYplus_PAseer, 1.0, 1.0)

    # Watch your specific file instead of Photoshop
    image, width, height = WatchImageFile(
        file_path="H:/test.png",
        wait_for_changes=True
    )
    
    latent = VAEEncode(image, vae)    
    latent = KSampler(model, seed, steps=4, cfg=1.7, sampler_name=Samplers.lcm,
        positive=CLIPTextEncode(pos, clip), 
        negative=CLIPTextEncode(neg, clip), 
        latent_image=latent, denoise=0.65)
    
    PreviewImage(VAEDecode(latent, vae))

# This queues the workflow whenever the queue is empty
# Combined with IS_CHANGED, it only re-generates when the file changes
# queue.when_empty(f)

# # Keep the script running so the watch loop can monitor the queue
# import time
# try:
#     while True:
#         time.sleep(1)
# except KeyboardInterrupt:
#     print("\nStopped watching.")