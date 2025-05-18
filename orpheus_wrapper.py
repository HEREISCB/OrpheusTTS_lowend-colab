#!/usr/bin/env python3
"""
Universal Wrapper for Orpheus TTS on GPUs with varying VRAM.

– Detects GPU VRAM (via torch.cuda.get_device_properties)
– Sets safe max_model_len and gpu_memory_utilization
– Forces float16 (FP16) to avoid BF16 on compute capability < 8.0
"""
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1) Dynamically pick limits based on GPU VRAM
def get_gpu_settings():
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        props = torch.cuda.get_device_properties(0)
        total_gb = props.total_memory / (1024 ** 3)
        logger.info(f"Detected GPU with {total_gb:.1f} GB VRAM (cc={props.major}.{props.minor})")

        # T4 (~16 GB)
        if total_gb >= 12:
            return dict(max_model_len=50_000, gpu_mem_util=0.90)
        # 1660 (~6 GB)
        else:
            return dict(max_model_len=20_000, gpu_mem_util=0.80)
    except Exception as e:
        logger.warning(f"Could not detect GPU details, falling back to safe defaults: {e}")
        return dict(max_model_len=10_000, gpu_mem_util=0.75)

settings = get_gpu_settings()
os.environ["VLLM_MAX_MODEL_LEN"] = str(settings["max_model_len"])
os.environ["VLLM_GPU_MEMORY_UTILIZATION"] = str(settings["gpu_mem_util"])
os.environ["VLLM_DISABLE_LOGGING"] = "1"
os.environ["VLLM_NO_USAGE_STATS"] = "1"
os.environ["VLLM_DO_NOT_TRACK"] = "1"
os.environ["GRADIO_ANALYTICS_ENABLED"] = "0"

try:
    # 2) Monkey-patch EngineArgs to force FP16
    from vllm.engine.arg_utils import EngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from orpheus_tts.engine_class import OrpheusModel

    _orig_init = EngineArgs.__init__
    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        # override any dtype settings
        setattr(self, "dtype", "float16")
        setattr(self, "torch_dtype", "float16")
    EngineArgs.__init__ = _patched_init
    logger.info("EngineArgs patched to force float16")

    # 3) Monkey-patch loader to enforce our settings
    _orig_from_args = AsyncLLMEngine.from_engine_args
    def _patched_from_engine_args(engine_args, **kwargs):
        engine_args.max_model_len = settings["max_model_len"]
        engine_args.gpu_memory_utilization = settings["gpu_mem_util"]
        # re‑apply FP16 override
        setattr(engine_args, "dtype", "float16")
        setattr(engine_args, "torch_dtype", "float16")
        logger.info(f"Loader patched: max_model_len={engine_args.max_model_len}, "
                    f"gpu_mem_util={engine_args.gpu_memory_utilization}, dtype={engine_args.dtype}")
        return _orig_from_args(engine_args, **kwargs)

    AsyncLLMEngine.from_engine_args = staticmethod(_patched_from_engine_args)
    logger.info("AsyncLLMEngine.from_engine_args patched successfully")

except Exception as e:
    logger.warning(f"Failed to patch vLLM engine args: {e}")

# 4) Launch the Orpheus TTS Gradio UI
logger.info("Starting Orpheus TTS UI…")
import orpheus

if __name__ == "__main__":
    demo = orpheus.create_ui()
    demo.launch(share=True)
