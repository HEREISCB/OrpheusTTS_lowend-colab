"""
Wrapper script for Orpheus TTS to enforce FP16 and a safe max_model_len for T4.
"""
import os
import logging


os.environ["VLLM_MAX_MODEL_LEN"] = "50000"                
os.environ["VLLM_DISABLE_LOGGING"] = "1"
os.environ["VLLM_NO_USAGE_STATS"] = "1"
os.environ["VLLM_DO_NOT_TRACK"] = "1"
os.environ["GRADIO_ANALYTICS_ENABLED"] = "0"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:

    from vllm.engine.arg_utils import EngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from orpheus_tts.engine_class import OrpheusModel

    _orig_init = EngineArgs.__init__
    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)

        setattr(self, "dtype", "float16")
        setattr(self, "torch_dtype", "float16")
    EngineArgs.__init__ = _patched_init
    logger.info("Patched EngineArgs to force dtype=float16")

    _orig_from_args = AsyncLLMEngine.from_engine_args
    def _patched_from_engine_args(engine_args, **kwargs):
        engine_args.max_model_len = 50000               
        engine_args.gpu_memory_utilization = 0.9

        setattr(engine_args, "dtype", "float16")
        setattr(engine_args, "torch_dtype", "float16")
        logger.info(f"Patched from_engine_args: max_model_len={engine_args.max_model_len}, dtype={engine_args.dtype}")
        return _orig_from_args(engine_args, **kwargs)
    AsyncLLMEngine.from_engine_args = staticmethod(_patched_from_engine_args)
    logger.info("Patched AsyncLLMEngine.from_engine_args")

except Exception as e:
    logger.warning(f"Failed to apply patches: {e}")


logger.info("Starting Orpheus TTS UI…")
import orpheus

if __name__ == "__main__":
    demo = orpheus.create_ui()
    demo.launch(share=True)
