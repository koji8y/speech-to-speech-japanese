import logging
from time import perf_counter
from baseHandler import BaseHandler
from lightning_whisper_mlx import LightningWhisperMLX
import numpy as np
from rich.console import Console
import torch
from reazonspeech.nemo.asr import load_model, transcribe, audio_from_path,audio_from_numpy


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

console = Console()



class LightningWhisperSTTHandler(BaseHandler):
    """
    Handles the Speech To Text generation using a Whisper model.
    """

    def setup(
        self,
        model_name="distil-whisper/distil-large-v3",
        device="cuda",
        torch_dtype="float16",
        compile_mode=None,
        gen_kwargs={},
    ):
        self.device = device
        #self.model = LightningWhisperMLX(
        #    model="distil-large-v3", batch_size=6, quant=None
        #)
        self.model=load_model()

        self.warmup()

    def warmup(self):
        logger.info(f"Warming up {self.__class__.__name__}")
        return

        # 2 warmup steps for no compile or compile mode with CUDA graphs capture
        n_steps = 1
        dummy_input = np.array([0] * 512)

        for _ in range(n_steps):
            _ = self.model.transcribe(dummy_input)["text"].strip()

    def process(self, spoken_prompt):
        logger.debug("infering whisper...")

        global pipeline_start
        pipeline_start = perf_counter()
        RATE=16000
        audio = audio_from_numpy(spoken_prompt,RATE)

        #pred_text = self.model.transcribe(spoken_prompt,language="ja")["text"].strip()
        result = transcribe(self.model, audio)
        pred_text = result.text
        torch.mps.empty_cache()

        logger.debug("finished whisper inference")
        console.print(f"[yellow]USER: {pred_text}")

        yield pred_text
