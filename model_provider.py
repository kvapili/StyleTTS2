import os
import logging
import soundfile as sf
import torch
from pathlib import Path
from inference import init_tts
from azure.storage.blob import ContainerClient

_LOGGER = logging.getLogger(__name__)

STYLETTS_MODELS_SAS_URL=os.getenv("STYLETTS_MODELS_SAS_URL")

def get_available_styletts_models():

    container_client = ContainerClient.from_container_url(STYLETTS_MODELS_SAS_URL)
    available_models = []
    for blob in container_client.list_blobs():
        if blob.name.endswith(".pth"):
            basename = os.path.splitext(blob.name)[0]
            available_models.append(basename)
    return available_models


def download_model_if_needed(model_name, local_dir):
    """
    Downloads a blob from Azure Blob Storage into styletts_models/
    only if it does not already exist locally.

    Args:
        filename: Full blob name inside container (e.g. "models/a.pth" or "a.pth")
        container_sas_url: Full container SAS URL including ?sv=... token

    Returns:
        Local file path
    """

    filename = f"{model_name}.pth"
    os.makedirs(local_dir, exist_ok=True)

    # Save locally using just the basename
    local_path = os.path.join(local_dir, os.path.basename(filename))

    # If file already exists locally → skip download
    if os.path.exists(local_path):
        print(f"File already exists: {local_path}")
        return local_path

    # Create container client from SAS URL
    container_client = ContainerClient.from_container_url(STYLETTS_MODELS_SAS_URL)

    # Get blob client
    blob_client = container_client.get_blob_client(filename)

    print(f"Downloading {filename}...")

    with open(local_path, "wb") as f:
        download_stream = blob_client.download_blob()
        f.write(download_stream.readall())

    print(f"Downloaded to {local_path}")

    return local_path
 

class StyleTTSProvider:
    def __init__(
        self,
        model_name,
        model_dir="Models",
        config_dir="Configs"
    ):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _LOGGER.info(
            f"Device: {self.device}",
        )
        config_path = f"{config_dir}/config_sts.yml"
        language = model_name[:2]
        model_path = download_model_if_needed(model_name, model_dir)
        self.model = init_tts(model_path=model_path, device=self.device, config_path=config_path, language=language)
        _LOGGER.info(
            "Initializing StyleTTS2Provider",
        )


    def synthesize(
        self,
        text: str,
        out_wav_path: Path,
        voice: str = "af",
    ):
        """
        Synthesizes `text` into a single WAV file at `out_wav_path`.
        Uses newline-based chunking and concatenates all audio.
        """
        _LOGGER.info("StyleTTS2Provider: Synthesizing text %s, voice=%s", text, voice)

        audio = self.model.run(text, diffusion_steps=10, embedding_scale=2)
        sf.write(str(out_wav_path), audio, samplerate=24000)
        _LOGGER.info("StyleTTS2 synthesis complete. Saved to %s", out_wav_path)
