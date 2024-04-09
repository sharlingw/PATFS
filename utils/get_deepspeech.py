from utils.audio_handler import AudioHandler
import torch as th
import torchaudio as ta


def process_audio(ds_path, audio, sample_rate, multiface):
    config = {}
    config["deepspeech_graph_fname"] = ds_path
    config["audio_feature_type"] = "deepspeech"
    config["num_audio_features"] = 29

    config["audio_window_size"] = 16
    config["audio_window_stride"] = 1

    tmp_audio = {"subj": {"seq": {"audio": audio, "sample_rate": sample_rate}}}
    audio_handler = AudioHandler(config)
    return audio_handler.process(tmp_audio, multiface)["subj"]["seq"]["audio"]


def audio_chunking(audio: th.Tensor, melspec, length, frame_rate: int = 60, chunk_size: int = 16000):
    """
    :param audio: 1 x T tensor containing a 16kHz audio signal
    :param frame_rate: frame rate for video (we need one audio chunk per video frame)
    :param chunk_size: number of audio samples per chunk
    :return: num_chunks x chunk_size tensor containing sliced audio
    """
    samples_per_frame = 16000 // frame_rate
    padding = (chunk_size - samples_per_frame) // 2
    audio = th.nn.functional.pad(audio.unsqueeze(0), pad=[padding, padding]).squeeze(0)
    anchor_points = list(range(chunk_size // 2, audio.shape[-1] - chunk_size // 2, samples_per_frame))
    audio = th.cat([audio[:, i - chunk_size // 2:i + chunk_size // 2] for i in anchor_points], dim=0)

    audio = audio.permute(1, 0).unsqueeze(0)
    audio = th.nn.functional.interpolate(input=audio, size=length, mode='linear', align_corners=True).squeeze().permute(
        1, 0)
    audio = melspec(audio)
    audio = th.log(audio.clamp(min=1e-10, max=None))
    return audio

def load_audio(wave_file: str):
    """
    :param wave_file: .wav file containing the audio input
    :return: 1 x T tensor containing input audio resampled to 16kHz
    """
    audio, sr = ta.load(wave_file)
    if not sr == 16000:
        audio = ta.transforms.Resample(sr, 16000)(audio)
    if audio.shape[0] > 1:
        audio = th.mean(audio, dim=0, keepdim=True)
    # normalize such that energy matches average energy of audio used in training
    audio = 0.01 * audio / th.mean(th.abs(audio))
    return audio