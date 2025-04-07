import torch
import torchaudio


def frame_audio(data, frame_size, hop_size):
    """Framing the audio data"""
    data = torch.nn.functional.pad(data, (frame_size // 2, frame_size // 2))
    data = data.unfold(dimension=-1, size=frame_size, step=hop_size)

    return data


def reconstruct_audio(data, real_output_length=32000, hop_size=2048, device='auto'):
    """
    Reconstructing audio signal from data of shape (frames_n, 1, frame_size) acquired by
    the framing procedure using overlap-add method
    """
    data = data.squeeze(1)
    _, frame_size = data.shape

    if device == 'auto':
        device = "cuda" if torch.cuda.is_available() else "cpu"

    window = torch.hann_window(frame_size, device=device)

    output = torch.zeros(real_output_length + frame_size)
    for i in range(0, real_output_length, hop_size):
        output[i: i + frame_size] += data[i // hop_size, :] * window

    # removing padding
    output = output[frame_size // 2: -frame_size // 2]
    output = output.unsqueeze(0)

    return output


def reconstruct_from_spectrogram(spec, phase, real_output_length=32000):
    """Reconstructing the output signal from the spectrogram given the original phase"""
    spec = spec.exp().sqrt()
    complex_spec = torch.polar(spec, phase)
    complex_spec = complex_spec.squeeze(0).squeeze(0)
    complex_spec = complex_spec.transpose(0, 1)

    n_fft = 254
    window = torch.hann_window(n_fft)

    output = torch.istft(complex_spec,
                         n_fft=n_fft,
                         win_length=n_fft,
                         hop_length=n_fft // 2,
                         window=window,
                         normalized=False
                         )

    output = output[:real_output_length]
    return output
