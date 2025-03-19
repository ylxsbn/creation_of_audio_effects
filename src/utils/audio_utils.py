import torch


def frame_audio(data, frame_size, hop_size):
    """framing the audio data"""
    data = torch.nn.functional.pad(data, (frame_size // 2, frame_size // 2))

    data = data.unfold(dimension=-1, size=frame_size, step=hop_size)

    window = torch.hann_window(frame_size)
    data *= window

    return data


def reconstruct_audio(data, real_output_length, hop_size):
    """
    reconstructing audio signal from data of shape (frames_n, 1, frame_size) acquired by
    the framing procedure using overlap-add method
    """
    data = data.squeeze(1)

    frames_n, frame_size = data.shape

    window = torch.hann_window(frame_size)
    eps = torch.FloatTensor([1e-6])
    window = torch.max(window, eps.expand_as(window))

    output = torch.zeros(real_output_length + frame_size)
    for i in range(0, real_output_length, hop_size):
        output[i : i + frame_size] += data[i // hop_size, :] / window

    # removing padding
    output = output[frame_size // 2 : -frame_size // 2]
    
    # normalizing
    output = output.unsqueeze(0)

    return output

