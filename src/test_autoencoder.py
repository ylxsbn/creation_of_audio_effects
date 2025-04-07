import torchaudio
import soundfile
import torch
from src.transforms import GetMelSpectrogram

import matplotlib.pyplot as plt

from src.audio_mae import models_mae
from src.utils.io_utils import ROOT_PATH


def prepare_model(chkpt_dir, arch='mae_vit_base_patch16'):
    model = getattr(models_mae, arch)(in_chans=1, audio_exp=True, img_size=(1024,128))
    checkpoint = torch.load(chkpt_dir, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=False)
    return model


def main():
    x, sr = torchaudio.load("audioset-test.wav")
    target_sr = 16000
    x = torchaudio.functional.resample(x, sr, target_sr)

    
    mel = GetMelSpectrogram()
    x = mel(x)

    x = (x - x.mean()) / x.std()

    model = prepare_model(ROOT_PATH / "saved" / "audio_mae" / "pretrained.pth")

    x = x.unsqueeze(0)
    print(x.shape)

    pred = model.forward_custom(x)

    loss_recon, _, _ = model(x)

    print("mse loss: ", loss_recon)

    plt.figure(figsize=(20, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(20 * x.squeeze().detach().numpy().T, origin='lower', interpolation='nearest',  aspect='auto')
    plt.subplot(1, 2, 2)
    plt.imshow(20 * pred.squeeze().detach().numpy().T, origin='lower', interpolation='nearest',  aspect='auto')

    plt.show()

if __name__ == "__main__":
    main()
