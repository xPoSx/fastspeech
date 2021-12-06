import torch
from torch import nn
import torchaudio
from tqdm import tqdm
from src.model import FastSpeech
from torch.utils.data import DataLoader
from src.aligner import GraphemeAligner
from src.dataset import LJSpeechDataset
from src.collate import LJSpeechCollator
from src.melspec import MelSpectrogramConfig, MelSpectrogram
from src.vocoder import Vocoder
from src.logger import WanDBWriter

warmup_steps = 4000
num_epochs = 20
bs = 4

val = [
    'A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest',
    'Massachusetts Institute of Technology may be best known for its math, science and engineering education',
    'Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space',
]

tokenizer = torchaudio.pipelines.TACOTRON2_GRIFFINLIM_CHAR_LJSPEECH.get_text_processor()

val_batch = tokenizer(val)[0].to('cuda')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger = WanDBWriter()
featurizer = MelSpectrogram(MelSpectrogramConfig())
aligner = GraphemeAligner().to(device)
dataloader = DataLoader(LJSpeechDataset('.'), batch_size=bs, collate_fn=LJSpeechCollator())
vocoder = Vocoder().to(device).eval()

model = FastSpeech().to(device)

try:
    model.load_state_dict(torch.load('fastspeech'))
except:
    pass

loss_func = nn.MSELoss()
dur_loss_func = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=5e-2, betas=(.9, .98), eps=1e-9)
scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda x: min((x + 1) ** (-0.5), (x + 1) * (warmup_steps ** (-1.5))))
# scheduler = torch.optim.lr_scheduler.StepLR(opt, 3275, gamma=0.8)

for e in range(num_epochs):
    model.train()
    for i, batch in tqdm(enumerate(dataloader)):
        opt.zero_grad()

        tokens = batch.tokens.to(device)
        durations = batch.durations.to(device).float()
        mels = featurizer(batch.waveform).to(device)
        mels = mels.transpose(1, 2)
        preds, dur_preds = model(batch.tokens.to(device), durations)

        min_dur = min(durations.shape[-1], dur_preds.shape[-1])
        dur_loss = dur_loss_func(durations[:, :min_dur], dur_preds[:, :min_dur])
        min_mel = min(mels.shape[1], preds.shape[1])
        mel_loss = loss_func(mels[:, :min_mel, :], preds[:, :min_mel, :])
        loss = dur_loss + mel_loss
        loss.backward()
        opt.step()
        logger.add_metrics({"Loss": loss.item(), "Melspec Loss": mel_loss.item(), "Duration Loss": dur_loss.item(),
                            "Learning rate": scheduler.get_last_lr()[0]})
        scheduler.step()

    with torch.no_grad():
        model.eval()
        preds = model(val_batch, None)[0]
        aud_val = vocoder.inference(preds.transpose(1, 2)).cpu()
        j = 0
        for audio, tr in zip(aud_val, val):
            logger.add_audio(audio.cpu(), tr, 'VALIDATION ' + str(j) + ' ')
            mel_p = preds[j].detach()
            logger.add_spectrogram(mel_p.cpu(), tr, 'VALIDATION ' + str(j) + ' ')
            j += 1

    torch.save(model.state_dict(), 'fastspeech')
