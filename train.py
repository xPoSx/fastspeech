import torch
import random
from torch import nn
import torchaudio
from itertools import repeat
from src.model import FastSpeech
from torch.utils.data import DataLoader
from src.aligner import GraphemeAligner
from src.dataset import LJSpeechDataset
from src.collate import LJSpeechCollator
from src.melspec import MelSpectrogramConfig, MelSpectrogram
from src.vocoder import Vocoder
from src.logger import WanDBWriter


def loop_loader(loader):
    for new_loader in repeat(loader):
        yield from new_loader


warmup_steps = 500
num_iters = 5000
bs = 128

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

loss = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=5e-2, betas=(.9, .98), eps=1e-9)
scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda x: min((x + 1) ** (-0.5), (x + 1) * (warmup_steps ** (-1.5))))


model.train()
i = 1
for batch in loop_loader(dataloader):
    opt.zero_grad()
    batch.durations = aligner(
        batch.waveform.to('cuda'), batch.waveform_length, batch.transcript
    )

    tokens = batch.tokens.to(device)
    mels = featurizer(batch.waveform).to(device)
    mel_len = mels.shape[-1] - (mels == -11.5129251)[:, 0, :].sum(dim=-1)
    mels = mels.transpose(1, 2)
    durs = batch.durations.to(device) * mel_len.unsqueeze(-1)
    preds, dur_preds = model(batch.tokens.to(device), durs)

    min_dur = min(durs.shape[-1], dur_preds.shape[-1])
    dur_loss = loss(durs[:, :min_dur], dur_preds[:, :min_dur])
    min_mel = min(mels.shape[1], preds.shape[1])
    mel_loss = loss(mels[:, :min_mel, :], preds[:, :min_mel, :])

    loss = dur_loss + mel_loss
    loss.backward()
    opt.step()
    logger.add_metrics({"Loss": loss.item(), "Learning rate": 3e-4})

    if i % 20 == 0:
        rand_idx = random.randint(0, bs)
        tr = batch.transcript[rand_idx]
        mel_t = mels[rand_idx].detach()
        mel_p = preds[rand_idx].detach()
        aud_t = batch.waveform[rand_idx]
        aud_p = vocoder.inference(preds[:rand_idx + 1].transpose(1, 2))[-1]
        logger.add_audio(aud_p.cpu(), aud_t.cpu(), tr)
        logger.add_spectrogram(mel_p.cpu(), mel_t.cpu(), tr)
        print('duration loss:', dur_loss.item(), '\t', 'melspec loss:', mel_loss.item())
    scheduler.step()

    if i % 125 == 0:
        with torch.no_grad():
            model.eval()
            preds, _ = model(val_batch, None)[0]
            aud_val = vocoder.inference(preds.transpose(1, 2)).cpu()
            j = 0
            for audio, tr in zip(aud_val, val):
                logger.add_audio(audio, audio, tr, 'VALIDATION')
                mel_p = preds[j].detach()
                logger.add_spectrogram(mel_p, mel_p, tr, 'VALIDATION')
                j += 1
        model.train()

    i += 1
    if i == num_iters:
        break

torch.save(model.state_dict(), 'fastspeech')
