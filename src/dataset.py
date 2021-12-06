import torch
import torchaudio
from src.aligner import FastSpeechAligner

# Took from https://keithito.com/LJ-Speech-Dataset/


def format_text(text):
    abbrs = {'Mr.': 'Mister', 'Mrs.' : 'Misess', 'Dr.': 'Doctor', 'No.': 'Number', 'St.': 'Saint',
             'Co.': 'Company', 'Jr.': 'Junior', 'Maj.': 'Major', 'Gen.': 'General', 'Drs.': 'Doctors',
             'Rev.': 'Reverend', 'Lt.': 'Lieutenant', 'Hon.': 'Honorable', 'Sgt.': 'Sergeant',
             'Capt.': 'Captain', 'Esq.': 'Esquire', 'Ltd.': 'Limited', 'Col.': 'Colonel', 'Ft.': 'Fort'}
    for k in abbrs:
        text = text.replace(k, abbrs[k])
    return text


class LJSpeechDataset(torchaudio.datasets.LJSPEECH):

    def __init__(self, root):
        super().__init__(root=root)
        self._tokenizer = torchaudio.pipelines.TACOTRON2_GRIFFINLIM_CHAR_LJSPEECH.get_text_processor()
        self._aligner = FastSpeechAligner()

    def __getitem__(self, index: int):
        waveform, _, _, transcript = super().__getitem__(index)
        waveforn_length = torch.tensor([waveform.shape[-1]]).int()

        transcript = format_text(transcript)
        tokens, token_lengths = self._tokenizer(transcript)

        return waveform, waveforn_length, transcript, tokens, token_lengths, self._aligner(index)

    def decode(self, tokens, lengths):
        result = []
        for tokens_, length in zip(tokens, lengths):
            text = "".join([
                self._tokenizer.tokens[token]
                for token in tokens_[:length]
            ])
            result.append(text)
        return result
