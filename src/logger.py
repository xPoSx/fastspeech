import wandb


class WanDBWriter:
    def __init__(self, project='tts_project'):
        wandb.login(key='777734be0649971345886f08a6b84c9b9b190223')
        wandb.init(project=project)

    def add_metrics(self, metrics):
        wandb.log(metrics)

    def add_audio(self, pred, text, t):
        wandb.log({
            t + ' pred audio': wandb.Audio(pred.squeeze().numpy(), sample_rate=22050, caption=text)
        })

    def add_spectrogram(self, pred, text, t):
        wandb.log({
            t + ' pred spectrogram': wandb.Image(pred.squeeze().numpy(), caption=text)
        })
