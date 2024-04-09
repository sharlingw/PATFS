import torch
import torch.nn as nn
from model.multi_audio import AudioNet, AudioLocalNet, Audio2Pose
from model.multiface_transformer import Transformer


class Mymodel(nn.Module):
    def __init__(self):
        super().__init__()

        self.audio_net = AudioNet()
        self.audio_localnet = AudioLocalNet()

        self.transformer = Transformer(
            d_model=64,
            dim_feedforward=64,
            nhead=1,
            dropout=0.1,
            num_encoder_layers=1,
            num_decoder_layers=0,
        )
        self.lin1 = nn.Linear(64, 3)


        self.batch_norm = nn.BatchNorm1d(num_features=29, eps=1e-5, momentum=0.9)
        self.temporal = Audio2Pose()

    def forward(self, audio, index=None, pose_audio=None):
        audio = self.batch_norm(audio.permute(0, 2, 1)).permute(0, 2, 1)
        pose_feature = self.temporal(pose_audio)[:, index : index + 2].reshape(
            audio.shape[0], 1, -1
        )

        aud = self.audio_net(audio)
        aud = aud.unsqueeze(1).expand(-1, 5023, -1)
        audio_local = self.audio_localnet(audio)

        src = audio_local + aud
        src = torch.cat((src, pose_feature), dim=1)

        output = self.transformer(src)
        output = self.lin1(output)

        return output
