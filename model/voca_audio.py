import torch.nn as nn
import torch
import torch.nn.functional as F


class AudioNet(nn.Module):
    def __init__(self, dim_aud=64, win_size=16):
        super(AudioNet, self).__init__()
        self.win_size = win_size
        self.dim_aud = dim_aud
        self.encoder_conv = nn.Sequential(  # n x 29 x 16
            nn.Conv1d(
                29, 32, kernel_size=3, stride=2, padding=1, bias=True
            ),  # n x 32 x 8
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(
                32, 32, kernel_size=3, stride=2, padding=1, bias=True
            ),  # n x 32 x 4
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(
                32, 64, kernel_size=3, stride=2, padding=1, bias=True
            ),  # n x 64 x 2
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(
                64, 64, kernel_size=3, stride=2, padding=1, bias=True
            ),  # n x 64 x 1
            nn.LeakyReLU(0.02, True),
        )
        self.encoder_fc1 = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU(0.02, True),
            nn.Linear(64, dim_aud),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.encoder_conv(x).squeeze(-1)
        x = self.encoder_fc1(x)

        return x


class AudioLocalNet(nn.Module):
    def __init__(self):
        super(AudioLocalNet, self).__init__()
        self.encoder_conv = nn.Sequential(  # n x 29 x 16
            nn.Conv1d(29, 32, kernel_size=4, stride=1, bias=True),  # n x 32 x 8
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(32, 128, kernel_size=4, stride=1, bias=True),  # n x 32 x 4
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(128, 256, kernel_size=4, stride=1, bias=True),  # n x 64 x 2
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(256, 1024, kernel_size=4, stride=1, bias=True),  # n x 64 x 1
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(1024, 5023, kernel_size=4, stride=1, bias=True),  # n x 64 x 1
            nn.LeakyReLU(0.02, True),
        )
        self.encoder_fc1 = nn.Sequential(
            nn.Linear(1, 4),
            nn.LeakyReLU(0.02, True),
            nn.Linear(4, 16),
            nn.LeakyReLU(0.02, True),
            nn.Linear(16, 64),
            nn.LeakyReLU(0.02, True),
            nn.Linear(64, 64),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.encoder_conv(x)
        x = self.encoder_fc1(x)
        return x


class Audio2Pose(nn.Module):
    def __init__(self, latent_dim=3):
        """
        :param latent_dim: size of the latent audio embedding
        :param model_name: name of the model, used to load and save the model
        """
        super().__init__()

        conv_len = 5
        self.convert_dimensions = torch.nn.Conv1d(80, 128, kernel_size=conv_len)
        self.weights_init(self.convert_dimensions)
        self.receptive_field = conv_len

        convs = []
        for i in range(6):
            dilation = 2 * (i % 3 + 1)
            self.receptive_field += (conv_len - 1) * dilation
            convs += [
                torch.nn.Conv1d(128, 128, kernel_size=conv_len, dilation=dilation)
            ]
            self.weights_init(convs[-1])
        self.convs = torch.nn.ModuleList(convs)
        self.code = nn.Linear(128, 64)

        self.apply(lambda x: self.weights_init(x))

        self.temporal = torch.nn.LSTM(
            input_size=128, hidden_size=128, num_layers=2, batch_first=True
        )

    def weights_init(self, m):
        if isinstance(m, torch.nn.Conv1d):
            torch.nn.init.xavier_uniform_(m.weight)
            try:
                torch.nn.init.constant_(m.bias, 0.01)
            except:
                pass

    def forward(self, audio: torch.Tensor):
        """
        :param audio: B x T x 16000 Tensor containing 1 sec of audio centered around the current time frame
        :return: code: B x T x latent_dim Tensor containing a latent audio code/embedding
        """
        # Convert to the right dimensionality
        x = F.leaky_relu(self.convert_dimensions(audio), 0.2)

        # Process stacks
        for conv in self.convs:
            x_ = F.leaky_relu(conv(x), 0.2)
            l = (x.shape[2] - x_.shape[2]) // 2
            x = (x[:, :, l:-l] + x_) / 2

        x = torch.mean(x, dim=-1)
        # x = torch.reshape(x, (-1, 30, 128))
        x = x.unsqueeze(0)
        x, _ = self.temporal(x)
        x = self.code(x)

        return x
