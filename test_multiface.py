import os
import pdb
import torch
from render.obj_renderer import Renderer
from model.multiface import Mymodel
import numpy as np
import torchaudio as ta
from scipy.io import wavfile
from utils.get_deepspeech import process_audio, audio_chunking, load_audio
import pickle
from utils.gussian_filter import apply_gussian
from pytorch3d.io import load_ply, save_ply, load_obj
from pytorch3d.structures import Meshes
from scipy.spatial.transform import Rotation as R

wav_path = './audio/multiface.wav'
template_path = './template/multiface.obj'

model = Mymodel()
model.load_state_dict(torch.load("./checkpoints/multiface.pth"))
model = model.cuda()
model.eval()

melspec = ta.transforms.MelSpectrogram(
    sample_rate=16000, n_fft=2048, win_length=800, hop_length=160, n_mels=80
)

neck_index = torch.load("./template/neck_index")
rigid_index = torch.load("./template/shoulder_index")

repeat_index = []
for i in range(len(rigid_index)):
    for j in range(len(neck_index)):
        if rigid_index[i] == neck_index[j]:
            repeat_index.append(j)
neck_index = np.delete(neck_index, repeat_index)

render = Renderer(template_path)

sample_rate, audio = wavfile.read(wav_path)
if audio.ndim != 1:
    print('Audio has multiple channels, only first channel is considered')
    audio = audio[:, 0]
proc_audio = process_audio('./ds_graph/output_graph.pb', audio, sample_rate, multiface=True)

pose_audio = audio_chunking(
        load_audio(wav_path),
        melspec,
        proc_audio.shape[0],
        frame_rate=30,
    ).cuda()

template_verts, f, _ = load_obj(template_path)

expression_offset = []
with torch.no_grad():
    proc_audio = torch.from_numpy(proc_audio).cuda().to(torch.float32)
    audio = model.batch_norm(proc_audio.permute(0, 2, 1)).permute(0, 2, 1)
    aud = model.audio_net(audio)
    aud = aud.unsqueeze(1).expand(-1, 6172, -1)
    audio_local = model.audio_localnet(audio)
    pose_feature = model.temporal(pose_audio).reshape(audio.shape[0], 1, -1)
    src = audio_local + aud
    src = torch.cat((src, pose_feature), dim=1)
    for j in range(proc_audio.shape[0]):
        output = model.transformer(src[j : j + 1])
        expression_offset.append(model.lin1(output))
    expression_offset = torch.stack(expression_offset, dim=0).squeeze(1)
    offset = expression_offset[:, :-1, :]
    pose = expression_offset[:, -1, :]
    pose = pose * 0.7
    pose = apply_gussian(pose).cpu().numpy()
    x = offset.cpu() + template_verts

out = []
for i in range(x.shape[0]):
    num = 0
    neck_num = 0
    rot = R.from_rotvec(pose[i]).as_matrix()
    rot1 = R.from_rotvec(pose[i] * 0.3).as_matrix()
    rot2 = R.from_rotvec(pose[i] * 0.6).as_matrix()

    rot_vert = np.zeros((6172, 3))
    for j in range(0, 6172):
        if j == rigid_index[num]:
            num = np.minimum(num + 1, 347)
            rot_vert[j] = np.dot(rot1, x[i, j])
        elif j == neck_index[neck_num]:
            neck_num = np.minimum(neck_num + 1, 570)
            rot_vert[j] = np.dot(rot2, x[i, j])
        else:
            rot_vert[j] = np.dot(rot, x[i, j])
    out.append(rot_vert)
out = torch.from_numpy(np.stack(out, axis=0)).to(torch.float32).cuda()


render.to_video(
    out,
    audio_file=wav_path,
    video_output="./mutiface.mp4",
    fps=30,
    batch_size=30,
)