# 모듈 불러오기
from turtle import st
from torch.utils.data import Dataset
from pathlib import Path
import torchaudio
import pandas as pd
import matplotlib.pyplot as plt
import torch

def stereo_to_mono(signal:torch.Tensor):
    if signal.shape[0] == 2:
        signal = torch.mean(signal, dim=0, keepdim=True)
    return signal

class BirdSoundDataset(Dataset):
    def __init__(self, annotations:Path, audio_dir:Path, classes):
        super(BirdSoundDataset, self).__init__()
        self.classes = classes
        self.annotations = pd.read_csv(annotations)
        self.audio_dir = audio_dir
        
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sample_rate = torchaudio.load(audio_sample_path)

        stereo_to_mono(signal) # Stereo 채널 데이터를 mono로 변경

        # 32000 프레임(1초)별로 분할
        num_seg = int(signal.shape[1] / sample_rate) + 1                     # 1초 단위의 조각이 몇 개 있는지 (마지막 0.x초 포함)
        pad = num_seg * sample_rate - signal.shape[1]                        # 마지막 0.x초의 부족한 프레임은 0으로 padding
        pad = (0, pad)                                                       # padding size
        signal = torch.nn.functional.pad(signal, pad, 'constant', value=0)   # padding
        signal = signal.reshape((-1, 1, sample_rate))                        # (audio_length, channel, 32000) 크기의 텐서로 변경
        
        item = {
            'signal': signal,
            'labels': label
        }
        return item

    def _get_audio_sample_path(self, index):
        file_name = self.annotations.iloc[index, -1]
        full_path = self.audio_dir / file_name
        return full_path

    def _get_audio_sample_label(self, index):
        label = self.annotations['label'].iloc[index]
        return label

class ScoredBirdDataset(BirdSoundDataset):
    def __init__(self, annotations:Path, audio_dir:Path, classes:list):
        super(ScoredBirdDataset, self).__init__(annotations, audio_dir, classes)
        self.annotations = pd.read_csv(annotations)
        labels = self.annotations['label']
        labels = labels.apply(self.str_to_float_list)
        self.annotations.drop('label', axis='columns', inplace=True)
        self.annotations.insert(0, 'label', labels)

        labels = labels.apply(sum)
        scored = labels > 0

        self.annotations = self.annotations[scored].reset_index(drop=True)

    def __getitem__(self, index):
        item = super(ScoredBirdDataset, self).__getitem__(index)
        item['labels'] = self.annotations['label'].iloc[index]
        return item

    def str_to_float_list(self, input:str):
        out = list(map(float, input[1:-1].split(', ')))
        return out

class BirdSoundDataset_te(Dataset):
    def __init__(self, audio_dir:Path, classes):
        super(BirdSoundDataset_te, self).__init__()
        self.classes = classes
        self.audio_list = list(audio_dir.glob('*'))
        self.audio_list.sort()
        self.audio_dir = audio_dir

    def __len__(self):
        return len(self.audio_list)

    def __getitem__(self, index):
        audio_path = self.audio_list[index]
        signal, sample_rate = torchaudio.load(audio_path)

        # sample rate를 32000으로 변경
        if not sample_rate == 32000:
            transform = torchaudio.transforms.Resample(sample_rate, 32000)
            signal, sample_rate = transform(signal), 32000

        # Stereo 채널 데이터를 mono로 변경
        stereo_to_mono(signal)

        # 5초 x sample_rate 만큼 데이터 확장 후 1초별로 프레임 분할된 5초간의 텐서들의 집합으로 변경
        num_5sec_seg = signal.shape[1] / (sample_rate*5)
        if num_5sec_seg == int(num_5sec_seg):
            num_5sec_seg = int(num_5sec_seg)
        else:
            num_5sec_seg = int(num_5sec_seg) + 1

        pad = num_5sec_seg * (sample_rate*5) - signal.shape[1]
        pad = (0, pad)
        signal = torch.nn.functional.pad(signal, pad, 'constant', value=0)
        signal = signal.reshape((-1, 5, 1, sample_rate))
        return audio_path.name[:-4], signal

# [모델 정의]
# Spectrogram을 이용하지 않고 Audio tensor를 바로 사용

# 1. 1-dimension Convolution Neural Network
#    * Input: 매 초마다의 32000 프레임 (channel=1, 32000)
#    * Output: (N_{ch}, N_{dim}) 피처맵 vector
#    * T.x초의 audio 파일은 T+1 길이의 피처맵 시퀀스로 변환됨 (T+1, $N_{ch}$ * $N_{dim}$)
# 2. Recurrent Neural Network (LSTM)
#    * batch = 1 이므로, Conv1d의 output tensor를 (batch=1, T+1, $N_{ch}$ * $N_{dim}$) 크기로 unsqeeze
#    * Input: (1, T+1, N_{ch} * N_{dim})
#    * Output: (마지막 unit에서) (1, N_{dim_rnn})
# 3. Single-Layered Perceptrons
#    * Input: (1, N_{dim_rnn})
#    * Output: (1, N_{bird_classes})
#       * Output value: Sigmoid(SLP(input))
# 4. N-binary classification
#    * Score되는 각 bird의 클래스별로 있는지/없는지 binary classification
#        * output value >= 0.5 이면 true, 아니면 false
#    * result / 답지(label)를 활용하여 binary cross entropy 값을 구함

class BirdRecognition(torch.nn.Module):
    def __init__(self, classes):
        super(BirdRecognition, self).__init__()

        # Convolution network 부분
        self.conv1      = torch.nn.Conv1d(1, 4, kernel_size=5, stride=2, padding=2)
        self.conv2     = torch.nn.Conv1d(4, 8, kernel_size=5, stride=2, padding=2)
        self.conv3    = torch.nn.Conv1d(8, 16, kernel_size=5, stride=2, padding=2)
        self.maxpool = torch.nn.MaxPool1d(kernel_size=2, stride=2, padding=0)

        # LSTM 정의
        dim_in, hid_size = 16 * 500, 256 # Convolution layer의 output은 (seq, ch, feature map) => (seq, ch * feature map)으로 변경
        self.lstm = torch.nn.LSTM(input_size=dim_in, hidden_size=hid_size, num_layers=1, bias=False, batch_first=True)

        # Single Layer Perceptron 정의
        self.linear = torch.nn.Linear(hid_size, len(classes), bias=False)

    def forward(self, x):
        # Convolution layer
        x = self.conv1(x)       # 1 x 32000 -> 4 x 16000
        x = self.maxpool(x)     # 4 x 16000 -> 4 x 8000
        x = self.conv2(x)       # 4 x 8000  -> 8 x 4000
        x = self.maxpool(x)     # 8 x 4000  -> 8 x 2000
        x = self.conv3(x)       # 8 x 2000  -> 16 x 1000
        x = self.maxpool(x)     # 16 x 1000 -> 16 x 500
        
        # LSTM layer
        x = torch.flatten(x, 1)
        x = x.unsqueeze(0)
        _, (x, _) = self.lstm(x)
        x = torch.nn.ReLU()(x)

        # SLP layer
        x = self.linear(x)
        x = torch.nn.Sigmoid()(x)
        x = x.squeeze()

        return x

# 오디오 파일을 입력으로 받아 스펙트로그램을 그려주는 함수
def plot_specgram(waveform, sample_rate, title='Spectrogram', xlim=None):
    waveform = waveform.squeeze(0).numpy()
    sample_rate = int(sample_rate)
    num_channels, num_frames = waveform.shape

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1: axes = [axes]

    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1: axes[c].set_ylabel(f'Channel {c+1}')
        if xlim: axes[c].set_xlim(xlim)

    figure.suptitle(title)
    plt.show(block=False)