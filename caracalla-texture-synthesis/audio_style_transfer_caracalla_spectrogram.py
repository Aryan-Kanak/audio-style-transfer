# optimize generated spectrogram

import numpy as np
import librosa
import soundfile as sf
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        kernel_sizes = [3, 5, 7, 11, 15, 19, 23, 27]
        self.convs = nn.ModuleList([
            nn.Conv2d(1, 32, kernel_size=k, stride=1, padding=k//2)
            for k in kernel_sizes
        ])
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        outputs = [conv(x) for conv in self.convs]
        
        out = torch.cat(outputs, dim=0)
        
        return self.relu(out)


COMPRESSION_FACTOR = 1000
FRAME_SIZE = 512
HOP_SIZE = 256

# returns log spectrogram of an audio file
def extract_log_spectrogram(wav):
    X = librosa.stft(wav, hop_length=HOP_SIZE, win_length=FRAME_SIZE)
    Y = np.abs(X)
    S = np.log(1 + COMPRESSION_FACTOR * Y) / np.log(1 + COMPRESSION_FACTOR)

    return S

# given a log spectrogram of an audio file, returns the original audio
def log_spec_to_wav(spec):
    S = spec * np.log(1 + COMPRESSION_FACTOR)
    S = np.exp(S)
    S = (S - 1) / COMPRESSION_FACTOR
    S = librosa.griffinlim(S, hop_length=HOP_SIZE, win_length=FRAME_SIZE)

    return S

# converts output tensor to audio
def output_tensor_to_wav(output):
    spec = output.squeeze(0).numpy(force=True)
    return log_spec_to_wav(spec)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load content and style audio here
content_audio, sr = librosa.load("../audio/moonlight-sonata-3rd-movement-snippet.wav")
style_audio, _ = librosa.load("../audio/reptilia-snippet.wav")

content_spec = extract_log_spectrogram(content_audio)
style_spec = extract_log_spectrogram(style_audio)

content_transform = transforms.Compose(
    [
        transforms.ToTensor()
    ]
)

style_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(content_spec.shape) # reshape the style to be same size as content
    ]
)

content_tensor = content_transform(content_spec).to(device, dtype=torch.float32)
style_tensor = style_transform(style_spec).to(device, dtype=torch.float32)
generated_tensor = content_tensor.clone().requires_grad_(True)
model = Model().to(device).eval()

# hyperparameters
total_steps = 1000
learning_rate = 0.1
alpha = 1
beta = 0.01
optimizer = optim.Adam([generated_tensor], lr=learning_rate)

for step in range(total_steps):
    gen_features = model(generated_tensor)
    content_features = model(content_tensor)
    style_features = model(style_tensor)

    style_loss = 0
    content_loss = 0

    channels, height, width = gen_features.shape
    content_loss += torch.mean((gen_features - content_features) ** 2)

    G = torch.bmm(torch.permute(gen_features, (1, 0, 2)), torch.permute(gen_features, (1, 2, 0)))
    A = torch.bmm(torch.permute(style_features, (1, 0, 2)), torch.permute(style_features, (1, 2, 0)))

    style_loss += torch.mean((G - A) ** 2)

    total_loss = alpha * content_loss + beta * style_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step % 200 == 0:
        print("Loss after step {}: {}".format(step, total_loss.item()))
        generated_audio = output_tensor_to_wav(generated_tensor)
        sf.write("generated.wav", generated_audio, sr)
    
# save generated audio
generated_audio = output_tensor_to_wav(generated_tensor)
sf.write("generated.wav", generated_audio, sr)