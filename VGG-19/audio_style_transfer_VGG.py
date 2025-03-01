import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models

# modified VGG-19 model that only outputs the feature maps
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.chosen_features = ['0', '5', '10', '19', '28']
        self.model = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features[:29]

    def forward(self, x):
        features = []

        for layer_num, layer in enumerate(self.model):
            x = layer(x)
            if str(layer_num) in self.chosen_features:
                features.append(x)
        
        return features

# returns spectrogram of an audio file
def extract_spectrogram(wav, sr):
    FRAME_SIZE = 2048
    HOP_SIZE = 512
    S = librosa.stft(wav, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
    Y = np.abs(S) ** 2
    Y_db = librosa.power_to_db(Y)
    # librosa.display.specshow(Y, sr=sr, hop_length=HOP_SIZE, x_axis="time",y_axis="log")
    # plt.colorbar(format="%+2.f")

    return Y_db

# given a spectrogram of an audio file, returns the original audio
def spec_to_wav(spec):
    return librosa.griffinlim(np.sqrt(librosa.db_to_power(spec)))

# converts output tensor to audio
def output_tensor_to_wav(output):
    return spec_to_wav(output.squeeze(0).mean(0).numpy(force=True))
    

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load content and style audio here
content_audio, sr = librosa.load("../audio/moonlight-sonata-3rd-movement-snippet.wav")
style_audio, _ = librosa.load("../audio/reptilia-snippet.wav")

content_spec = extract_spectrogram(content_audio, sr)
style_spec = extract_spectrogram(style_audio, sr)

content_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Lambda(lambda x : x.squeeze(0).repeat(3, 1, 1).unsqueeze(0))
    ]
)

style_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Lambda(lambda x : x.squeeze(0).repeat(3, 1, 1)),
        transforms.Resize(content_spec.shape), # reshape the style to be same size as content
        transforms.Lambda(lambda x : x.unsqueeze(0))
    ]
)

content_tensor = content_transform(content_spec).to(device)
style_tensor = style_transform(style_spec).to(device)
generated_tensor = content_tensor.clone().requires_grad_(True)
model = VGG().to(device).eval()

# hyperparameters
total_steps = 6000
learning_rate = 0.1
alpha = 1
beta = 0.01
optimizer = optim.Adam([generated_tensor], lr=learning_rate)

for step in range(total_steps):
    generated_features = model(generated_tensor)
    content_features = model(content_tensor)
    style_features = model(style_tensor)

    style_loss = 0
    content_loss = 0

    for gen_feature, content_feature, style_feature in zip(
        generated_features, content_features, style_features
    ):
        batch_size, channel, height, width = gen_feature.shape
        content_loss += torch.mean((gen_feature - content_feature) ** 2)

        G = gen_feature.view(channel, height * width).mm(gen_feature.view(channel, height * width).t())
        A = style_feature.view(channel, height * width).mm(style_feature.view(channel, height * width).t())

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
