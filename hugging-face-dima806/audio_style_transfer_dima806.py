import soundfile as sf
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as F
from transformers import AutoModelForAudioClassification
import torchaudio

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        model = AutoModelForAudioClassification.from_pretrained("dima806/music_genres_classification")
        self.model = model.wav2vec2.feature_extractor.conv_layers
    
    def forward(self, x):
        features = []

        x = x.unsqueeze(0)
        for layer_num, layer in enumerate(self.model):
            x = layer.conv(x)
            features.append(x)
            x = layer.activation(x)
            if layer_num == 0:
                x = layer.layer_norm(x)
        
        return features

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

content_audio, sr = torchaudio.load("../audio/moonlight-sonata-3rd-movement-snippet.wav")
content_audio = content_audio.to(device)
style_audio, _ = torchaudio.load("../audio/reptilia-snippet.wav")
style_audio = F.resize(style_audio.unsqueeze(0), content_audio.shape).squeeze(0) # resize style to shape of content
style_audio = style_audio.to(device)
generated_audio = content_audio.clone().requires_grad_(True)

model = Model().to(device).eval()

# hyperparameters
total_steps = 4000
learning_rate = 0.1
alpha = 1
beta = 0.01
optimizer = optim.Adam([generated_audio], lr=learning_rate)

for step in range(total_steps):
    generated_features = model(generated_audio)
    content_features = model(content_audio)
    style_features = model(style_audio)

    style_loss = 0
    content_loss = 0

    for gen_feature, content_feature, style_feature in zip(
        generated_features, content_features, style_features
    ):
        batch_size, num_features, length = gen_feature.shape # batch_size will always be 1
        content_loss += torch.mean((gen_feature - content_feature) ** 2)
        
        G = gen_feature.view(num_features, length).mm(gen_feature.view(num_features, length).t())
        A = style_feature.view(num_features, length).mm(style_feature.view(num_features, length).t())
        G.div(num_features * length)
        A.div(num_features * length)

        style_loss += torch.mean((G - A) ** 2)

    total_loss = alpha * content_loss + beta * style_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step % 200 == 0:
        print("Loss after step {}: {}".format(step, total_loss.item()))
        sf.write("generated.wav", generated_audio.squeeze(0).numpy(force=True), sr)

sf.write("generated.wav", generated_audio.squeeze(0).numpy(force=True), sr)
