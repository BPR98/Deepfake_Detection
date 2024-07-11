import os
from flask import Flask, request, render_template, redirect, url_for
import torch
from torch import nn
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'mp4'}

# Load your model here
class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(pretrained=True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(torch.mean(x_lstm, dim=1)))

model = Model(2)
model.load_state_dict(torch.load('C:/Users/Sujan kumar/OneDrive/Desktop/New folder/model_93_acc_100_frames_celeb_FF_data.pt'))
model.eval()

im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_frames(video_path, count=20):
    vidObj = cv2.VideoCapture(video_path)
    frames = []
    frame_count = int(vidObj.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, frame_count // count)
    success, image = vidObj.read()
    idx = 0

    while success and len(frames) < count:
        if idx % interval == 0:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            frames.append(transform(image))
        success, image = vidObj.read()
        idx += 1

    vidObj.release()
    if len(frames) < count:
        return None  # Not enough frames
    return torch.stack(frames)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            frames = extract_frames(file_path)
            if frames is None:
                return render_template('error.html', message="Insufficient frames in video.")
            frames = frames.unsqueeze(0)  # Add batch dimension
            with torch.no_grad():
                _, output = model(frames)
                _, prediction = torch.max(output, 1)
            label = 'FAKE' if prediction.item() == 0 else 'REAL'
            return render_template('result.html', label=label, filename=filename)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)


