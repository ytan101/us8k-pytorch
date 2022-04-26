import torch
import torchaudio
from torch.utils.data import DataLoader
from torch import nn

from urbansounddataset import UrbanSoundDataset
from cnn import CNNNetwork

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001
ANNOTATIONS_FILE = "UrbanSound8K/metadata/UrbanSound8K.csv"
AUDIO_DIR = "UrbanSound8K/audio"
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050


def create_data_loader(train_data, batch_size):
    train_data_loader = DataLoader(train_data, batch_size=batch_size)
    return train_data_loader


def train_single_epoch(model, data_loader, loss_fn, optimizer, device):
    for input, target in data_loader:
        input, target = input.to(device), target.to(device)

        # calculate loss
        prediction = model(input)
        loss = loss_fn(prediction, target)

        # backpropagate error and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"loss: {loss.item()}")


def train(model, data_loader, loss_fn, optimizer, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_single_epoch(model, data_loader, loss_fn, optimizer, device)
        print("-------------------")
    print("Finished training")


if __name__ == "__main__":

    # Checks if GPU (cuda) is available and use it. If not, use CPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using {device}")

    # Create callable object -> mel_spectrogram can be directly applied to the signal
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=64
    )

    usd = UrbanSoundDataset(
        ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES, device
    )

    train_data_loader = create_data_loader(usd, BATCH_SIZE)

    # construct model and assign to device
    cnn = CNNNetwork().to(device)
    print(cnn)

    # initialize loss function + optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE)

    # train model
    train(cnn, train_data_loader, loss_fn, optimizer, device, EPOCHS)

    # save model
    torch.save(cnn.state_dict(), "neuralnet.pth")
    print("Trained neural net saved at neuralnet.pth")
