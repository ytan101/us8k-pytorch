import torch
import torchaudio

from tqdm import tqdm
from train import ANNOTATIONS_FILE, AUDIO_DIR, NUM_SAMPLES, SAMPLE_RATE
from urbansounddataset import UrbanSoundDataset
from cnn import CNNNetwork


class_mapping = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music",
]


def predict(model, input, target, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
    return predicted, expected


if __name__ == "__main__":
    # Load the model
    cnn = CNNNetwork()
    state_dict = torch.load("neuralnet.pth")
    cnn.load_state_dict(state_dict)

    # Load urban sound dataset
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=64
    )

    usd = UrbanSoundDataset(
        ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES, "cpu"
    )

    prediction_list = []
    expected_list = []

    for sample_index in tqdm(range(len(usd))):
        # get a sample from the USD for inference
        input, target = (
            usd[sample_index][0],
            usd[sample_index][1],
        )  # [[batch_size, num_channels, fr, time]
        input.unsqueeze_(0)  # introduce new dimension (batch_size) at index 0

        # make an inference
        predicted, expected = predict(cnn, input, target, class_mapping)
        prediction_list.append(predicted)
        expected_list.append(expected)

    correct_preds_counter = 0

    for prediction_index in tqdm(range(len(prediction_list))):
        if prediction_list[prediction_index] == expected_list[prediction_index]:
            correct_preds_counter += 1

    print(f"The accuracy is {correct_preds_counter/len(prediction_list)}")
