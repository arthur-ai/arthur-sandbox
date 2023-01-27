from data import load_and_preprocess_image, tensor_to_lime_array, batched_lime_array_to_tensor
from model import load_model

from torch import no_grad

model = load_model()


def load_image(image_path):
    loaded = load_and_preprocess_image(image_path)
    return tensor_to_lime_array(loaded)


def predict(batched_input):
    processed_input = batched_lime_array_to_tensor(batched_input)
    model.eval()
    with no_grad():
        return model.forward(processed_input).numpy()
