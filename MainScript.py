'''
Date: 2024-09-02
Author: Vedant Agarwal

'''
import torch
import torch.onnx
import onnxruntime as ort
import numpy as np
import librosa
import os

# Assuming ryModels.py is in the same directory
from ryModels import ryM3 as ryM, theLabels, index_to_label

# Function to preprocess audio
def preprocess_audio(file_path, sample_rate=16000, duration=1):
    audio, sr = librosa.load(file_path, sr=sample_rate, duration=duration)
    if len(audio) > sample_rate:
        audio = audio[:sample_rate]
    else:
        audio = np.pad(audio, (0, max(0, sample_rate - len(audio))))
    audio = audio.reshape(1, 1, -1).astype(np.float32)
    return audio

# Load PyTorch model
model = ryM(in_chs=1, out_cls=35)
model.load_state_dict(torch.load('ryM.pt', map_location=torch.device('cpu')))
model.eval()

# Convert to ONNX
dummy_input = torch.randn(1, 1, 16000)
onnx_path = "ryM.onnx"
torch.onnx.export(model, dummy_input, onnx_path, export_params=True, opset_version=10, do_constant_folding=True,
                  input_names=['input'], output_names=['output'], dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

# Load ONNX model
ort_session = ort.InferenceSession(onnx_path)

# Function to run inference
def run_inference(audio_path):
    # Preprocess audio
    input_data = preprocess_audio(audio_path)
    
    # Run inference
    ort_inputs = {ort_session.get_inputs()[0].name: input_data}
    ort_outs = ort_session.run(None, ort_inputs)
    
    # Process output
    output = ort_outs[0]
    predicted_class = np.argmax(output)
    predicted_label = index_to_label(predicted_class)
    
    return predicted_label, output[0]

# Main execution
if __name__ == "__main__":
    # Replace with the path to your audio file
    audio_file = "path_to_your_audio_file.wav"
    
    if not os.path.exists(audio_file):
        print(f"Audio file not found: {audio_file}")
    else:
        predicted_label, probabilities = run_inference(audio_file)
        print(f"Predicted label: {predicted_label}")
        print("Top 5 probabilities:")
        top5_indices = np.argsort(probabilities)[-5:][::-1]
        for i in top5_indices:
            print(f"{index_to_label(i)}: {probabilities[i]:.4f}")
