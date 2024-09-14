# Edge AI Keyword Detect
##### By Vedant Agarwal
 Compact CNNs on Audio for Keyword Spotting on Resource-Constrained Edge AI Devices

To use the Main script:
Make sure you have the following files in the same directory:

`MainScript.py`

`ryModels.py` (containing the model definition and label functions)

`ryM.pt` (the trained model weights)




Install the required libraries if you haven't already:
`pip install torch onnx onnxruntime numpy librosa`



Replace "`path_to_your_audio_file.wav`" with the actual path to your audio file.


Run the script. It will:

Load the PyTorch model

Convert it to ONNX format

Run inference on your audio file

Print the predicted label and top 5 probabilities


This script provides a complete workflow from loading the model to running inference. It's designed to be easy to use and modify for your specific needs.
