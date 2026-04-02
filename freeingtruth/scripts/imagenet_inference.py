import torch
import numpy as np
import onnxruntime as ort

def run_inference(example_path, session):

    # Load saved dataset sample
    sample = torch.load(example_path)

    image = sample['pixel_values']
    label = sample['labels']

    # add batch dimension
    image = image.unsqueeze(0)
    
    # Convert tensor to numpy
    image = image.cpu().numpy()

    # Run inference
    outputs = session.run(None, {"input": image})
    pred = np.argmax(outputs[0], axis=1)[0]

    print(f"\nExample: {example_path}")
    print(f"True label:", {label})
    print(f"Predicted label:", {pred})

def main():
    model_path = "Trained_ImagenetCNN.onnx"

    trained_example = "results/hw03_train_example.pt"
    val_example = "results/hw03_val_example.pt"

    # Load ONNX model
    session = ort.InferenceSession(model_path)

    # test examples
    run_inference(trained_example, session)
    run_inference(val_example, session)

if __name__ == "__main__":
    main()

# active PID 1393508