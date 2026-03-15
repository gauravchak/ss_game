import torch
import os
from train import PolicyNetwork

def export_to_onnx(model_path="peg_solitaire_policy.pth", output_path="../public/peg_solitaire_policy.onnx"):
    """
    Loads the trained PyTorch Policy Network and exports it to ONNX format.
    Saves the output directly into the Next.js public/ folder so it can be served to the browser.
    """
    if not os.path.exists(model_path):
        print(f"Error: Could not find model weights at {model_path}")
        print("Run `python train.py` first to generate the model.")
        return

    print("Loading PyTorch model...")
    model = PolicyNetwork()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Create a dummy input tensor matching the expected input shape:
    # 1 batch, 1 channel, 7x7 board
    dummy_input = torch.randn(1, 1, 7, 7, requires_grad=True)

    # Ensure the destination directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Exporting to ONNX format at {output_path}...")
    torch.onnx.export(
        model,                      # Model being run
        dummy_input,                # Model input (or a tuple for multiple inputs)
        output_path,                # Where to save the model
        export_params=True,         # Store the trained parameter weights inside the model file
        opset_version=12,           # The ONNX version to export the model to
        do_constant_folding=True,   # Whether to execute constant folding for optimization
        input_names=['board_state'],# The model's input names
        output_names=['action_logits'], # The model's output names
        dynamic_axes={
            'board_state': {0: 'batch_size'},   # Variable length axes
            'action_logits': {0: 'batch_size'}
        }
    )
    
    print(f"Successfully exported ONNX model to {output_path}")

if __name__ == "__main__":
    export_to_onnx()
