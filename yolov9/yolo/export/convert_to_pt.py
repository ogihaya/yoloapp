import torch

def convert_ckpt_to_pt(ckpt_path: str, output_path: str):
    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location="cuda",weights_only=True)

    # Extract only the model weights (assuming key is "state_dict")
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint  # If the checkpoint directly contains weights

    # Save as .pt file (weights only)
    torch.save(state_dict, output_path)
    print(f"Converted .ckpt to .pt: {output_path}")

if __name__ == "__main__":
    ckpt_path = 'weights/ckpt/test-model.ckpt'
    output_path = 'weights/test-model.pt'
    
    # Example usage
    convert_ckpt_to_pt(ckpt_path, output_path)