import os
import torch

def prepare_ckpt(model_name: str):
    checkpoint_path = f"runs/train/v9-dev/checkpoints/{model_name}.ckpt" # ckptファイルのパスを入れる
    checkpoint = torch.load(checkpoint_path, map_location="cuda")  

    state_dict = checkpoint["state_dict"]

    new_state_dict = {}

    for key, value in state_dict.items():
        if key.startswith("ema.model."): 
            new_key = key.replace("ema.model.", "")
            new_state_dict[new_key] = value
        # Skip "model.model" weights

    # Create a new checkpoint dictionary with cleaned state_dict
    cleaned_checkpoint = {
        "epoch": checkpoint["epoch"],  # Keep epoch info
        "global_step": checkpoint["global_step"],  # Keep global step info
        "pytorch-lightning_version": checkpoint["pytorch-lightning_version"],  # Keep version
        "state_dict": new_state_dict,  # Use the filtered EMA weights
        "loops": checkpoint.get("loops", None),  # Keep training loops if available
        "callbacks": checkpoint.get("callbacks", None),  # Keep callbacks
        "optimizer_states": checkpoint.get("optimizer_states", None),  # Keep optimizer state
        "lr_schedulers": checkpoint.get("lr_schedulers", None),  # Keep LR scheduler
        "MixedPrecision": checkpoint.get("MixedPrecision", None),  # Keep precision settings
    }

    # Save the cleaned checkpoint
    os.makedirs('weights/ckpt', exist_ok=True)
    cleaned_checkpoint_path = f"weights/ckpt/{model_name}.ckpt"
    torch.save(cleaned_checkpoint, cleaned_checkpoint_path)

    print(f"✅ Cleaned checkpoint saved to: {cleaned_checkpoint_path}")



if __name__ == "__main__":
    prepare_ckpt('best')