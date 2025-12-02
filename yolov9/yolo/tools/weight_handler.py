import os
from dataclasses import dataclass
import sys
from pathlib import Path


import torch

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
print (project_root)

from yolo import create_model


@dataclass
class WeightHandler:
    model_name: str
    ckpt_path: str
    pt_path: str


    def export_pt(self, generated_ckpt_path: str):
        self._prepare_ckpt(generated_ckpt_path)
        self._convert_ckpt_to_pt()


    def _prepare_ckpt(self, generated_ckpt_path: str):
        checkpoint_path = os.path.join(generated_ckpt_path, f'{self.model_name}.ckpt')
        checkpoint = torch.load(checkpoint_path, map_location="cuda")  

        state_dict = checkpoint["state_dict"]

        new_state_dict = {}

        for key, value in state_dict.items():
            if key.startswith("ema.model."): 
                new_key = key.replace("ema.model.", "model.")
                print (new_key)
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
        os.makedirs(self.ckpt_path, exist_ok=True)
        cleaned_checkpoint_path = f"weights/ckpt/{self.model_name}.ckpt"
        torch.save(cleaned_checkpoint, cleaned_checkpoint_path)

        print(f"✅ Cleaned checkpoint saved to: {cleaned_checkpoint_path}")


    def _convert_ckpt_to_pt(self):
        ckpt_path = os.path.join(self.ckpt_path, f'{self.model_name}.ckpt')
        output_path = os.path.join(self.pt_path, f'{self.model_name}.pt')

        checkpoint = torch.load(ckpt_path, map_location="cuda",weights_only=True)

        # Extract only the model weights (assuming key is "state_dict")
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint  # If the checkpoint directly contains weights

        # Save as .pt file (weights only)
        torch.save(state_dict, output_path)
        print(f"Converted .ckpt to .pt: {output_path}")



