import copy
import os
import torch
import numpy as np
from torchvision.utils import save_image
from core.DownstreamEvaluator import DownstreamEvaluator
from PIL import Image
import logging

class PDownstreamEvaluator(DownstreamEvaluator):
    """
    Single downstream task:
      - iterate test_data_dict
      - for each sample: reconstruct, then save orig, rec, mask, residual
    """

    def __init__(self, name, model, device, test_data_dict, checkpoint_path):
        """
        name: downstream task name
        model: your thor‑ddmp model, must implement get_anomaly(x)
        device: torch.device
        test_data_dict: dict of {dataset_name: Dataset or DataLoader}
        checkpoint_path: path to model ckpt
        output_root: root folder to dump per-subject folders
        """
        super(PDownstreamEvaluator, self).__init__(name, model, device, test_data_dict, checkpoint_path)

    @staticmethod
    def _normalize_residual(residual: torch.Tensor):
        # residual: (H,W) or (1,H,W)
        r = residual.detach().cpu()
        r -= r.min()
        if r.max() > 0:
            r /= r.max()
        return (r * 255).byte().squeeze().numpy()

    def _save_case(self, subject_id, orig, rec, mask):
        """
        orig, rec, mask: torch.Tensor (C,H,W) or (1,H,W), in [0–1]
        """
        out_dir = os.path.join(self.image_path, f"subject_{subject_id:04d}")
        os.makedirs(out_dir, exist_ok=True)

        # 1) original & reconstruction
        save_image(orig, os.path.join(out_dir, "original.png"))
        save_image(rec, os.path.join(out_dir, "reconstructed.png"))

        # 2) mask → binary uint8
        m = (mask.detach().cpu().squeeze().numpy() > 0).astype(np.uint8) * 255
        Image.fromarray(m).save(os.path.join(out_dir, "mask.png"))

        # 3) residual = mean_abs(rec - orig) across channels
        res = torch.abs(rec - orig).mean(dim=0)
        res_np = self._normalize_residual(res)
        Image.fromarray(res_np).save(os.path.join(out_dir, "residual.png"))

    def start_task(self, global_model):
        """
        Run one pass over all test_data_dict entries and save outputs.
        """
        logging.info(f"################ OPMED Anomaly Detection #################")
        self.model.load_state_dict(global_model, strict=False)
        self.model.eval()
        
        counter = 1
        for dataset_key in self.test_data_dict.keys():
            dataset = self.test_data_dict[dataset_key]
            
            for idx, data in enumerate(dataset):
                
                if 'dict' in str(type(data)) and 'images' in data.keys():
                    data0 = data['images']
                else:
                    data0 = data[0]
                x = data0.to(self.device)
                
                masks = data[1].to(self.device)
                masks[masks > 0] = 1

                anomaly_map, anomaly_score, x_rec_dict = self.model.get_anomaly(copy.deepcopy(x))
                x_rec = x_rec_dict['x_rec'] if 'x_rec' in x_rec_dict.keys() else torch.zeros_like(x)
                x_rec = torch.clamp(x_rec, 0, 1)

                # per-sample (batch_size may be >1)
                for i in range(x.size(0)):
                    orig_i = x[i]
                    rec_i  = x_rec[i]
                    mask_i = masks[i]
                    self._save_case(counter, orig_i, rec_i, mask_i)
                    counter += 1

                    print(f"Saved {counter} cases under {self.image_path}")
