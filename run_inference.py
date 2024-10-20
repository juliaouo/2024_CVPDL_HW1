import os
import torch
import json
from tqdm import tqdm
import argparse

from main import build_model_main
from util.slconfig import SLConfig
from datasets import build_dataset
from torch.utils.data import DataLoader
from util.utils import to_device
import util.misc as utils

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def parse_args():
    parser = argparse.ArgumentParser(description="Run DINO inference")
    parser.add_argument('--model_config', type=str, required=True, help='Path to the model config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--coco_path', type=str, required=True, help='Path to COCO dataset')
    parser.add_argument('--mode', type=str, required=True, help='train, val or test')
    parser.add_argument('--output', type=str, default="out/result.json", help='Output file name for results')
    parser.add_argument('--num_workers', type=int, default=10, help='Number of data loading workers')
    return parser.parse_args()

def main():
    args = parse_args()


    model_config_path = args.model_config
    model_checkpoint_path = args.checkpoint

    # Load model configuration
    config_args = SLConfig.fromfile(model_config_path)
    config_args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config_args.num_workers = args.num_workers

    # Build model
    model, criterion, postprocessors = build_model_main(config_args)
    checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    _ = model.eval()

    config_args.dataset_file = 'coco'
    config_args.coco_path = args.coco_path
    config_args.fix_size = False

    @torch.no_grad()
    def test(model, postprocessors, data_loader, device, args=None):
        model.to(device)
        model.eval()
        
        final_res = {}
        for samples, targets in tqdm(data_loader, desc="Processing Images", ncols=80):
            samples = samples.to(device)

            # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

            outputs = model(samples)

            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            results = postprocessors['bbox'](outputs, orig_target_sizes)
            # [scores: [100], labels: [100], boxes: [100, 4]] x B

            # if 'segm' in postprocessors.keys():
            #     target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            #     results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
            res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        
            for image_id, outputs in res.items():
                _scores = outputs['scores'].tolist()
                _labels = outputs['labels'].tolist()
                _boxes = outputs['boxes'].tolist()

                itemsdict = {
                        'scores': _scores,
                        'boxes': _boxes,
                        'labels': _labels,
                        }
        
                final_res[dataset_val.coco.dataset['images'][image_id]['file_name']] = itemsdict

        return final_res

    # Build dataset and dataloader
    dataset_val = build_dataset(image_set=args.mode, args=config_args)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    data_loader_val = DataLoader(dataset_val, 2, sampler=sampler_val,
                                drop_last=False, collate_fn=utils.collate_fn, num_workers=config_args.num_workers)

    # Run inference
    val_final_res = test(model, postprocessors, data_loader_val, torch.device('cuda'), args=config_args)

    # Save results to JSON
    with open(args.output, "w") as f:
        json.dump(val_final_res, f, indent=4)

if __name__ == "__main__":
    main()
