# 2024_CVPDL_HW1
2024 Computer Vision Practice with Deep Learning HW1
> name: 林怡萱

> id: R13944021

## Environment
- OS: Ubuntu 22.04
- GPU: NVIDIA GeForce RTX 3090
- Python: 3.9.20
- PyTorch: 2.6.0
- CUDA: 11.8

## Installation

   1. Clone this repo
   ```sh
   git clone https://github.com/juliaouo/2024_CVPDL_HW1.git
   cd 2024_CVPDL_HW1
   ```

   2. Install Pytorch and torchvision

   Follow the instruction on https://pytorch.org/get-started/locally/.
   ```sh
   # an example:
   conda install -c pytorch pytorch torchvision
   ```

   3. Install other needed packages
   ```sh
   pip install -r requirements.txt
   ```

   4. Compile CUDA operators
   ```sh
   cd models/dino/ops
   python setup.py build install
   # Unit test (should see all checking is True)
   python test.py
   cd ../../..
   ```


## Data

After downloading the dataset, covert the dataset to COCO annotation format

Modify `data2coco.py`:
```
mode = "valid" # train, valid, test
input_dir = "path/to/your/input_dir"
output_dir = "path/to/your/output_dir"
```

Then run:
```
python data2coco.py
```

The folder structure and names should be like this:
```
COCODIR/
  ├── train/
  ├── val/
  ├── test/
  └── annotations/
  	├── instances_train.json
  	└── instances_val.json
    └── instances_test.json
```



## Run

### Fine-tune

1. Download pretrianed models

Download the DINO model checkpoint ["checkpoint0029_4scale_swin.pth"](https://drive.google.com/file/d/1CrzFP0RycSC24KKmF5k0libLRJgpX9x0/view?usp=drive_link) and the Swin-L backbone ["swin_large_patch4_window12_384_22k.pth"](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth)

2. Training/Fine-tuning a DINO on custom dataset

To train a DINO on a custom dataset **from scratch**, you need to tune two parameters in a config file:
- Tuning the `num_classes` to the number of classes to detect in your dataset.
- Tuning the parameter `dn_labebook_size` to ensure that `dn_labebook_size >= num_classes + 1`

In `/config/DINO/DINO_4scale_swin_custom.py`:
```
num_classes=17
dn_labebook_size=18
```

Run to train:
```
bash scripts/DINO_train_swin.sh /path/to/your/custom/data /path/to/backbone 0 \
--pretrain_model_path /path/to/checkpoint0029_4scale_swin.pth \
--finetune_ignore label_enc.weight class_embed
```
The `0` refers to setting CUDA_VISIBLE_DEVICES=0, which can be changed


### Test

Modify `run_inference.py` to use specific GPU:
```
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```

And run:
```
python run_inference.py --model_config /path/to/your/config/DINO/DINO_4scale_swin_custom.py \
--checkpoint /path/to/your/checkpoint_best_regular.pth --coco_path /project/n/julialin/hw1/data/coco_format --mode test
```
`--mode` can use "train", "val" or "test"
The result will be saved in the `out/result.json`. You can use `--output` to change the output path.

The output data structure is as follows:
```
{
    "file_name": {
        "scores": [...],
        "boxes": [[...],[...]],
        "labels": [...]
    },
    "file_name": {
        "scores": [...],
        "boxes": [[...],[...]],
        "labels": [...]
    }
}
```

Then you can run filter.py to filter the bounding boxes where the score is greater than the specified threshold
```
python filter.py
```
`--threshold`: default=0.3
`--input_file`: default="out/result.json"
`--output_file`: default="out/result_thresh_0.3.json"

And the final result structure is as follows:
```
{
    "file_name": {
        "boxes": [[...],[...]],
        "labels": [...]
    },
    "file_name": {
        "boxes": [[...],[...]],
        "labels": [...]
    }
}
```

Run to calculate mAP(50-95):

```
python path/to/eval.py path/to/result path/to/valid_target.json
```