# U-Net vs DeepLabV3+ — Test Results

| Model | Dice left_lung | Dice right_lung | Dice infection | mean FG Dice | mean FG IoU | Params (M) |
|---|---|---|---|---|---|---|
| unet | 0.980 | 0.971 | 0.794 | 0.915 | 0.854 | 14.3 |
| deeplabv3plus | 0.978 | 0.960 | 0.635 | 0.858 | 0.782 | 12.3 |

## Best hyperparameters (Optuna)

| Model | lr | batch_size | encoder | aug_strength |
|---|---|---|---|---|
| unet | 0.0006647135865318024 | 16 | resnet18 | light |
| deeplabv3plus | 0.0001329291894316216 | 4 | resnet18 | none |
