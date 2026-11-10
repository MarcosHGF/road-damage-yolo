# Evaluation (test split)

Weights: `C:\Users\marco\Desktop\road-damage-yolo\runs\detect\rdd_yolo\weights\best.pt`

**mAP50 0.613**, mAP50-95 0.330, precision 0.647, recall 0.578

## Per class

| Class | AP50 | AP50-95 | Precision | Recall |
|---|---|---|---|---|
| longitudinal_crack | 0.535 | 0.282 | 0.629 | 0.500 |
| transverse_crack | 0.532 | 0.245 | 0.588 | 0.518 |
| alligator_crack | 0.647 | 0.340 | 0.693 | 0.572 |
| pothole | 0.737 | 0.451 | 0.678 | 0.721 |

## Per country

| Country | Images | mAP50 | mAP50-95 |
|---|---|---|---|
| China_Drone | 377 | 0.678 | 0.402 |
| China_MotorBike | 305 | 0.862 | 0.578 |
| Czech | 436 | 0.412 | 0.152 |
| India | 1166 | 0.331 | 0.158 |
| Japan | 1524 | 0.600 | 0.303 |
| Norway | 1197 | 0.210 | 0.087 |
| United_States | 753 | 0.661 | 0.363 |
