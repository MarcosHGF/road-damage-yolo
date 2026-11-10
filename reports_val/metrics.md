# Evaluation (val split)

Weights: `C:\Users\marco\Desktop\road-damage-yolo\runs\detect\rdd_yolo\weights\best.pt`

**mAP50 0.619**, mAP50-95 0.334, precision 0.651, recall 0.572

## Per class

| Class | AP50 | AP50-95 | Precision | Recall |
|---|---|---|---|---|
| longitudinal_crack | 0.525 | 0.280 | 0.631 | 0.478 |
| transverse_crack | 0.529 | 0.247 | 0.603 | 0.493 |
| alligator_crack | 0.657 | 0.345 | 0.686 | 0.583 |
| pothole | 0.765 | 0.464 | 0.682 | 0.734 |

## Per country

| Country | Images | mAP50 | mAP50-95 |
|---|---|---|---|
| China_Drone | 349 | 0.638 | 0.382 |
| China_MotorBike | 296 | 0.872 | 0.588 |
| Czech | 431 | 0.371 | 0.136 |
| India | 1172 | 0.390 | 0.194 |
| Japan | 1550 | 0.614 | 0.313 |
| Norway | 1256 | 0.205 | 0.093 |
| United_States | 704 | 0.696 | 0.388 |
