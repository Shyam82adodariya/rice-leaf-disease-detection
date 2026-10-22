# Data

The images are not included in this repo.

The project uses a rice leaf disease image set of 120 photos (about 40 each of bacterial leaf blight, brown spot and leaf smut; 119 were usable). It matches the public [Rice Leaf Diseases dataset on the UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/486/rice+leaf+diseases). My copy came as a nested zip (an outer zip containing one zip per class), so the notebook extracts it in two steps.

Place the extracted class folders in this layout:

```
data/raw/Data/
  Bacterial leaf blight/
  Brown spot/
  Leaf smut/
```

Please follow the dataset's terms of use.
