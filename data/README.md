# data/

Letakkan file kompetisi di sini (tidak di-commit):

```
data/
  train/images/train_000.png … train_122.png
  train/masks/train_000.png  … train_122.png     # 0 = latar, 255 = sawah
  test_images/test_000.png   … test_128.png
  sample_submission.csv
```

Lokasi lain: set `PADDY_DATA` (dan `PADDY_TEST` bila gambar test berada di folder terpisah).
Di Kaggle folder ini tidak dipakai; notebook membaca `/kaggle/input/competitions/holomine-paddy-field-segmentation-finals`.
