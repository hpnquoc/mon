# How to choose: CPU vs GPU

Choosing the right device (CPU or GPU) for each step in a vision deep learning pipeline can greatly improve performance and resource efficiency.

---

## Task-by-Task Device Breakdown

| Task                                          | Best Device      | Reason                                                                 |
|-----------------------------------------------|------------------|------------------------------------------------------------------------|
| 🔄 Image Loading (PIL, OpenCV)                | **CPU**          | Lightweight I/O; no benefit from GPU                                   |
| 🧹 Preprocessing (resize, crop, etc.)         | **CPU**          | Simple transforms are efficient on CPU                                 |
| 📦 Data Augmentation (torchvision transforms) | **CPU**          | Basic augmentations are fast on CPU                                    |
| 📦 Data Augmentation (batch, heavy transforms)| **GPU** (optional)| If using `kornia` or `transforms.v2`, GPU helps with heavy ops         |
| 📤 DataLoader (batching, shuffling)           | **CPU** + workers| Use `num_workers > 0` for parallel loading                             |
| 🔄 ToTensor & Normalize                       | **CPU**          | Lightweight and usually part of preprocessing                          |
| 🚀 Model Forward Pass (inference/training)    | **GPU**          | Convolutions, attention, etc. are much faster on GPU                   |
| 🎯 Loss Calculation                           | **GPU**          | Small, but should be on same device as model output                    |
| 🔁 Backpropagation                            | **GPU**          | Compute-intensive, best on GPU                                         |
| 📉 Evaluation (model.eval())                  | **GPU** or CPU   | GPU for large batches, CPU is fine for small models/batches            |
| 💾 Saving/Loading Models                      | **CPU or GPU**   | File I/O, but model must be moved to correct device                    |
| 🖼️ Visualization (e.g., matplotlib)          | **CPU**          | Rendering and plotting work on CPU; move tensors if needed             |

## Best Practices

- Preprocessing & augmentations → **CPU**
- Model + training/inference → **GPU**
- Use `pin_memory=True` in DataLoader when training on GPU
- Move data to GPU **after** loading and preprocessing

## Efficient Workflow Summary

```text
[Disk (CPU)] → Load image
        ↓
  Preprocess & Augment (CPU)
        ↓
   DataLoader (CPU + workers)
        ↓
    .to(device='cuda')
        ↓
 Forward Pass + Loss + Backprop (GPU)
