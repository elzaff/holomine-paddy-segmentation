"""HoloMine paddy-field segmentation: full retraining pipeline in one file.

Four segmentation models are trained on the labelled train tiles only, their test probabilities are averaged
with equal weights, and the mask shape is cleaned:

    DINOv3 ViT-L/16 (satellite pretraining) + conv head
    UNet++ with EfficientNetV2-L encoder
    UPerNet with Swin-L backbone
    Mask2Former with Swin-L backbone

    mean probability -> blur sigma=2 -> threshold 0.35 -> closing 15 px
                     -> fill holes < 2000 px -> remove regions < 5000 px -> RLE

Supervised training only: no pseudo-labels, no external data, no test labels. Uses the competition files
<DATA>/train/images, <DATA>/train/masks, <DATA>/test_images, <DATA>/sample_submission.csv and public
pretrained backbones (timm, segmentation-models-pytorch, Hugging Face transformers).

    python paddy_repro.py --data /kaggle/input/competitions/holomine-paddy-field-segmentation-finals --work /kaggle/working

GPU training is not bit-reproducible; expect the score to move by a few 0.001 between runs.
"""
import argparse
import os
import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import timm
import torch
from torch import nn

MODELS = {
    "dinov3_sat": dict(arch="dinov3", encoder="vit_large_patch16_dinov3.sat493m"),
    "unetpp_effv2l": dict(arch="UnetPlusPlus", encoder="tu-tf_efficientnetv2_l"),
    "upernet_swinl": dict(arch="hf_upernet", encoder="openmmlab/upernet-swin-large"),
    # fp32: deformable attention and Hungarian matching are not fp16-safe; needs ~40 GB at batch 4
    "mask2former_swinl": dict(arch="mask2former", encoder="facebook/mask2former-swin-large-ade-semantic",
                              amp=False, big=True),
}
ENSEMBLE = list(MODELS)   # equal weights
EPOCHS = 60
CROP, BATCH, SEED = 768, 4, 2026
FOLDS = 5                 # one fold (25 tiles) is held out for a validation read-out
FINAL = dict(thr=0.35, sigma=2, close=15, hole=2000, speck=5000)
H = W = 1024


class Dino(nn.Module):
    """DINOv3 ViT-L/16 backbone + small conv head on four intermediate layers (1/16 -> 1/4 -> input size)."""

    def __init__(self, encoder, pretrained=True):
        super().__init__()
        self.vit = timm.create_model(encoder, pretrained=pretrained)
        n = len(self.vit.blocks)
        self.idx = [n // 4 - 1, n // 2 - 1, 3 * n // 4 - 1, n - 1]
        self.head = nn.Sequential(
            nn.Conv2d(4 * self.vit.embed_dim, 256, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 2, 2), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 2, 2), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 1, 3, padding=1),
        )

    def forward(self, x):  # H, W multiples of 16
        f = self.vit.forward_intermediates(x, indices=self.idx, intermediates_only=True)
        return nn.functional.interpolate(self.head(torch.cat(f, 1)), x.shape[2:], mode="bilinear")


class HFSeg(nn.Module):
    """Hugging Face UPerNet / Mask2Former behind one interface: model(x) -> paddy logits Bx1xHxW.
    Mask2Former only: model(x, y) -> its own set-prediction loss."""

    def __init__(self, net, m2f):
        super().__init__()
        self.net, self.m2f = net, m2f

    def forward(self, x, y=None):
        if not self.m2f:
            return self.net(pixel_values=x).logits
        if y is not None:  # one binary mask per class present in the crop
            cl = [torch.unique(m).long() for m in y[:, 0]]
            ml = [torch.stack([(m == k).float() for k in c]) for m, c in zip(y[:, 0], cl)]
            return self.net(pixel_values=x, mask_labels=ml, class_labels=cl).loss
        o = self.net(pixel_values=x)
        seg = torch.einsum("bqc,bqhw->bchw", o.class_queries_logits.softmax(-1)[..., :-1],
                           o.masks_queries_logits.sigmoid())
        seg = nn.functional.interpolate(seg, x.shape[2:], mode="bilinear")
        p = seg[:, 1:2] / (seg.sum(1, keepdim=True) + 1e-6)
        return torch.logit(p.clamp(1e-4, 1 - 1e-4))


def build(name, pretrained=True):
    """-> model, mean, std, optimizer parameter groups (pretrained backbones train at 0.1x the head rate)."""
    cfg = MODELS[name]
    imagenet = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    if cfg["arch"] == "dinov3":
        model = Dino(cfg["encoder"], pretrained)
        pc = model.vit.pretrained_cfg  # satellite statistics, not ImageNet
        return model, pc["mean"], pc["std"], [dict(params=model.head.parameters(), lr=1e-4),
                                              dict(params=model.vit.parameters(), lr=1e-5)]
    if cfg["arch"] in ("hf_upernet", "mask2former"):
        import transformers as T

        m2f = cfg["arch"] == "mask2former"
        cls = T.Mask2FormerForUniversalSegmentation if m2f else T.UperNetForSemanticSegmentation
        n = 2 if m2f else 1  # Mask2Former: background + paddy queries; UPerNet: one paddy logit
        net = (cls.from_pretrained(cfg["encoder"], num_labels=n, ignore_mismatched_sizes=True) if pretrained
               else cls(T.AutoConfig.from_pretrained(cfg["encoder"], num_labels=n)))
        bb = net.model.pixel_level_module.encoder if m2f else net.backbone
        skip = {id(p) for p in bb.parameters()}
        groups = [dict(params=[p for p in net.parameters() if id(p) not in skip], lr=1e-4),
                  dict(params=bb.parameters(), lr=1e-5)]
        return HFSeg(net, m2f), *imagenet, groups
    model = getattr(smp, cfg["arch"])(cfg["encoder"], encoder_weights="imagenet" if pretrained else None, classes=1)
    return model, *imagenet, [dict(params=model.parameters(), lr=1e-4)]


def read_rgb(p):
    return cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)


def holdout(n, fold=0):
    """Validation tile indices: every FOLDS-th tile of a fixed random permutation."""
    return sorted(np.random.RandomState(0).permutation(n)[fold::FOLDS].tolist())


def tile_groups(imgs):
    """Analysis helper (not used in training). Tiles were cut with a 512 px stride, so neighbours share half
    their area; tiles sharing a 512x512 quadrant get the same group id. Found from pixels only."""
    desc, key = [], []
    for i, im in enumerate(imgs):
        g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        for r in (0, 1):
            for c in (0, 1):
                q = g[r * 512:(r + 1) * 512, c * 512:(c + 1) * 512]
                if (q == 0).mean() <= 0.5:  # skip nodata quadrants, they all look alike
                    desc.append(cv2.resize(q, (24, 24), interpolation=cv2.INTER_AREA).astype(np.float32).ravel())
                    key.append(i)
    d = np.stack(desc)
    d2 = ((d ** 2).sum(1)[:, None] + (d ** 2).sum(1)[None] - 2 * d @ d.T) / d.shape[1]
    parent = list(range(len(imgs)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in zip(*np.where(d2 < 9)):  # rms < 3 grey levels: overlaps are resampled, not bit-identical
        parent[find(key[a])] = find(key[b])
    return [find(i) for i in range(len(imgs))]


def make_aug():
    """Training augmentation: random scale crop (0.25-1.0 of the tile) to 768 px, flips, 90-degree turns, mild colour."""
    return A.Compose([
        A.RandomResizedCrop(size=(CROP, CROP), scale=(0.25, 1.0), ratio=(0.75, 1.33)),
        A.HorizontalFlip(), A.VerticalFlip(), A.RandomRotate90(),
        A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
        A.HueSaturationValue(10, 15, 10, p=0.5),
    ])


def train_predict(name, data, work, epochs=EPOCHS, smoke=False):
    """Train one model, save its test probabilities to <work>/probs/<name>.npy (uint8, 255 = certain paddy)."""
    data, work = Path(data), Path(work)
    out = work / "probs" / f"{name}.npy"
    if out.exists():  # resume
        return str(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    random.seed(SEED), np.random.seed(SEED), torch.manual_seed(SEED)
    dev = "cuda"

    ids = sorted(p.stem for p in (data / "train/images").glob("*.png"))
    test_ids = pd.read_csv(data / "sample_submission.csv").ImageId.tolist()
    if smoke:
        ids, test_ids, epochs = ids[:12], test_ids[:4], 1
    imgs = [read_rgb(data / f"train/images/{i}.png") for i in ids]
    masks = [(cv2.imread(str(data / f"train/masks/{i}.png"), 0) > 127).astype(np.float32) for i in ids]
    val = holdout(len(ids))
    trn = [i for i in range(len(ids)) if i not in set(val)]
    aug = make_aug()

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(trn) * 4  # four random crops per tile per epoch

        def __getitem__(self, k):
            i = trn[k % len(trn)]
            a = aug(image=imgs[i], mask=masks[i])
            return torch.from_numpy(a["image"]).permute(2, 0, 1), torch.from_numpy(a["mask"])[None]

    dl = torch.utils.data.DataLoader(DS(), BATCH, shuffle=True, drop_last=True,
                                     num_workers=min(8, os.cpu_count() or 1))
    cfg = MODELS[name]
    amp, own_loss = cfg.get("amp", True), cfg["arch"] == "mask2former"
    model, mean, std, groups = build(name)
    model.to(dev)
    mean = torch.tensor(mean, device=dev).view(1, 3, 1, 1) * 255
    std = torch.tensor(std, device=dev).view(1, 3, 1, 1) * 255

    def norm(x):
        return (x.to(dev).float() - mean) / std

    opt = torch.optim.AdamW(groups, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups],
                                                total_steps=epochs * len(dl), pct_start=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    bce, dice = nn.BCEWithLogitsLoss(), smp.losses.DiceLoss("binary")  # Dice over the batch, like the micro-IoU metric
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for x, y in dl:
            y = y.to(dev)
            with torch.autocast("cuda", enabled=amp):
                o = model(norm(x), y) if own_loss else model(norm(x))
            o = o.float()
            loss = o if own_loss else bce(o, y) + dice(o, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
        print(f"{name} epoch {ep + 1}/{epochs} loss {tot / len(dl):.4f}", flush=True)

    model.eval()

    @torch.no_grad()
    def predict(img):
        """HxWx3 uint8 -> HxW uint8 probability, 4-flip test-time augmentation."""
        x = norm(torch.from_numpy(img).permute(2, 0, 1)[None])
        p = 0
        for d in ([], [2], [3], [2, 3]):
            with torch.autocast("cuda", enabled=amp):
                o = model(x.flip(d))
            p = p + o.float().sigmoid().flip(d)
        p = (p[0, 0] / 4).cpu().numpy()
        p[(img == 0).all(2)] = 0  # black nodata border is never paddy
        return np.round(p * 255).astype(np.uint8)

    inter = union = 0
    for i in val:
        p, g = predict(imgs[i]) > 127, masks[i] > 0
        inter, union = inter + (p & g).sum(), union + (p | g).sum()
    print(f"{name} validation micro-IoU on {len(val)} held-out train tiles: {inter / max(union, 1):.4f}", flush=True)
    np.save(out, np.stack([predict(read_rgb(data / f"test_images/{i}.png")) for i in test_ids]))
    torch.save({k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()},
               work / "probs" / f"{name}.pt")
    return str(out)


def blend(work, names=ENSEMBLE):
    """Equal-weight mean probability of the given models -> float32 (N,1024,1024) in 0..1."""
    acc = 0
    for n in names:
        acc = acc + np.load(Path(work) / "probs" / f"{n}.npy").astype(np.float32)
    return acc / (255 * len(names))


def postprocess(prob, thr, sigma, close, hole, speck):
    """One tile: blur -> threshold -> closing -> fill small holes -> drop small regions.
    Train masks are coarse polygons (about one small hole per tile); raw predictions have hundreds."""
    if sigma:
        prob = cv2.GaussianBlur(prob, (0, 0), sigma)
    a = (prob > thr).astype(np.uint8)
    if close:
        a = cv2.morphologyEx(a, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close, close)))
    _, lab, st, _ = cv2.connectedComponentsWithStats(1 - a, connectivity=4)
    k = np.where(st[:, 4] < hole)[0]
    a[np.isin(lab, k[k > 0])] = 1
    _, lab, st, _ = cv2.connectedComponentsWithStats(a, connectivity=4)
    k = np.where(st[:, 4] < speck)[0]
    a[np.isin(lab, k[k > 0])] = 0
    return a > 0


def rle_encode(m):
    """Airbus-style RLE: column-major, 1-based start/length pairs, '' for an empty mask."""
    p = np.concatenate([[0], m.flatten(order="F").astype(np.uint8), [0]])
    r = np.where(p[1:] != p[:-1])[0] + 1
    r[1::2] -= r[::2]
    return " ".join(map(str, r))


def make_submission(data, work, names=ENSEMBLE):
    ids = pd.read_csv(Path(data) / "sample_submission.csv").ImageId.tolist()
    rows = [rle_encode(postprocess(p, **FINAL)) for p in blend(work, names)]
    path = Path(work) / "submission.csv"
    pd.DataFrame({"ImageId": ids[:len(rows)], "EncodedPixels": rows}).to_csv(path, index=False)
    return str(path)


def run_all(data, work, smoke=False):
    """Whole pipeline on one GPU, sequentially. Every finished model is cached, so a rerun resumes."""
    for m in ENSEMBLE:
        train_predict(m, data, work, smoke=smoke)
    return make_submission(data, work)


def _selfcheck():
    m = np.zeros((H, W), bool)
    m[:2, 0] = m[0, 1] = True
    assert rle_encode(m) == "1 2 1025 1" and rle_encode(np.zeros((H, W), bool)) == ""
    p = np.zeros((H, W), np.float32)
    p[100:400, 100:400] = 1
    p[200:210, 200:210] = 0   # small hole -> filled
    p[900:905, 900:905] = 1   # speck -> removed
    out = postprocess(p, thr=0.5, sigma=0, close=0, hole=2000, speck=2000)
    assert out[205, 205] and not out[902, 902] and out.sum() == 300 * 300
    v = holdout(123)
    assert len(v) == 25 and v == holdout(123), "validation split must be deterministic"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/kaggle/input/competitions/holomine-paddy-field-segmentation-finals")
    ap.add_argument("--work", default="/kaggle/working")
    ap.add_argument("--smoke", action="store_true", help="1 epoch on 12 tiles: checks every code path, not the score")
    a = ap.parse_args()
    _selfcheck()
    print("submission:", run_all(a.data, a.work, a.smoke))
