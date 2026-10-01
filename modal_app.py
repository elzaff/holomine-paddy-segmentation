"""Run the pipeline (paddy_repro.py) on Modal: the four models train in parallel, one GPU each.

    modal run --detach modal_app.py              # full pipeline, keeps running if this machine disconnects
    modal run --detach modal_app.py --smoke      # 1 epoch on 12 tiles: checks every code path
    modal volume get paddy-repro submission.csv .

Expects the competition files in ./data (train/images, train/masks, test_images, sample_submission.csv).
Other locations: set PADDY_DATA (and PADDY_TEST if the test images sit elsewhere).
"""
import os
from pathlib import Path

import modal

HERE = Path(__file__).parent
DATA = Path(os.environ.get("PADDY_DATA", HERE / "data"))
TEST = Path(os.environ.get("PADDY_TEST", DATA / "test_images"))

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.7.1", "torchvision==0.22.1", "segmentation-models-pytorch==0.5.0", "timm==1.0.20",
                 "albumentations==2.0.8", "opencv-python-headless==4.10.0.84", "huggingface_hub==0.36.0",
                 "transformers==4.57.6", "scipy", "pandas")  # scipy: Mask2Former Hungarian matcher
    .add_local_dir(DATA / "train", "/data/train")
    .add_local_dir(TEST, "/data/test_images")
    .add_local_file(DATA / "sample_submission.csv", "/data/sample_submission.csv")
    .add_local_file(HERE / "paddy_repro.py", "/root/paddy_repro.py")
)
app = modal.App("paddy-repro")
vol = modal.Volume.from_name("paddy-repro", create_if_missing=True)


def _fit(name, work, smoke):
    import paddy_repro as R

    vol.reload()
    R.train_predict(name, "/data", work, smoke=smoke)
    vol.commit()


@app.function(image=image, gpu=["A100-40GB", "A100-80GB"], timeout=4 * 3600, volumes={"/work": vol})
def fit(name: str, work: str, smoke: bool):
    _fit(name, work, smoke)


# Mask2Former trains in fp32 at batch 4 and does not fit in 40 GB
@app.function(image=image, gpu="A100-80GB", memory=32768, timeout=6 * 3600, volumes={"/work": vol})
def fit_big(name: str, work: str, smoke: bool):
    _fit(name, work, smoke)


@app.function(image=image, memory=8192, timeout=8 * 3600, volumes={"/work": vol})
def pipeline(smoke: bool = False):
    """Orchestrates server-side, so the run does not depend on the launching machine."""
    import paddy_repro as R

    R._selfcheck()
    work = "/work/smoke" if smoke else "/work"
    calls = [(fit_big if R.MODELS[m].get("big") else fit).spawn(m, work, smoke) for m in R.ENSEMBLE]
    for c in calls:
        c.get()
    vol.reload()
    path = R.make_submission("/data", work)
    vol.commit()
    return Path(path).read_text()


@app.local_entrypoint()
def main(smoke: bool = False):
    # spawn, never .remote(): a dropped connection on the launching machine must not cancel the run
    where = "smoke/submission.csv" if smoke else "submission.csv"
    print("started", pipeline.spawn(smoke).object_id, f"- when done: modal volume get paddy-repro {where} .")
