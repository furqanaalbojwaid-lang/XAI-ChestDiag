"""Generate the self-contained Kaggle notebooks from the current source code.

    python scripts/build_kaggle_notebook.py
writes notebooks/kaggle_xai_chestdiag.ipynb (run 1) and notebooks/kaggle_run2.ipynb (run 2).
"""
import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.splitlines(keepends=True)}


INTRO = """# XAI-ChestDiag - real experiment on Kaggle (NIH ChestX-ray14)

**Before running:**
1. *Add Input* -> **NIH Chest X-rays** (`nih-chest-xrays/data`).
2. *Settings* -> Accelerator -> **GPU T4 x2**; Internet **On**.
3. Run 1 trains `densenet121` + `efficientnet_b4`. Run 2 trains `vgg16` + `inception_v3`:
   add the output of Run 1 as an input (*Add Input -> Your Work*). Run 2 then reuses
   Run 1's trained models and resized images, and produces the full 4-model comparison.
4. **Save Version -> Save & Run All (Commit)**.
"""

SANITY = '''# Sanity check (fails fast if anything is broken)
import sys, torch, pandas as pd
sys.path.insert(0, '.')
from src.models import build_model, gradcam_target_layer, count_parameters
from src.gradcam import GradCAM
from src.dataset import CXRDataset, Augmenter
from src.common import index_images
from src.pointing_game import load_bboxes
for m in MODELS:
    net = build_model(m, pretrained=True).cuda()
    x = torch.randn(2, 3, 224, 224).cuda()
    net(x).sum().backward()
    cam, p = GradCAM(net.eval(), gradcam_target_layer(net, m))(x[:1])
    print(f'{m:16s} params={count_parameters(net)/1e6:.2f}M cam={cam.shape} ok')
    del net; torch.cuda.empty_cache()
df = pd.read_csv(cfg['data']['labels_csv']).rename(columns={'Image Index': 'image'}).head(16)
df['label'] = (df['Finding Labels'] != 'No Finding').astype(int)
ds = CXRDataset(df, index_images(NIH), 224, True, Augmenter(cfg['augmentation']))
xb, yb = next(iter(torch.utils.data.DataLoader(ds, batch_size=8, num_workers=2)))
print('loader ok', xb.shape, yb.tolist())
print(load_bboxes(cfg['data']['bbox_csv']).head(2))
'''

CONFIG = """import glob, yaml, os, shutil
hits = glob.glob('/kaggle/input/**/Data_Entry_2017.csv', recursive=True)
assert hits, 'Add the NIH dataset (nih-chest-xrays/data) as notebook input'
NIH = os.path.dirname(sorted(hits, key=len)[0]); print('NIH dataset at', NIH)
bbox = glob.glob(os.path.join(NIH, '**/BBox_List_2017.csv'), recursive=True)[0]
# reuse resized images from a previous run if attached as input
prev = [d for d in glob.glob(f'/kaggle/input/**/images_{RESIZE}', recursive=True) if os.path.isdir(d)]
IMG = prev[0] if prev and len(os.listdir(prev[0])) == 112120 else f'/kaggle/working/images_{RESIZE}'
print('image cache:', IMG)
cfg = yaml.safe_load(open('configs/default.yaml'))
cfg['data'].update(image_root=IMG, labels_csv=os.path.join(NIH, 'Data_Entry_2017.csv'), bbox_csv=bbox,
                   split_dir='outputs/splits')
cfg['training']['num_workers'] = os.cpu_count()
cfg['models'] = ['densenet121', 'vgg16', 'inception_v3', 'efficientnet_b4']
yaml.safe_dump(cfg, open('configs/kaggle.yaml', 'w'), sort_keys=False)
print(yaml.safe_dump(cfg, sort_keys=False))"""

RESIZE_CELL = """if IMG.startswith('/kaggle/working'):
    !python -m src.resize_cache --src {NIH} --dst {IMG} --size {RESIZE} --workers {os.cpu_count()}
print(len(os.listdir(IMG)), 'images ready')"""

IMPORT_PREV = """for p in glob.glob('/kaggle/input/**/outputs/*/seed*/test_metrics.json', recursive=True):
    run_dir = os.path.dirname(p); model = run_dir.split('/')[-2]; seed = run_dir.split('/')[-1]
    dst = f'outputs/{model}/{seed}'
    if not os.path.exists(dst):
        shutil.copytree(run_dir, dst); print('imported', model, seed)"""

TRAIN = """for seed in SEEDS:
    for m in MODELS:
        !python -m src.train --config configs/kaggle.yaml --model {m} --seed {seed}
        !python -m src.evaluate --config configs/kaggle.yaml --model {m} --seed {seed}"""

FINAL = """!python -m src.throughput --config configs/kaggle.yaml --batch-sizes 1 64 --amp
!python -m src.compare --config configs/kaggle.yaml --seed {SEEDS[0]}
if os.path.exists(f'outputs/densenet121/seed{SEEDS[0]}/best.pt'):
    !python -m src.pointing_game --config configs/kaggle.yaml --model densenet121 --seed {SEEDS[0]}
    !python -m src.gradcam_figure --config configs/kaggle.yaml --model densenet121 --seed {SEEDS[0]} --findings Pneumonia Infiltration Effusion Consolidation
!python -m src.plots --config configs/kaggle.yaml --seed {SEEDS[0]}"""

PACKAGE = """!cd outputs && zip -qr ../results.zip . -x '*.pt'
!ls -la results.zip
!rm -rf /kaggle/working/images_*   # keep the saved output small (cache can be re-created)
from IPython.display import Image, display
for f in sorted(glob.glob('outputs/figures/*.png')): print(f); display(Image(f, width=600))"""


def build(models, out_name, keep_cache):
    cells = [md(INTRO),
             code(f"MODELS = {models!r}\nSEEDS = [42]\nRESIZE = 256"),
             code("!nvidia-smi\nimport torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"),
             md("## 1. Write the project code"),
             code("import os\nos.makedirs('/kaggle/working/XAI-ChestDiag/src', exist_ok=True)\n"
                  "os.makedirs('/kaggle/working/XAI-ChestDiag/configs', exist_ok=True)\n"
                  "os.chdir('/kaggle/working/XAI-ChestDiag')")]
    for f in sorted(glob.glob(str(ROOT / "src/*.py"))) + [str(ROOT / "configs/default.yaml")]:
        rel = Path(f).relative_to(ROOT).as_posix()
        body = open(f).read()
        if body.strip():
            cells.append(code(f"%%writefile {rel}\n" + body))
    cells += [code("open('src/__init__.py','w').close()\n!ls src"),
              md("## 2. Dataset, image cache, config"), code(CONFIG), code(SANITY), code(RESIZE_CELL),
              md("## 3. Import results of a previous run (if attached)"), code(IMPORT_PREV),
              md("## 4. Patient-level split (identical in every run: fixed seed)"),
              code("!python -m src.prepare_splits --config configs/kaggle.yaml"),
              md("## 5. Train + evaluate"), code(TRAIN),
              md("## 6. Throughput, comparison + DeLong, Pointing Game, figures"), code(FINAL),
              md("## 7. Package results"),
              code(PACKAGE if not keep_cache else PACKAGE.replace(
                  "!rm -rf /kaggle/working/images_*   # keep the saved output small (cache can be re-created)\n", ""))]
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    (ROOT / "notebooks").mkdir(exist_ok=True)
    json.dump(nb, open(ROOT / "notebooks" / out_name, "w"), indent=1)
    print("wrote", out_name, len(cells), "cells")


GATHER = """import zipfile
print('--- /kaggle/input tree (depth 4) ---')
for root, dirs, files in os.walk('/kaggle/input', followlinks=True):
    depth = root.count('/') - 2
    if depth <= 4:
        print(root, len(files), 'files', [f for f in files if not f.endswith('.png')][:8])
    if 'images' in os.path.basename(root) or depth > 6:
        dirs[:] = []
found = 0
for root, dirs, files in os.walk('/kaggle/input', followlinks=True):
    if os.path.basename(root).startswith('images'):
        dirs[:] = []; continue
    if 'test_metrics.json' in files:
        model, seed = root.split('/')[-2], root.split('/')[-1]
        dst = f'outputs/{model}/{seed}'
        if not os.path.exists(dst):
            shutil.copytree(root, dst); print('imported', model, seed); found += 1
    if 'throughput.json' in files and not os.path.exists('outputs/summary/throughput.json'):
        os.makedirs('outputs/summary', exist_ok=True); shutil.copy(os.path.join(root, 'throughput.json'), 'outputs/summary/')
    for f in files:
        if f.endswith('.zip'):
            z = zipfile.ZipFile(os.path.join(root, f))
            for n in z.namelist():
                if '/outputs/' in n and '/seed' in n and not n.endswith('/'):
                    rel = n[n.index('outputs/'):]
                    if not os.path.exists(rel):
                        os.makedirs(os.path.dirname(rel), exist_ok=True)
                        with z.open(n) as src, open(rel, 'wb') as dst: shutil.copyfileobj(src, dst)
                        print('extracted', rel); found += 1
print(found, 'items imported'); print(sorted(glob.glob('outputs/*/seed*/*')))"""

COMBINE_CONFIG = """import glob, yaml, os, shutil
hits = glob.glob('/kaggle/input/**/Data_Entry_2017.csv', recursive=True)
NIH = os.path.dirname(sorted(hits, key=len)[0]); print('NIH dataset at', NIH)
bbox = glob.glob(os.path.join(NIH, '**/BBox_List_2017.csv'), recursive=True)[0]
cfg = yaml.safe_load(open('configs/default.yaml'))
cfg['data'].update(image_root=NIH, labels_csv=os.path.join(NIH, 'Data_Entry_2017.csv'), bbox_csv=bbox,
                   split_dir='outputs/splits')
cfg['training']['num_workers'] = os.cpu_count()
cfg['models'] = ['densenet121', 'vgg16', 'inception_v3', 'efficientnet_b4']
yaml.safe_dump(cfg, open('configs/kaggle.yaml', 'w'), sort_keys=False)"""

COMBINE_FINAL = """!python -m src.prepare_splits --config configs/kaggle.yaml
!python -m src.compare --config configs/kaggle.yaml --seed 42
!python -m src.pointing_game --config configs/kaggle.yaml --model densenet121 --seed 42
!python -m src.gradcam_figure --config configs/kaggle.yaml --model densenet121 --seed 42 --findings Pneumonia Infiltration Effusion Consolidation
!python -m src.plots --config configs/kaggle.yaml --seed 42
!zip -qr results_final.zip outputs -x '*.pt'
!cp -r outputs /kaggle/working/final_outputs && find /kaggle/working/final_outputs -name '*.pt' -delete
from IPython.display import Image, display
for f in sorted(glob.glob('outputs/figures/*.png')): print(f); display(Image(f, width=600))"""


def build_combine(out_name="kaggle_combine.ipynb"):
    cells = [md("# XAI-ChestDiag - combine Run 1 + Run 2 (CPU is enough)\n\nInputs: NIH Chest X-rays, XAI-ChestDiag Run1, XAI-ChestDiag Run2.\n"),
             code("import os\nos.makedirs('/kaggle/working/XAI-ChestDiag/src', exist_ok=True)\n"
                  "os.makedirs('/kaggle/working/XAI-ChestDiag/configs', exist_ok=True)\n"
                  "os.chdir('/kaggle/working/XAI-ChestDiag')")]
    for f in sorted(glob.glob(str(ROOT / "src/*.py"))) + [str(ROOT / "configs/default.yaml")]:
        rel = Path(f).relative_to(ROOT).as_posix()
        body = open(f).read()
        if body.strip():
            cells.append(code(f"%%writefile {rel}\n" + body))
    cells += [code("open('src/__init__.py','w').close()"), code(COMBINE_CONFIG), code(GATHER), code(COMBINE_FINAL)]
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    json.dump(nb, open(ROOT / "notebooks" / out_name, "w"), indent=1)
    print("wrote", out_name, len(cells), "cells")


ABL_SANITY = """import sys, torch
sys.path.insert(0, '.')
from src.proposed import XAIChestNet, multilabel_targets
from src.gradcam import GradCAM
for att, mt in [('none', True), ('cbam', True)]:
    net = XAIChestNet(pretrained=True, attention=att, multitask=mt).cuda()
    x = torch.randn(2, 3, 224, 224).cuda()
    b, a = net.forward_all(x); (b.sum() + a.sum()).backward()
    layer = net.attn if att == 'cbam' else net.features.denseblock4
    cam, p = GradCAM(net.eval(), layer)(x[:1])
    print(att, mt, b.shape, a.shape, cam.shape, 'ok')
print(multilabel_targets(['Effusion|Mass', 'No Finding']))"""


def write_code_cells():
    cells = []
    files = sorted(glob.glob(str(ROOT / "src/*.py"))) + [str(ROOT / "configs/default.yaml")] + \
        sorted(glob.glob(str(ROOT / "configs/ablation/*.yaml")))
    for f in files:
        rel = Path(f).relative_to(ROOT).as_posix()
        body = open(f).read()
        if body.strip():
            cells.append(code(f"%%writefile {rel}\n" + body))
    return cells


CACHE_FROM_INPUT = """# Reuse a pre-built image cache (zip in an attached input) instead of resizing 112k images again
import zipfile, time
t0 = time.time()
os.makedirs(IMG, exist_ok=True)
if len(os.listdir(IMG)) < 112120:
    zips = []
    for root, dirs, files in os.walk('/kaggle/input', followlinks=True):
        dirs[:] = [d for d in dirs if not d.startswith('images')]
        zips += [os.path.join(root, f) for f in files if f.endswith('.zip')]
    print('zips found:', zips)
    for z in zips:
        zf = zipfile.ZipFile(z)
        names = [n for n in zf.namelist() if f'images_{RESIZE}/' in n and n.endswith('.png')]
        print(z, len(names), 'cached images')
        if len(names) >= 112120:
            for i, n in enumerate(names):
                with zf.open(n) as src, open(os.path.join(IMG, os.path.basename(n)), 'wb') as dst:
                    shutil.copyfileobj(src, dst)
                if i % 20000 == 0: print(i, round(time.time() - t0), 's', flush=True)
            break
print(len(os.listdir(IMG)), 'images in cache', round(time.time() - t0), 's')"""


def build_cache(resize, out_name):
    cells = [md(f"# XAI-ChestDiag - build {resize}px image cache (CPU only, no GPU quota)\n\nInput: NIH Chest X-rays. Output: images_{resize}.zip\n"),
             code(f"RESIZE = {resize}"),
             code("import os\nos.makedirs('/kaggle/working/XAI-ChestDiag/src', exist_ok=True)\nos.chdir('/kaggle/working/XAI-ChestDiag')"),
             code("%%writefile src/resize_cache.py\n" + open(ROOT / "src/resize_cache.py").read()),
             code("open('src/__init__.py','w').close()\nimport glob\n"
                  "hits = glob.glob('/kaggle/input/**/Data_Entry_2017.csv', recursive=True)\n"
                  "NIH = os.path.dirname(sorted(hits, key=len)[0]); IMG = f'/tmp/images_{RESIZE}'\nprint(NIH)"),
             code("!python -m src.resize_cache --src {NIH} --dst {IMG} --size {RESIZE} --workers {os.cpu_count()}"),
             code("import zipfile\nn = 0\nwith zipfile.ZipFile(f'/kaggle/working/images_{RESIZE}.zip', 'w', zipfile.ZIP_STORED) as z:\n"
                  "    for f in sorted(os.listdir(IMG)):\n        z.write(os.path.join(IMG, f), f'images_{RESIZE}/{f}'); n += 1\n"
                  "print(n, 'images zipped')\n!ls -la /kaggle/working")]
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    json.dump(nb, open(ROOT / "notebooks" / out_name, "w"), indent=1)
    print("wrote", out_name, len(cells), "cells")


def build_ablation(variants, resize, out_name, tta_for=()):
    train = "for v in VARIANTS:\n" \
            "    !python -m src.train_proposed --config configs/kaggle.yaml --variant configs/ablation/{v}.yaml --seed 42\n" \
            "    !python -m src.evaluate --config configs/kaggle.yaml --model {v} --seed 42\n" \
            "    if v in TTA_FOR:\n" \
            "        !python -m src.evaluate --config configs/kaggle.yaml --model {v} --seed 42 --tta"
    cells = [md("# XAI-ChestDiag - ablation of the proposed model\n\nInputs: NIH Chest X-rays. Accelerator: GPU T4 x2.\n"),
             code(f"VARIANTS = {variants!r}\nTTA_FOR = {list(tta_for)!r}\nRESIZE = {resize}\nMODELS = []\nSEEDS = [42]"),
             code("!nvidia-smi"),
             code("import os\nos.makedirs('/kaggle/working/XAI-ChestDiag/src', exist_ok=True)\n"
                  "os.makedirs('/kaggle/working/XAI-ChestDiag/configs/ablation', exist_ok=True)\n"
                  "os.chdir('/kaggle/working/XAI-ChestDiag')")]
    cells += write_code_cells()
    cells += [code("open('src/__init__.py','w').close()"), code(CONFIG.replace("prev = [", "prev = [] and [")),
              code(ABL_SANITY), code(CACHE_FROM_INPUT), code(RESIZE_CELL),
              code("!python -m src.prepare_splits --config configs/kaggle.yaml"),
              code(train),
              code("!cd outputs && zip -qr ../results.zip . -x '*.pt'\n!rm -rf /kaggle/working/images_*\n!ls -la outputs/*/seed42")]
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    json.dump(nb, open(ROOT / "notebooks" / out_name, "w"), indent=1)
    print("wrote", out_name, len(cells), "cells")


FINAL_STEPS = """!python -m src.prepare_splits --config configs/kaggle.yaml
!python -m src.final_report --config configs/kaggle.yaml --final a4_full_384
for run in ['a4_full_384', 'densenet121']:
    if os.path.exists(f'outputs/{run}/seed42/best.pt'):
        !python -m src.pointing_game --config configs/kaggle.yaml --model {run} --seed 42
!python -m src.gradcam_figure --config configs/kaggle.yaml --model a4_full_384 --seed 42 --findings Pneumonia Infiltration Effusion Consolidation
!cp -r outputs /kaggle/working/final_outputs && find /kaggle/working/final_outputs -name '*.pt' -delete
!cd /kaggle/working && zip -qr final_results.zip final_outputs
from IPython.display import Image, display
for f in sorted(glob.glob('outputs/figures/*.png')): print(f); display(Image(f, width=600))"""


def build_final(out_name="kaggle_final.ipynb"):
    cells = [md("# XAI-ChestDiag - final report (CPU)\n\nInputs: NIH Chest X-rays, Run1, Run2, Ablation A, Ablation B.\n"),
             code("import os\nos.makedirs('/kaggle/working/XAI-ChestDiag/src', exist_ok=True)\n"
                  "os.makedirs('/kaggle/working/XAI-ChestDiag/configs/ablation', exist_ok=True)\n"
                  "os.chdir('/kaggle/working/XAI-ChestDiag')")]
    cells += write_code_cells()
    cells += [code("open('src/__init__.py','w').close()"), code(COMBINE_CONFIG), code(GATHER), code(FINAL_STEPS)]
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    json.dump(nb, open(ROOT / "notebooks" / out_name, "w"), indent=1)
    print("wrote", out_name, len(cells), "cells")


if __name__ == "__main__":
    build(["densenet121", "efficientnet_b4"], "kaggle_xai_chestdiag.ipynb", keep_cache=True)
    build(["vgg16", "inception_v3"], "kaggle_run2.ipynb", keep_cache=False)
    build_combine()
    build_ablation(["a1_multitask", "a2_mt_cbam", "a3_mt_cbam_modern"], 256, "kaggle_ablation_A.ipynb")
    build_ablation(["a4_full_384"], 384, "kaggle_ablation_B.ipynb", tta_for=["a4_full_384"])
    build_final()
    build_cache(384, "kaggle_cache_384.ipynb")
