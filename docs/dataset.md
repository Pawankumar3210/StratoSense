# Dataset: 38-Cloud

Mohajerani & Saeedi. 38 Landsat 8 scenes cut into 384×384 patches (8,400 training, 9,201 test) with
manually labelled pixel-level cloud masks. Thin cloud and haze are labelled as cloud.
Repository: https://github.com/SorourMo/38-Cloud-A-Cloud-Segmentation-Dataset · Kaggle: `sorour/38cloud-cloud-segmentation-in-satellite-images`.

**Licence and attribution.** The dataset repository is published under Apache License 2.0 (its `LICENSE` file).
Confirm the terms shown on the Kaggle page when downloading and keep this attribution. Do not redistribute the full
dataset; only a few attributed demonstration samples may be bundled (`samples/README.md`).
Cite both papers listed in the dataset README:
- S. Mohajerani and P. Saeedi, "Cloud-Net: An End-To-End Cloud Detection Algorithm for Landsat 8 Imagery," IGARSS 2019, pp. 1029–1032, doi:10.1109/IGARSS.2019.8898776.
- S. Mohajerani, T. A. Krammer and P. Saeedi, "A Cloud Detection Algorithm for Remote Sensing Images Using Fully Convolutional Neural Networks," MMSP 2018, pp. 1–5, doi:10.1109/MMSP.2018.8547095.

## Structure and conventions (verified from the official README, evaluation code and authors' loader)
| Item | Convention |
|---|---|
| Patch CSV | column `name`, e.g. `patch_192_10_by_12_LC08_L1TP_002053_20160520_20170324_01_T1` |
| Band file | `{split}_{band}/{band}_{name}.TIF` for red, green, blue, nir |
| Ground truth | `gt_{name}.TIF` in `train_gt` (README) or `train_gts` (authors' code); the loader accepts both |
| Patch name | `patch_{index}_{row}_by_{col}_{scene_id}`; `row`/`col` are 1-indexed tile positions |
| Image dtype / scale | 16-bit; authors divide by 65,535 |
| Label encoding | 0 = clear, 255 = cloud (any other values raise an error) |
| Nodata | the black margin around each scene is zero in all bands |
| Test labels | **complete-scene** masks only (`Entire_scene_gts/edited_corrected_gts_{scene_id}.TIF`) |

## Official test evaluation (what `ml/scene.py` implements)
Read from the official `evaluation.m`:
1. Tiles are **non-overlapping**; there is no blending. (The project spec mentions overlap handling: it does not apply to the official protocol. Overlapping tiles apply only to our own sliding-window inference on user uploads.)
2. A tile occupies rows `(row-1)*384 … row*384` and columns `(col-1)*384 … col*384` of the padded scene.
3. Padding is removed with a centred crop to the reference mask: `offset = floor((padded − gt) / 2)` per axis.
4. Metrics are computed per scene on the **cloud class** (precision, recall, specificity, Jaccard, accuracy) and averaged over the 20 test scenes.
5. The official script scores **all** pixels of the cropped scene, including nodata margin; it binarizes at 12/255 (the Cloud-Net threshold).

Our plan: report both (a) the **official protocol** (all pixels) for comparison with published numbers and (b) a **valid-pixel** variant
that excludes nodata, as the product requires. The decision threshold is selected on validation scenes only, never on test scenes.

## Deviations from the authors' example code
- The authors' training code splits validation randomly by patch (`val_ratio=0.2`), which puts neighbouring tiles of one scene on both sides (spatial leakage). We split by scene (`ml.dataset.split_scenes`, seeded SHA-256 ranking).

## To confirm on the real data (run `notebooks/01_data_exploration.ipynb`)
These were **not** verified because the dataset could not be downloaded in the build environment:
- [ ] Training and test patch counts equal 8,400 / 9,201 and all four bands plus GT exist for every name.
- [ ] CSV file names and column header as above; whether `*_nonempty.csv` ships with the Kaggle copy.
- [ ] GT directory spelling (`train_gt` vs `train_gts`) and label values (0/255).
- [ ] Reconstructed grid size vs reference mask size for each test scene (padding assumption; notebook §4).
- [ ] Whether any cloud labels fall on nodata pixels (notebook §5).
- [ ] Final decision on `use_nonempty_patches_only` (default true) after comparing the official list with the ≥80%-valid rule.
- [ ] Record the outcomes here, then compute `artifacts/normalization.json`.
