# %%
# https://colab.research.google.com/drive/1U9n40rwdnn73bdWruONA3hIs1-H3f74Q#scrollTo=4emxnFT4qHNK
from huggingface_hub import hf_hub_download
import tacoreader
import rasterio as rio
import matplotlib.pyplot as plt

tacoreader.__version__  # 0.5.2

# Select the flavour!
dataset = tacoreader.load("tacofoundation:cloudsen12-l1c")
sample_idx = 1000
s2_l1c = dataset.read(sample_idx).read(0)
s2_label = dataset.read(sample_idx).read(1)
# dataset_extra = tacoreader.load("tacofoundation:cloudsen12-extra")
# dataset = tacoreader.load("tacofoundation:cloudsen12-l2a")
# %%
# %%timeit -> Remote access 827 ms ± 43.9 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
# Read a sample
sample_idx = 3500
s2_l1c = dataset.read(sample_idx).read(0)
s2_label = dataset.read(sample_idx).read(1)

# Retrieve the S2 data
with rio.open(s2_l1c) as src, rio.open(s2_label) as dst:
    s2_l1c_data = src.read(
        [4, 3, 2], window=rio.windows.Window(0, 0, 512, 512))
    s2_label_data = dst.read(window=rio.windows.Window(0, 0, 512, 512))
# %%
# Display
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(s2_l1c_data.transpose(1, 2, 0) / 3000)
ax[0].set_title("Sentinel-2 L2A")
ax[1].imshow(s2_label_data[0])
ax[1].set_title("Human Label")
plt.tight_layout()
# %%
dataset1 = hf_hub_download("tacofoundation/CloudSEN12",
                           "cloudsen12-l1c.0000.part.taco", repo_type="dataset", local_dir="/home/telepix_nas/junghwan/cloud_seg/")
dataset2 = hf_hub_download("tacofoundation/CloudSEN12",
                           "cloudsen12-l1c.0001.part.taco", repo_type="dataset", local_dir="/home/telepix_nas/junghwan/cloud_seg/")
dataset3 = hf_hub_download("tacofoundation/CloudSEN12",
                           "cloudsen12-l1c.0002.part.taco", repo_type="dataset", local_dir="/home/telepix_nas/junghwan/cloud_seg/")
dataset4 = hf_hub_download("tacofoundation/CloudSEN12",
                           "cloudsen12-l1c.0003.part.taco", repo_type="dataset", local_dir="/home/telepix_nas/junghwan/cloud_seg/")
dataset5 = hf_hub_download("tacofoundation/CloudSEN12",
                           "cloudsen12-l1c.0004.part.taco", repo_type="dataset", local_dir="/home/telepix_nas/junghwan/cloud_seg/")

dataset6 = hf_hub_download("tacofoundation/CloudSEN12",
                           "cloudsen12-l2a.0000.part.taco", repo_type="dataset", local_dir="/home/telepix_nas/junghwan/cloud_seg/")
dataset7 = hf_hub_download("tacofoundation/CloudSEN12",
                           "cloudsen12-l2a.0001.part.taco", repo_type="dataset", local_dir="/home/telepix_nas/junghwan/cloud_seg/")
dataset3 = hf_hub_download("tacofoundation/CloudSEN12",
                           "cloudsen12-l2a.0002.part.taco", repo_type="dataset", local_dir="/home/telepix_nas/junghwan/cloud_seg/")
dataset8 = hf_hub_download("tacofoundation/CloudSEN12",
                           "cloudsen12-l2a.0003.part.taco", repo_type="dataset", local_dir="/home/telepix_nas/junghwan/cloud_seg/")
dataset9 = hf_hub_download("tacofoundation/CloudSEN12",
                           "cloudsen12-l2a.0004.part.taco", repo_type="dataset", local_dir="/home/telepix_nas/junghwan/cloud_seg/")
dataset10 = hf_hub_download("tacofoundation/CloudSEN12",
                            "cloudsen12-l2a.0005.part.taco", repo_type="dataset", local_dir="/home/telepix_nas/junghwan/cloud_seg/")
