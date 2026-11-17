import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset


def _default_transform(image_size):
    """Plain PIL + numpy resize/normalize, avoids a torchvision dependency
    entirely (torchvision/torch version mismatches are a common source of
    import errors, e.g. `AttributeError: module 'torch.library' has no
    attribute 'register_fake'` when torch and torchvision versions drift
    apart in a conda env)."""
    def transform(pil_image):
        pil_image = pil_image.resize((image_size, image_size))
        arr = np.asarray(pil_image, dtype=np.float32) / 255.0  # [H, W, 3] in [0,1]
        arr = (arr - 0.5) / 0.5                                 # normalize to [-1,1]
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # -> [3, H, W]
        return tensor
    return transform
 
 
def _augmented_transform(image_size, max_rotation_deg=15, brightness_jitter=0.2, contrast_jitter=0.2):
    """Adds light data augmentation on top of the plain resize/normalize
    above. Useful when the training set is small (e.g. ~90 images/class),
    where augmentation helps the model generalize instead of memorizing
    the few examples it has.
 
    Deliberately NO horizontal/vertical flip: a flipped traffic sign can
    change or invert its real-world meaning (e.g. a left-turn arrow
    becomes right-turn), so flipping would create incorrect training
    labels rather than valid new examples.
 
    - Random rotation: simulates the camera not being perfectly level
      (real photos of signs are rarely dead-on).
    - Brightness/contrast jitter: simulates different lighting conditions.
    """
    from PIL import ImageEnhance
    import random as _random
 
    def transform(pil_image):
        pil_image = pil_image.resize((image_size, image_size))
 
        angle = _random.uniform(-max_rotation_deg, max_rotation_deg)
        pil_image = pil_image.rotate(angle, fillcolor=(128, 128, 128))
 
        if brightness_jitter > 0:
            factor = 1.0 + _random.uniform(-brightness_jitter, brightness_jitter)
            pil_image = ImageEnhance.Brightness(pil_image).enhance(factor)
        if contrast_jitter > 0:
            factor = 1.0 + _random.uniform(-contrast_jitter, contrast_jitter)
            pil_image = ImageEnhance.Contrast(pil_image).enhance(factor)
 
        arr = np.asarray(pil_image, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        return tensor
    return transform
 
 
# ---------------------------------------------------------------------------
# 0. Dataset: reads a CSV of (filename, label) pairs + an image folder
# ---------------------------------------------------------------------------
class ArabicTrafficSignsDataset(Dataset):
    _FILENAME_CANDIDATES = ["filename", "Filename", "image", "Image", "image_name", "Image_Name", "path", "Path"]
    _LABEL_CANDIDATES = ["label", "Label", "class", "Class", "ClassId", "class_id", "Category", "category"]
 
    def __init__(self, img_dir, csv_path, filename_col=None, label_col=None,
                 image_size=64, transform=None, augment=False):
        self.img_dir = img_dir
        df = pd.read_csv(csv_path)
 
        self.filename_col = filename_col or self._detect_column(df, self._FILENAME_CANDIDATES)
        self.label_col = label_col or self._detect_column(df, self._LABEL_CANDIDATES)
 
        if self.filename_col is None or self.label_col is None:
            # Common cause: the CSV has NO header row, so pandas treated the
            # first data row (e.g. "front_or_left-1.png,0") as column names
            # instead of data. Retry assuming no header and 2 positional
            # columns: [filename, label].
            df_noheader = pd.read_csv(csv_path, header=None)
            if df_noheader.shape[1] == 2:
                df = df_noheader
                df.columns = ["filename", "label"]
                self.filename_col, self.label_col = "filename", "label"
                print(f"Note: '{csv_path}' has no header row -- assuming "
                      f"column 0 = filename, column 1 = label. "
                      f"First 3 rows:\n{df.head(3)}\n")
            else:
                raise ValueError(
                    f"Could not auto-detect filename/label columns in {csv_path}. "
                    f"With header: {list(df.columns)}. "
                    f"Without header, found {df_noheader.shape[1]} columns "
                    f"(expected 2): {df_noheader.head(3).to_dict()}. "
                    f"Pass filename_col=... and label_col=... explicitly, "
                    f"or header_row=None if your CSV truly has no header."
                )
 
        self.df = df.reset_index(drop=True)
 
        # map arbitrary label values (strings or non-zero-based ints) to 0..N-1
        # cast to native Python types (not numpy.int64/str_) so the mapping
        # can be safely pickled/unpickled by torch.load(weights_only=True)
        unique_labels = sorted(self.df[self.label_col].unique().tolist())
        self.label_to_idx = {(lbl.item() if hasattr(lbl, "item") else lbl): i
                              for i, lbl in enumerate(unique_labels)}
        self.num_classes = len(unique_labels)
 
        self.transform = transform or (
            _augmented_transform(image_size) if augment else _default_transform(image_size)
        )
 
    @staticmethod
    def _detect_column(df, candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None
 
    _IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"]
 
    def __len__(self):
        return len(self.df)
 
    def _resolve_image_path(self, fname):
        """Handles CSV filenames that lack an extension (a common mismatch
        between label CSVs and the actual image files on disk)."""
        direct_path = os.path.join(self.img_dir, fname)
        if os.path.exists(direct_path):
            return direct_path
 
        # no extension in the CSV entry (or a wrong one) -- try common ones
        base, ext = os.path.splitext(fname)
        if ext:  # already had *some* extension but it still didn't exist
            base = fname
        for candidate_ext in self._IMAGE_EXTENSIONS:
            candidate_path = os.path.join(self.img_dir, base + candidate_ext)
            if os.path.exists(candidate_path):
                return candidate_path
 
        raise FileNotFoundError(
            f"Could not find image for CSV entry '{fname}' in '{self.img_dir}'. "
            f"Tried direct path and extensions {self._IMAGE_EXTENSIONS}. "
            f"Check that the CSV filename matches an actual file in that folder."
        )
 
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = str(row[self.filename_col])
        img_path = self._resolve_image_path(fname)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        label = self.label_to_idx[row[self.label_col]]
        return image, label
 
 
def split_train_val(train_dir, train_csv, image_size=64, val_ratio=0.15,
                     augment_train=True, batch_size=32, split_seed=42):
    """Memecah data TRAINING (bukan test) jadi train + validation, supaya
    semua monitoring/model-selection selama training (keep_best di
    kd_train, iterative_prune, sweep_pruning_rates) bisa memakai validation
    set -- BUKAN test set. Test set (TEST_DIR/TEST_CSV) harus tetap
    tersentuh HANYA SEKALI, di akhir segalanya, untuk laporan final yang
    jujur secara metodologis.
 
    PENTING -- split_seed=42 (TETAP, bukan acak): supaya split train/val
    KONSISTEN di setiap kali main() dipanggil, di run manapun. Tanpa ini,
    checkpoint yang dilatih di Run 1 (dengan split acak #1) lalu dimuat
    ulang di Run 2 (dengan split acak #2 yang BEDA) akan menyebabkan
    sebagian gambar "validation" di Run 2 ternyata adalah gambar yang
    SUDAH DILATIH di Run 1 (karena termasuk di "train" Run 1 tapi kebetulan
    jadi "val" di split acak Run 2) -- mencemari validation TANPA disadari,
    walau split di masing-masing run sendiri sudah benar secara internal.
    Dengan seed tetap, 330 gambar validation SELALU sama persis di run
    manapun, sehingga sebuah checkpoint yang dilatih HANYA di 1870 gambar
    "train" di satu run tetap konsisten bersih ketika dievaluasi ulang di
    run lain manapun.
 
    PENTING: validation TIDAK diaugmentasi (augment=False selalu) --
    supaya pengukurannya stabil/bersih, sama seperti alasan test set juga
    tidak diaugmentasi. Training tetap diaugmentasi (augment_train=True
    default), sesuai temuan kita sebelumnya bahwa augmentasi membantu
    dataset kecil seperti ini.
 
    Dua Dataset OBJEK TERPISAH dibangun dari CSV/folder yang SAMA (satu
    augment=True untuk train, satu augment=False untuk val) supaya kedua
    subset punya perilaku transform yang benar -- lalu di-split pakai
    INDEKS yang sama (via Subset) supaya gambar yang masuk ke train vs val
    konsisten, tidak ada overlap.
 
    Return: (train_loader, val_loader, num_classes, label_to_idx)"""
    ds_augmented = ArabicTrafficSignsDataset(train_dir, train_csv, image_size=image_size,
                                              augment=augment_train)
    ds_plain = ArabicTrafficSignsDataset(train_dir, train_csv, image_size=image_size,
                                          augment=False)
    # pastikan label_to_idx & num_classes identik di kedua objek (harusnya
    # otomatis sama karena baca CSV yang sama, tapi dipaksa sinkron untuk aman)
    ds_plain.label_to_idx = ds_augmented.label_to_idx
    ds_plain.num_classes = ds_augmented.num_classes
 
    n_total = len(ds_augmented)
    n_val = max(1, int(round(n_total * val_ratio)))
    n_train = n_total - n_val
 
    # generator TERPISAH dengan seed TETAP -- supaya urutan acak SELALU
    # sama, tidak peduli apakah ada kode lain yang sudah "mencuri" nomor
    # acak dari generator torch global sebelumnya (mis. inisialisasi bobot
    # model, augmentasi, dsb -- semua itu masih tetap acak seperti biasa,
    # cuma split ini yang dikunci reproducible)
    split_generator = torch.Generator().manual_seed(split_seed)
    perm = torch.randperm(n_total, generator=split_generator).tolist()
    train_indices = perm[:n_train]
    val_indices = perm[n_train:]
 
    train_subset = Subset(ds_augmented, train_indices)
    val_subset = Subset(ds_plain, val_indices)
 
    print(f"Split train/val: {n_train} train ({100*(1-val_ratio):.0f}%), "
          f"{n_val} val ({100*val_ratio:.0f}%) -- val TANPA augmentasi, "
          f"TIDAK PERNAH memakai test set asli.")
 
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
 
    return train_loader, val_loader, ds_augmented.num_classes, ds_augmented.label_to_idx
 
 
# ---------------------------------------------------------------------------
# 1a. RepVGG: multi-branch training block that RE-PARAMETERIZES into a
#     single plain Conv-BN-ReLU (no skip connections) before pruning.
#
#     Why: training with 3 branches (3x3 conv, 1x1 conv, identity) gives
#     ResNet-like gradient flow and typically better accuracy than a plain
#     stack. After training, the 3 branches are fused mathematically into
#     ONE Conv2d(bias=False) + BatchNorm2d pair per block -- so the network
#     that actually gets pruned is purely sequential, exactly matching what
#     prune_layerwise() / prune_sequential_model() below already handle,
#     with NO modification needed to that code.
#
#     Fusion correctness was verified block-by-block and end-to-end before
#     shipping this: multi-branch output vs. fused output differ by ~1e-5,
#     i.e. floating-point noise, not a real discrepancy.
# ---------------------------------------------------------------------------
class RepVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
 
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn3x3 = nn.BatchNorm2d(out_channels)
 
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0, bias=False)
        self.bn1x1 = nn.BatchNorm2d(out_channels)
 
        self.identity_bn = nn.BatchNorm2d(out_channels) \
            if (in_channels == out_channels and stride == 1) else None
 
        self.nonlinearity = nn.ReLU()
 
    def forward(self, x):
        out = self.bn3x3(self.conv3x3(x)) + self.bn1x1(self.conv1x1(x))
        if self.identity_bn is not None:
            out = out + self.identity_bn(x)
        return self.nonlinearity(out)
 
    @staticmethod
    def _fuse_conv_bn(conv_weight, bn):
        gamma, beta = bn.weight, bn.bias
        running_mean, running_var, eps = bn.running_mean, bn.running_var, bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return conv_weight * t, beta - running_mean * gamma / std
 
    def _identity_to_3x3_kernel(self):
        input_dim = self.in_channels
        kernel = torch.zeros(self.out_channels, input_dim, 3, 3,
                              device=self.conv3x3.weight.device, dtype=self.conv3x3.weight.dtype)
        for i in range(self.out_channels):
            kernel[i, i % input_dim, 1, 1] = 1.0
        return kernel
 
    def get_equivalent_kernel_bias(self):
        k3, b3 = self._fuse_conv_bn(self.conv3x3.weight, self.bn3x3)
        k1, b1 = self._fuse_conv_bn(self.conv1x1.weight, self.bn1x1)
        k1_padded = F.pad(k1, [1, 1, 1, 1])
        if self.identity_bn is not None:
            kid_raw = self._identity_to_3x3_kernel()
            kid, bid = self._fuse_conv_bn(kid_raw, self.identity_bn)
        else:
            kid, bid = 0, 0
        return k3 + k1_padded + kid, b3 + b1 + bid
 
 
class RepVGGNet(nn.Module):
    """Multi-branch training-time network. switch_to_deploy() fuses every
    block and returns a plain TrafficSignNet instance."""
 
    def __init__(self, num_classes, channels=(3, 32, 64, 128, 128)):
        super().__init__()
        self.channels = channels
        self.blocks = nn.ModuleList(
            RepVGGBlock(c_in, c_out, stride=1) for c_in, c_out in zip(channels[:-1], channels[1:])
        )
        self.pool = nn.MaxPool2d(2)
        self.classifier = nn.Linear(channels[-1], num_classes)
 
    def forward(self, x):
        for block in self.blocks:
            x = self.pool(block(x))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.classifier(x)
 
    @torch.no_grad()
    def switch_to_deploy(self):
        """Fuses every RepVGGBlock into Conv2d(bias=False)+BatchNorm2d and
        returns a TrafficSignNet -- plugs directly into sparse_train /
        prune_layerwise / kd_train / save-load with zero changes needed
        to that code."""
        self.eval()  # BN must use running stats for the fusion math to be exact
 
        deploy_model = TrafficSignNet.__new__(TrafficSignNet)
        nn.Module.__init__(deploy_model)
        new_convs, new_bns = [], []
 
        for block in self.blocks:
            kernel, bias = block.get_equivalent_kernel_bias()
 
            conv = nn.Conv2d(block.in_channels, block.out_channels,
                              kernel_size=3, stride=block.stride, padding=1, bias=False)
            conv.weight.data = kernel.clone()
 
            # Absorb the fused bias into a fresh BN's beta (gamma=1,
            # running_mean=0, running_var=1) so BN(conv(x)) == conv(x)+bias
            # exactly -- keeps the BN-gamma pruning interface (Eq. 10)
            # intact while preserving the fused numerical output.
            bn = nn.BatchNorm2d(block.out_channels)
            bn.weight.data = torch.ones_like(bn.weight.data)
            bn.bias.data = bias.clone()
            bn.running_mean.data = torch.zeros_like(bn.running_mean.data)
            bn.running_var.data = torch.ones_like(bn.running_var.data)
 
            new_convs.append(conv)
            new_bns.append(bn)
 
        deploy_model.convs = nn.ModuleList(new_convs)
        deploy_model.bns = nn.ModuleList(new_bns)
        deploy_model.pool = nn.MaxPool2d(2)
        deploy_model.classifier = nn.Linear(self.channels[-1], self.classifier.out_features)
        deploy_model.classifier.weight.data = self.classifier.weight.data.clone()
        deploy_model.classifier.bias.data = self.classifier.bias.data.clone()
 
        return deploy_model
 
 
def train_repvgg_and_fuse(num_classes, dataloader, channels=(3, 32, 64, 128, 128),
                           epochs=15, lr=1e-3, device="cpu"):
    """Trains a RepVGGNet (multi-branch) with plain cross-entropy, then
    fuses it into a deploy-mode TrafficSignNet, ready to be handed to
    sparse_train() for the usual pruning pipeline."""
    model = RepVGGNet(num_classes=num_classes, channels=channels)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.937, weight_decay=0.005)
    model.train()
 
    for epoch in range(epochs):
        total_loss, n_batches = 0.0, 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        print(f"  [repvgg-train] epoch {epoch + 1}/{epochs}  loss={total_loss / max(n_batches,1):.4f}")
 
    deploy_model = model.switch_to_deploy()
    deploy_model.to(device)
    return deploy_model
 
 
# ---------------------------------------------------------------------------
# 1. Sequential Conv-BN-ReLU classifier (no skip connections -- required by
#    the pruning method below, see the scope note in bn_channel_pruning.py)
# ---------------------------------------------------------------------------
class TrafficSignNet(nn.Module):
    def __init__(self, num_classes, channels=(3, 32, 64, 128, 128)):
        super().__init__()
        convs, bns = [], []
        for c_in, c_out in zip(channels[:-1], channels[1:]):
            convs.append(nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(c_out))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.pool = nn.MaxPool2d(2)  # downsample between conv blocks
        self.classifier = nn.Linear(channels[-1], num_classes)
 
    def forward(self, x):
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x)))
            x = self.pool(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.classifier(x)
 
    def bn_layers(self):
        triples = []
        for i, bn in enumerate(self.bns):
            producer = self.convs[i]
            consumer = self.convs[i + 1] if i + 1 < len(self.convs) else None
            triples.append((producer, bn, consumer))
        return triples
 
 
 
 
# ---------------------------------------------------------------------------
# 1b. VGG (pretrained ImageNet) -- backbone alternatif. Dibangun MANUAL di
#     sini (bukan import torchvision.models) supaya tidak bergantung pada
#     torchvision sama sekali -- ini menghindari error
#     "AttributeError: module 'torch.library' has no attribute
#     'register_fake'" yang terjadi kalau versi torch dan torchvision di
#     environment Anda tidak sinkron (masalah umum di conda env lama).
#
#     Arsitektur di bawah direplikasi PERSIS sama dengan definisi resmi
#     torchvision (nama layer, urutan, susunan channel) -- sudah diverifikasi
#     bahwa state_dict() keduanya punya kunci yang identik 100%, sehingga
#     bobot pretrained ImageNet resmi tetap bisa dimuat langsung tanpa
#     konversi apapun. Bobotnya diambil lewat torch.hub (bagian INTI
#     PyTorch, bukan torchvision), jadi satu-satunya kebutuhan adalah akses
#     internet ke download.pytorch.org -- tidak butuh package torchvision
#     terinstall/berfungsi sama sekali.
#
#     VGG TIDAK PERNAH punya skip connection sejak awal (beda dari ResNet),
#     jadi tidak butuh trik fusion seperti RepVGG -- features-nya sudah
#     sekuensial murni. Bedanya dari TrafficSignNet: VGG tidak pooling di
#     SETIAP layer, jadi perlu fungsi pruning yang lebih umum
#     (prune_vgg_layerwise di bawah).
# ---------------------------------------------------------------------------
_VGG_CFGS = {
    "vgg11_bn": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg16_bn": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M",
                 512, 512, 512, "M", 512, 512, 512, "M"],
}
_VGG_WEIGHT_URLS = {
    "vgg11_bn": "https://download.pytorch.org/models/vgg11_bn-6002323d.pth",
    "vgg16_bn": "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
}
 
 
def _make_vgg_feature_layers(cfg):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers.append(nn.Conv2d(in_channels, v, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(v))
            layers.append(nn.ReLU(inplace=True))
            in_channels = v
    return nn.Sequential(*layers)
 
 
class VGGManual(nn.Module):
    """Replikasi manual VGG-BN (arsitektur ImageNet, 1000 kelas default),
    tanpa dependency ke torchvision.models. Nama/urutan layer PERSIS sama
    dengan definisi resmi torchvision, jadi state_dict pretrained resmi
    bisa dimuat langsung.
 
    Bisa dibangun dari nama arsitektur standar (arch="vgg16_bn") ATAU dari
    cfg custom (list channel + 'M' untuk pooling) -- yang kedua dipakai
    untuk merekonstruksi model yang SUDAH di-pruning (channel per layer
    beda dari standar), supaya bisa disimpan & dimuat ulang dengan benar.
 
    fc_widths=(fc1_out, fc2_out): default (4096, 4096) seperti VGG standar,
    tapi kalau classifier sudah dipangkas lebarnya (prune_vgg_classifier_fc),
    lebar barunya HARUS diberikan di sini supaya rekonstruksi checkpoint
    tidak size-mismatch."""
 
    def __init__(self, arch="vgg16_bn", num_classes=1000, cfg=None, fc_widths=(4096, 4096)):
        super().__init__()
        self.cfg = cfg if cfg is not None else _VGG_CFGS[arch]
        self.fc_widths = fc_widths
        self.features = _make_vgg_feature_layers(self.cfg)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        last_channels = [v for v in self.cfg if v != "M"][-1]
        fc1_out, fc2_out = fc_widths
        self.classifier = nn.Sequential(
            nn.Linear(last_channels * 7 * 7, fc1_out), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(fc1_out, fc2_out), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(fc2_out, num_classes),
        )
 
    def forward(self, x):
        x = self.avgpool(self.features(x))
        x = x.flatten(1)
        return self.classifier(x)
 
 
def get_vgg_fc_widths_from_model(vgg_model):
    """Baca lebar fc1/fc2 SEBENARNYA dari model VGG (bisa yang sudah
    dipangkas ataupun belum) -- dipakai supaya save_checkpoint/load_checkpoint
    bisa merekonstruksi classifier dengan shape yang tepat."""
    layers = list(vgg_model.classifier.children())
    fc1, fc2 = layers[0], layers[3]
    return (fc1.out_features, fc2.out_features)
 
 
def get_vgg_cfg_from_model(vgg_model):
    """Walks a (possibly pruned) VGGManual's .features and reconstructs its
    actual cfg list (channel counts + 'M' for pooling), reflecting whatever
    the CURRENT architecture is -- needed to correctly save/reload a model
    after pruning has changed its per-layer channel counts."""
    cfg = []
    for layer in vgg_model.features.children():
        if isinstance(layer, nn.Conv2d):
            cfg.append(layer.out_channels)
        elif isinstance(layer, nn.MaxPool2d):
            cfg.append("M")
    return cfg
 
 
def build_vgg_classifier(num_classes, arch="vgg16_bn", pretrained=True):
    """Bangun VGG manual, opsional muat bobot pretrained ImageNet lewat
    torch.hub (TIDAK butuh torchvision sama sekali), lalu ganti layer
    classifier terakhir untuk num_classes Anda."""
    if arch not in _VGG_CFGS:
        raise ValueError(f"arch tidak dikenal: '{arch}' (pakai 'vgg16_bn' atau 'vgg11_bn')")
 
    model = VGGManual(arch=arch, num_classes=1000)  # 1000 dulu, biar cocok sama bobot ImageNet
 
    if pretrained:
        print(f"  Mengunduh bobot pretrained {arch} dari {_VGG_WEIGHT_URLS[arch]} ...")
        state_dict = torch.hub.load_state_dict_from_url(
            _VGG_WEIGHT_URLS[arch], progress=True
        )
        model.load_state_dict(state_dict)
        print("  Bobot pretrained berhasil dimuat.")
 
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model
 
 
def get_conv_bn_pairs(sequential_module):
    """Telusuri sebuah nn.Sequential, kembalikan list (conv, bn, index)
    untuk setiap pasangan Conv2d diikuti langsung BatchNorm2d."""
    pairs = []
    children = list(sequential_module.children())
    i = 0
    while i < len(children):
        if isinstance(children[i], nn.Conv2d) and i + 1 < len(children) \
                and isinstance(children[i + 1], nn.BatchNorm2d):
            pairs.append((children[i], children[i + 1], i))
            i += 2
        else:
            i += 1
    return pairs
 
 
def prune_vgg_layerwise(vgg_model, pruning_rate, min_channels_per_layer=8):
    """Versi umum prune_layerwise() untuk struktur VGG (nn.Sequential
    dengan ReLU/MaxPool2d di antara pasangan Conv-BN, bukan pooling
    seragam setiap layer seperti TrafficSignNet)."""
    pairs = get_conv_bn_pairs(vgg_model.features)
    new_features = list(vgg_model.features.children())
    prev_keep_idx = None
 
    for conv, bn, idx in pairs:
        gamma_abs = bn.weight.data.abs()
        n = gamma_abs.numel()
        n_keep = max(min_channels_per_layer, int(round(n * (1 - pruning_rate))))
        n_keep = min(n_keep, n)
        topk = torch.topk(gamma_abs, k=n_keep).indices
        keep_idx = torch.sort(topk).values
 
        in_idx = prev_keep_idx
        in_channels = conv.in_channels if in_idx is None else in_idx.numel()
 
        new_conv = nn.Conv2d(
            in_channels=in_channels, out_channels=keep_idx.numel(),
            kernel_size=conv.kernel_size, stride=conv.stride,
            padding=conv.padding, bias=conv.bias is not None,
        )
        w = conv.weight.data[keep_idx, :, :, :]
        if in_idx is not None:
            w = w[:, in_idx, :, :]
        new_conv.weight.data = w.clone()
        if conv.bias is not None:
            new_conv.bias.data = conv.bias.data[keep_idx].clone()
 
        new_bn = nn.BatchNorm2d(num_features=keep_idx.numel())
        new_bn.weight.data = bn.weight.data[keep_idx].clone()
        new_bn.bias.data = bn.bias.data[keep_idx].clone()
        new_bn.running_mean.data = bn.running_mean.data[keep_idx].clone()
        new_bn.running_var.data = bn.running_var.data[keep_idx].clone()
 
        new_features[idx] = new_conv
        new_features[idx + 1] = new_bn
        prev_keep_idx = keep_idx
 
    is_vgg16 = len(pairs) > 8
    arch = "vgg16_bn" if is_vgg16 else "vgg11_bn"
    pruned_model = VGGManual(arch=arch, num_classes=vgg_model.classifier[-1].out_features)
    pruned_model.features = nn.Sequential(*new_features)
 
    old_classifier = vgg_model.classifier
    old_fc0 = old_classifier[0]
    spatial = 7 * 7
    new_fc0 = nn.Linear(prev_keep_idx.numel() * spatial, old_fc0.out_features)
    old_weight = old_fc0.weight.data.view(old_fc0.out_features, -1, spatial)
    new_weight = old_weight[:, prev_keep_idx, :].reshape(old_fc0.out_features, -1)
    new_fc0.weight.data = new_weight.clone()
    new_fc0.bias.data = old_fc0.bias.data.clone()
 
    new_classifier_layers = [new_fc0] + list(old_classifier.children())[1:]
    pruned_model.classifier = nn.Sequential(*new_classifier_layers)
    pruned_model.avgpool = vgg_model.avgpool
 
    return pruned_model
 
 
def prune_vgg_classifier_fc(vgg_model, pruning_rate, min_neurons=64):
    """Memangkas LEBAR classifier VGG (fc1 output, fc2 input+output) --
    BERBEDA dari prune_vgg_layerwise() yang cuma memangkas conv/features.
 
    VGG classifier (fc1: 4096, fc2: 4096->4096, fc3: 4096->num_classes)
    TIDAK punya BatchNorm, jadi tidak ada sinyal 'gamma' seperti di conv
    layers. Kriteria kepentingan di sini: L1-norm (jumlah nilai absolut)
    dari bobot masuk tiap neuron output -- neuron dengan bobot yang secara
    kolektif besar dianggap lebih penting, mirip semangatnya dengan
    kriteria gamma tapi diterapkan langsung ke magnitude bobot.
 
    fc3 output TIDAK dipangkas (itu = jumlah kelas, tidak boleh diubah).
    fc2 mendapat manfaat kompounding kuadratik yang sama seperti conv layers
    (in DAN out dimension-nya sama-sama menyusut), jadi ini efektif untuk
    memangkas ukuran file secara signifikan -- fc2 sendirian menyumbang
    ~16.8 juta parameter (~64 MB) di VGG16 standar, dan TIDAK tersentuh
    sama sekali oleh prune_vgg_layerwise() saja."""
    old_layers = list(vgg_model.classifier.children())
    fc1, relu1, drop1, fc2, relu2, drop2, fc3 = old_layers
 
    # --- pangkas OUTPUT fc1 (biasanya 4096) ---
    fc1_importance = fc1.weight.data.abs().sum(dim=1)
    n_keep_fc1 = max(min_neurons, int(round(fc1.out_features * (1 - pruning_rate))))
    n_keep_fc1 = min(n_keep_fc1, fc1.out_features)
    fc1_keep_idx = torch.sort(torch.topk(fc1_importance, k=n_keep_fc1).indices).values
 
    new_fc1 = nn.Linear(fc1.in_features, n_keep_fc1)
    new_fc1.weight.data = fc1.weight.data[fc1_keep_idx, :].clone()
    new_fc1.bias.data = fc1.bias.data[fc1_keep_idx].clone()
 
    # --- pangkas INPUT fc2 (ikut fc1 yang dipangkas) dan OUTPUT fc2 ---
    fc2_weight_input_sliced = fc2.weight.data[:, fc1_keep_idx]
    fc2_importance = fc2_weight_input_sliced.abs().sum(dim=1)
    n_keep_fc2 = max(min_neurons, int(round(fc2.out_features * (1 - pruning_rate))))
    n_keep_fc2 = min(n_keep_fc2, fc2.out_features)
    fc2_keep_idx = torch.sort(torch.topk(fc2_importance, k=n_keep_fc2).indices).values
 
    new_fc2 = nn.Linear(n_keep_fc1, n_keep_fc2)
    new_fc2.weight.data = fc2_weight_input_sliced[fc2_keep_idx, :].clone()
    new_fc2.bias.data = fc2.bias.data[fc2_keep_idx].clone()
 
    # --- pangkas INPUT fc3 saja (output = num_classes, TIDAK diubah) ---
    new_fc3 = nn.Linear(n_keep_fc2, fc3.out_features)
    new_fc3.weight.data = fc3.weight.data[:, fc2_keep_idx].clone()
    new_fc3.bias.data = fc3.bias.data.clone()
 
    vgg_model.classifier = nn.Sequential(new_fc1, relu1, drop1, new_fc2, relu2, drop2, new_fc3)
    return vgg_model
 
 
def prune_vgg_full(vgg_model, pruning_rate, min_channels_per_layer=8, min_neurons=64):
    """Gabungan prune_vgg_layerwise() (conv/features) + prune_vgg_classifier_fc()
    (fc1/fc2 classifier) dengan SATU pruning_rate yang sama -- supaya bisa
    dipakai langsung sebagai prune_fn di iterative_prune(), yang cuma
    menerima (model, pruning_rate, min_channels_per_layer)."""
    pruned = prune_vgg_layerwise(vgg_model, pruning_rate, min_channels_per_layer)
    pruned = prune_vgg_classifier_fc(pruned, pruning_rate, min_neurons=min_neurons)
    return pruned
 
 
# ---------------------------------------------------------------------------
# 2. Stage 1: sparse regularization training (Eq. 10, lambda = 0.008)
# ---------------------------------------------------------------------------
def sparse_train(model, dataloader, epochs=5, lr=1e-3, sparsity_lambda=0.008, device="cpu"):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.937, weight_decay=0.005)
    model.train()
    model.to(device)
 
    loss_history = []
    for epoch in range(epochs):
        total_loss, n_batches = 0.0, 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            task_loss = F.cross_entropy(out, y)
 
            l1_penalty = sum(
                m.weight.abs().sum() for m in model.modules() if isinstance(m, nn.BatchNorm2d)
            )
            loss = task_loss + sparsity_lambda * l1_penalty
 
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
 
        avg_loss = total_loss / max(n_batches, 1)
        loss_history.append(avg_loss)
        print(f"  [sparse-train] epoch {epoch + 1}/{epochs}  loss={avg_loss:.4f}")
    return model, loss_history
 
 
# ---------------------------------------------------------------------------
# 3. Stage 2a: global threshold (Section 4.2, default 45% pruning rate)
# ---------------------------------------------------------------------------
def compute_global_threshold(model, pruning_rate=0.45):
    all_gammas = torch.cat(
        [m.weight.data.abs().clone() for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    )
    sorted_gammas, _ = torch.sort(all_gammas)
    idx = int(len(sorted_gammas) * pruning_rate)
    idx = min(max(idx, 0), len(sorted_gammas) - 1)
    return sorted_gammas[idx].item()
 
 
# ---------------------------------------------------------------------------
# 4. Stage 2b: physical channel removal (network surgery)
#
#    Two pruning strategies are provided:
#
#    - prune_sequential_model(): a SINGLE global threshold across all BN
#      layers (the paper's Section 4.2 approach). This assumes gamma values
#      are on a comparable scale across layers -- if they aren't, a global
#      threshold can disproportionately gut one layer (observed: on a real
#      run, the first conv layer lost 38% of its channels while a later
#      layer lost only 6%, purely from gamma-scale differences, not real
#      importance differences).
#
#    - prune_layerwise(): each layer independently keeps its own top
#      (1 - pruning_rate) fraction of channels by |gamma|. This avoids the
#      cross-layer scale bias above. Empirically, on a controlled learnable
#      task, this recovered full accuracy at rates where the global-
#      threshold method collapsed to ~25% (near chance): see the comparison
#      table in this repo's test notes. RECOMMENDED DEFAULT.
# ---------------------------------------------------------------------------
def print_gamma_diagnostics(model):
    """Prints per-layer BN gamma spread, so you can see whether sparse
    training actually separated 'important' from 'unimportant' channels
    (paper Fig. 10), or whether gamma just shrank uniformly (in which case
    ANY pruning method is picking near-arbitrary channels, and the real fix
    is more sparse_epochs / a higher sparsity_lambda / more training data).
    Works for both TrafficSignNet (model.bns) and VGG (model.features)."""
    print("\nGamma diagnostics (per BN layer):")
    bn_layers = list(model.bns) if hasattr(model, "bns") else \
        [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    for i, bn in enumerate(bn_layers):
        g = bn.weight.data.abs()
        spread = (g.max() - g.min()).item()
        relative_spread = spread / (g.mean().item() + 1e-8)
        flag = "  <- LOW SEPARATION, pruning here is close to arbitrary" \
            if relative_spread < 0.05 else ""
        print(f"  layer {i}: min={g.min():.4f} mean={g.mean():.4f} max={g.max():.4f} "
              f"relative_spread={relative_spread:.4f}{flag}")
 
 
def export_gamma_distribution(model, path):
    """Ekspor NILAI |gamma| MENTAH tiap channel di tiap layer ke CSV --
    beda dari print_gamma_diagnostics() yang cuma cetak ringkasan
    (min/mean/max/relative_spread). Data mentah ini yang dibutuhkan untuk
    membuat histogram/violin plot distribusi gamma per layer (pembuktian
    visual 'clear separation' antara channel penting vs tidak penting,
    konsep dari Eq. 8-10 paper).
 
    Format CSV: kolom 'layer' (indeks layer), 'gamma_abs' (nilai |gamma|
    satu channel) -- long format, satu baris per channel, siap langsung
    dipakai seaborn.violinplot() / histplot() dengan hue='layer'."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    bn_layers = list(model.bns) if hasattr(model, "bns") else \
        [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
 
    rows = []
    for layer_idx, bn in enumerate(bn_layers):
        gamma_abs = bn.weight.data.abs().cpu().numpy()
        for ch_idx, g in enumerate(gamma_abs):
            rows.append({"layer": layer_idx, "channel": ch_idx, "gamma_abs": float(g)})
 
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  Distribusi gamma ({len(rows)} nilai, {len(bn_layers)} layer) disimpan ke: {path}")
 
 
def prune_sequential_model(model, threshold, min_channels_per_layer=2):
    """Global-threshold pruning (paper Section 4.2). See module note above
    for when this can behave worse than prune_layerwise()."""
    new_convs, new_bns = [], []
    prev_keep_idx = None
 
    for conv, bn, _ in model.bn_layers():
        gamma_abs = bn.weight.data.abs()
        mask = gamma_abs.gt(threshold)
        keep_idx = torch.where(mask)[0]
 
        if keep_idx.numel() < min_channels_per_layer:
            topk = torch.topk(gamma_abs, k=min(min_channels_per_layer, gamma_abs.numel())).indices
            keep_idx = torch.sort(topk).values
 
        in_idx = prev_keep_idx
        in_channels = conv.in_channels if in_idx is None else in_idx.numel()
 
        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=keep_idx.numel(),
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            bias=conv.bias is not None,
        )
        w = conv.weight.data[keep_idx, :, :, :]
        if in_idx is not None:
            w = w[:, in_idx, :, :]
        new_conv.weight.data = w.clone()
        if conv.bias is not None:
            new_conv.bias.data = conv.bias.data[keep_idx].clone()
 
        new_bn = nn.BatchNorm2d(num_features=keep_idx.numel())
        new_bn.weight.data = bn.weight.data[keep_idx].clone()
        new_bn.bias.data = bn.bias.data[keep_idx].clone()
        new_bn.running_mean.data = bn.running_mean.data[keep_idx].clone()
        new_bn.running_var.data = bn.running_var.data[keep_idx].clone()
 
        new_convs.append(new_conv)
        new_bns.append(new_bn)
        prev_keep_idx = keep_idx
 
    pruned = TrafficSignNet.__new__(TrafficSignNet)
    nn.Module.__init__(pruned)
    pruned.convs = nn.ModuleList(new_convs)
    pruned.bns = nn.ModuleList(new_bns)
    pruned.pool = nn.MaxPool2d(2)
 
    old_fc = model.classifier
    pruned.classifier = nn.Linear(prev_keep_idx.numel(), old_fc.out_features)
    pruned.classifier.weight.data = old_fc.weight.data[:, prev_keep_idx].clone()
    pruned.classifier.bias.data = old_fc.bias.data.clone()
 
    return pruned
 
 
def prune_layerwise(model, pruning_rate, min_channels_per_layer=2):
    """RECOMMENDED. Each layer independently keeps its own top
    (1 - pruning_rate) fraction of channels by |gamma|, avoiding the
    cross-layer gamma-scale bias that a single global threshold can suffer
    from (see module note above)."""
    new_convs, new_bns = [], []
    prev_keep_idx = None
 
    for conv, bn, _ in model.bn_layers():
        gamma_abs = bn.weight.data.abs()
        n = gamma_abs.numel()
        n_keep = max(min_channels_per_layer, int(round(n * (1 - pruning_rate))))
        n_keep = min(n_keep, n)
        topk = torch.topk(gamma_abs, k=n_keep).indices
        keep_idx = torch.sort(topk).values
 
        in_idx = prev_keep_idx
        in_channels = conv.in_channels if in_idx is None else in_idx.numel()
 
        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=keep_idx.numel(),
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            bias=conv.bias is not None,
        )
        w = conv.weight.data[keep_idx, :, :, :]
        if in_idx is not None:
            w = w[:, in_idx, :, :]
        new_conv.weight.data = w.clone()
        if conv.bias is not None:
            new_conv.bias.data = conv.bias.data[keep_idx].clone()
 
        new_bn = nn.BatchNorm2d(num_features=keep_idx.numel())
        new_bn.weight.data = bn.weight.data[keep_idx].clone()
        new_bn.bias.data = bn.bias.data[keep_idx].clone()
        new_bn.running_mean.data = bn.running_mean.data[keep_idx].clone()
        new_bn.running_var.data = bn.running_var.data[keep_idx].clone()
 
        new_convs.append(new_conv)
        new_bns.append(new_bn)
        prev_keep_idx = keep_idx
 
    pruned = TrafficSignNet.__new__(TrafficSignNet)
    nn.Module.__init__(pruned)
    pruned.convs = nn.ModuleList(new_convs)
    pruned.bns = nn.ModuleList(new_bns)
    pruned.pool = nn.MaxPool2d(2)
 
    old_fc = model.classifier
    pruned.classifier = nn.Linear(prev_keep_idx.numel(), old_fc.out_features)
    pruned.classifier.weight.data = old_fc.weight.data[:, prev_keep_idx].clone()
    pruned.classifier.bias.data = old_fc.bias.data.clone()
 
    return pruned
 
 
# ---------------------------------------------------------------------------
# 5. Stage 3: knowledge distillation (paper Eq. 11)
#    L_total = mu * L_hard(P_student, y) + (1 - mu) * L_soft(P_student, P_teacher)
#    where P = softmax(logits / temperature). This is what recovers the
#    accuracy lost during pruning -- pruning alone is NOT expected to
#    preserve accuracy on its own (see the paper's own Table 2, where
#    precision drops from 0.821 to 0.757 right after pruning, before KD).
# ---------------------------------------------------------------------------
def kd_train(student, teacher, dataloader, epochs=20, lr=1e-3,
             mu=0.5, temperature=4.0, device="cpu",
             test_loader=None, eval_every=1, keep_best=True):
    """teacher = the UNPRUNED trained model (frozen, eval mode).
    student = the PRUNED model (trainable).
 
    Sekarang mencatat hard_loss dan soft_loss TERPISAH per epoch (seperti
    Fig. 13(a) paper -- 'Hard Label Loss' dan 'Soft Label Loss'), bukan
    cuma loss gabungan. Kalau test_loader diberikan, akurasi juga dicatat
    tiap eval_every epoch (untuk membuat grafik seperti Fig. 13(b) paper --
    mAP@0.5 vs mAP@0.5_KD, di kita jadi accuracy vs accuracy_KD).
 
    keep_best=True (default): loss yang menurun TIDAK SELALU berarti akurasi
    membaik (soft_loss bisa terus turun -- model makin mirip distribusi
    probabilitas teacher -- tanpa argmax/akurasinya ikut membaik, bahkan
    kadang memburuk). Kalau ini terjadi, epoch TERAKHIR belum tentu yang
    TERBAIK. keep_best=True melacak state_dict di epoch dengan akurasi
    tertinggi, dan mengembalikan model di titik TERBAIK itu, bukan cuma
    epoch terakhir. Butuh test_loader untuk bekerja (kalau tidak ada,
    otomatis fallback ke epoch terakhir seperti biasa).
 
    Return: (student, history) -- history adalah dict berisi list per-epoch:
    'hard_loss', 'soft_loss', 'total_loss', 'accuracy', dan
    'best_epoch'/'best_accuracy' (kalau keep_best=True dan test_loader ada)."""
    teacher.eval()
    student.train()
    student.to(device)
    teacher.to(device)
    optimizer = torch.optim.SGD(student.parameters(), lr=lr, momentum=0.937, weight_decay=0.005)
 
    history = {"hard_loss": [], "soft_loss": [], "total_loss": [], "accuracy": []}
    best_acc = -1.0
    best_state = None
    best_epoch = None
 
    for epoch in range(epochs):
        total_loss, total_hard, total_soft, n_batches = 0.0, 0.0, 0.0, 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
 
            student_logits = student(x)
            with torch.no_grad():
                teacher_logits = teacher(x)
 
            hard_loss = F.cross_entropy(student_logits, y)
            soft_loss = F.kl_div(
                F.log_softmax(student_logits / temperature, dim=1),
                F.softmax(teacher_logits / temperature, dim=1),
                reduction="batchmean",
            ) * (temperature ** 2)
 
            loss = mu * hard_loss + (1 - mu) * soft_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_hard += hard_loss.item()
            total_soft += soft_loss.item()
            n_batches += 1
 
        avg_hard = total_hard / max(n_batches, 1)
        avg_soft = total_soft / max(n_batches, 1)
        avg_total = total_loss / max(n_batches, 1)
        history["hard_loss"].append(avg_hard)
        history["soft_loss"].append(avg_soft)
        history["total_loss"].append(avg_total)
 
        acc_str = ""
        if test_loader is not None and (epoch + 1) % eval_every == 0:
            student.eval()
            acc = evaluate(student, test_loader, device=device)
            student.train()
            history["accuracy"].append(acc)
            acc_str = f"  acc={acc:.4f}"
 
            if keep_best and acc > best_acc:
                best_acc = acc
                best_epoch = epoch + 1
                best_state = copy.deepcopy(student.state_dict())
                acc_str += "  <- TERBAIK sejauh ini"
 
        print(f"  [distill]     epoch {epoch + 1}/{epochs}  "
              f"hard_loss={avg_hard:.4f}  soft_loss={avg_soft:.4f}  "
              f"total={avg_total:.4f}{acc_str}")
 
    if keep_best and best_state is not None:
        student.load_state_dict(best_state)
        print(f"\n  [distill] Memulihkan bobot dari EPOCH TERBAIK ({best_epoch}, "
              f"akurasi={best_acc:.4f}) -- bukan epoch terakhir ({epochs}, "
              f"akurasi={history['accuracy'][-1] if history['accuracy'] else 'N/A'})")
        history["best_epoch"] = best_epoch
        history["best_accuracy"] = best_acc
 
    return student, history
 
 
def compare_kd_vs_no_kd(pruned_model, teacher, train_loader, test_loader,
                         epochs=30, lr=1e-4, mu=0.5, temperature=4.0, device="cpu"):
    """Melatih DUA salinan model paralel dari titik pruned yang SAMA:
    - 'no_kd'  : fine-tune biasa, HANYA label asli (mu=1.0 -> soft_loss
                 berbobot nol, jadi identik dengan cross-entropy murni)
    - 'kd'     : distillation penuh (hard+soft loss, seperti biasa)
 
    Keduanya dilatih dengan JUMLAH EPOCH DAN DATA YANG SAMA, satu-satunya
    perbedaan adalah ada/tidaknya bimbingan teacher -- supaya perbandingan
    adil. Ini mereplikasi Fig. 13(b) paper (mAP@0.5 vs mAP@0.5_KD): dua
    kurva akurasi berdampingan, satu tanpa KD, satu dengan KD.
 
    Return: (model_no_kd, model_kd, comparison_history) dengan
    comparison_history = {'accuracy_no_kd': [...], 'accuracy_kd': [...],
    'hard_loss_no_kd': [...], 'hard_loss_kd': [...], 'soft_loss_kd': [...]}
    """
    print("\n=== Perbandingan: TANPA KD vs DENGAN KD (paralel, titik awal sama) ===")
 
    model_no_kd = copy.deepcopy(pruned_model)
    model_kd = copy.deepcopy(pruned_model)
 
    print("\n--- Melatih baseline TANPA KD (mu=1.0, murni cross-entropy label asli) ---")
    model_no_kd, hist_no_kd = kd_train(
        model_no_kd, teacher, train_loader, epochs=epochs, lr=lr,
        mu=1.0, temperature=temperature, device=device, test_loader=test_loader,
    )
 
    print("\n--- Melatih DENGAN KD penuh (mu={:.2f}) ---".format(mu))
    model_kd, hist_kd = kd_train(
        model_kd, teacher, train_loader, epochs=epochs, lr=lr,
        mu=mu, temperature=temperature, device=device, test_loader=test_loader,
    )
 
    comparison_history = {
        "accuracy_no_kd": hist_no_kd["accuracy"],
        "accuracy_kd": hist_kd["accuracy"],
        "hard_loss_no_kd": hist_no_kd["hard_loss"],
        "hard_loss_kd": hist_kd["hard_loss"],
        "soft_loss_kd": hist_kd["soft_loss"],
    }
 
    final_no_kd = comparison_history["accuracy_no_kd"][-1] if comparison_history["accuracy_no_kd"] else None
    final_kd = comparison_history["accuracy_kd"][-1] if comparison_history["accuracy_kd"] else None
    print(f"\nHasil akhir -- TANPA KD: {final_no_kd:.4f}  |  DENGAN KD: {final_kd:.4f}")
    if final_no_kd is not None and final_kd is not None:
        print(f"Selisih murni akibat KD: {(final_kd - final_no_kd)*100:+.2f} poin persen")
 
    return model_no_kd, model_kd, comparison_history
 
 
@torch.no_grad()
def recalibrate_batchnorm(model, dataloader, num_batches=20, device="cpu"):
    """CRITICAL step after ANY pruning: refreshes each BatchNorm layer's
    running_mean/running_var to match the activation distribution of the
    PRUNED network, via a few forward passes in train() mode -- WITHOUT
    any backward pass or optimizer step (no weight updates at all).
 
    Why this matters: sparse_train() calibrates each BN's running stats
    against the FULL (unpruned) set of channels. Once prune_layerwise() /
    prune_sequential_model() physically removes some channels, the
    surviving channels' activation distribution shifts (the contribution
    of removed channels is gone) -- but the old running stats don't know
    that, so BN normalizes with stale statistics. This was confirmed
    empirically to cause severe accuracy collapse even at very small
    pruning rates (5-10%) on a harder classification task: recalibration
    alone (zero weight updates) recovered 17-52% -> 100% accuracy. This
    is a pure statistics mismatch, not lost network capacity -- always
    call this right after pruning, before evaluating accuracy or running
    sweep_pruning_rates(), and before deciding whether distillation is
    even needed.
    """
    model.train()  # BN uses batch statistics + updates running stats in this mode
    model.to(device)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()  # start fresh, don't blend with stale pre-pruning stats
    for i, (x, _) in enumerate(dataloader):
        if i >= num_batches:
            break
        model(x.to(device))
    model.eval()
    return model
 
 
def count_params(model):
    return sum(p.numel() for p in model.parameters())
 
 
@torch.no_grad()
def count_flops(model, input_size=(1, 3, 64, 64), device="cpu"):
    """Hitung FLOPs (floating point operations) untuk SATU forward pass,
    lewat forward hook -- otomatis bekerja untuk arsitektur apapun (VGG,
    TrafficSignNet, RepVGG-fused), tanpa perlu library eksternal (thop/
    ptflops/fvcore, yang tidak selalu bisa diinstal di environment
    tertentu).
 
    Formula per jenis layer (konvensi umum: 1 MAC = 2 FLOPs):
      Conv2d: FLOPs = 2 * C_in * C_out * K_h * K_w * H_out * W_out / groups
      Linear: FLOPs = 2 * in_features * out_features
 
    Return: total_flops (int). Untuk laporan biasa, bagi dengan 1e9 untuk
    dapat satuan GFLOPs (seperti kolom 'FLOPs(G)' di paper)."""
    flops = [0]
    hooks = []
 
    def conv_hook(module, inp, out):
        out_h, out_w = out.shape[2], out.shape[3]
        k_h, k_w = module.kernel_size
        c_in = module.in_channels // module.groups
        c_out = module.out_channels
        flops[0] += 2 * c_in * c_out * k_h * k_w * out_h * out_w
 
    def linear_hook(module, inp, out):
        flops[0] += 2 * module.in_features * module.out_features
 
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))
 
    model.eval()
    model.to(device)
    dummy = torch.randn(*input_size, device=device)
    model(dummy)
 
    for h in hooks:
        h.remove()
 
    return flops[0]
 
 
def measure_inference_time(model, input_size=(1, 3, 64, 64), device="cpu",
                            n_warmup=50, n_runs=500):
    """Ukur waktu inferensi RATA-RATA (dalam detik) untuk satu forward pass,
    di HARDWARE Anda sendiri saat ini (CPU/GPU manapun yang Anda pakai --
    ini SENGAJA tidak bisa saya ukurkan dari sandbox saya, karena angka ini
    HARUS dari hardware target deployment Anda supaya bermakna).
 
    Untuk GPU (device='cuda'): pakai torch.cuda.Event -- pola BAKU untuk
    benchmark CUDA, mengukur waktu LANGSUNG dari hardware GPU (bukan cuma
    wall-clock Python + torch.cuda.synchronize() seperti versi sebelumnya).
    Lebih presisi terutama untuk model kecil/cepat, di mana overhead
    Python sendiri bisa jadi signifikan relatif terhadap waktu GPU asli.
    Untuk CPU: tetap pakai time.perf_counter() (torch.cuda.Event tidak
    berlaku untuk CPU).
 
    input_size: (batch_size, C, H, W) -- UBAH batch_size pertama untuk uji
    beban lebih besar (mis. (32,3,64,64)), karena keuntungan FLOPs sering
    baru terlihat jelas di batch besar, bukan batch_size=1 (lihat diskusi
    soal GPU under-utilization di batch tunggal).
 
    n_warmup: forward pass awal yang DIBUANG dari perhitungan (CUDA/cache
    butuh 'pemanasan' -- run pertama biasanya jauh lebih lambat dari
    run-run berikutnya, bukan representasi kecepatan sebenarnya).
    n_runs: jumlah forward pass yang DIUKUR dan dirata-rata, supaya hasil
    stabil (bukan cuma sekali ukur yang rentan noise)."""
    import time
    model.eval()
    model.to(device)
    dummy = torch.randn(*input_size, device=device)
 
    with torch.no_grad():
        for _ in range(n_warmup):
            model(dummy)
 
        if device == "cuda":
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            start_evt.record()
            for _ in range(n_runs):
                model(dummy)
            end_evt.record()
            torch.cuda.synchronize()
            avg_time = start_evt.elapsed_time(end_evt) / n_runs / 1000.0  # ms -> detik
            return avg_time
        else:
            start = time.perf_counter()
            for _ in range(n_runs):
                model(dummy)
            end = time.perf_counter()
 
    avg_time = (end - start) / n_runs
    return avg_time
 
 
def load_model(checkpoint_path, device="cpu"):
    """Rebuilds a TrafficSignNet from a checkpoint saved by run_pipeline().
    Works for BOTH unpruned and pruned checkpoints, since the pruned one's
    'channels' entry reflects its actual (smaller) per-layer channel counts."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = TrafficSignNet(num_classes=ckpt["num_classes"], channels=tuple(ckpt["channels"]))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    return model, ckpt
 
 
def save_checkpoint(model, path, backbone_type, num_classes, extra=None,
                     vgg_arch=None, channels=None):
    """Generic save: works for ANY architecture this file supports
    (TrafficSignNet, VGG -- termasuk VGG yang SUDAH di-pruning). Stores
    enough metadata to reconstruct the exact architecture on load, since
    a pruned model has different per-layer channel counts than a freshly
    built VGGManual(arch=...)."""
    path = os.path.normpath(path)  # normalisasi -- toleran backslash ganda/tunggal tercampur
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "backbone_type": backbone_type,  # "trafficsignnet" or "vgg"
        "state_dict": model.state_dict(),
        "num_classes": num_classes,
        "channels": channels,            # only if backbone_type == "trafficsignnet"
    }
    if backbone_type == "vgg":
        # ALWAYS store the model's actual current cfg (reflects pruning if
        # any has happened), not just the standard arch name -- this is
        # what makes save/load correct for pruned/distilled VGG too.
        payload["vgg_cfg"] = get_vgg_cfg_from_model(model)
        payload["vgg_arch"] = vgg_arch  # kept for reference/logging only
        payload["vgg_fc_widths"] = get_vgg_fc_widths_from_model(model)  # (fc1_out, fc2_out) --
                                                                          # PENTING kalau classifier
                                                                          # sudah dipangkas via
                                                                          # prune_vgg_classifier_fc
    if extra:
        payload.update(extra)
 
    torch.save(payload, path)
    file_size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  Checkpoint disimpan ke: {path}  ({file_size_mb:.1f} MB di disk, "
          f"{count_params(model):,} params)")
 
 
def load_checkpoint(path, device="cpu"):
    """Counterpart to save_checkpoint() -- rebuilds whichever architecture
    was saved (TrafficSignNet or VGG, PRUNED or not) with the correct
    shape, then loads the weights in."""
    path = os.path.normpath(path)  # normalisasi -- toleran backslash ganda/tunggal tercampur
    if not os.path.exists(path):
        # cari folder induknya, tampilkan isi -- supaya langsung tahu nama file
        # yang BENAR-BENAR ada, bukan cuma pesan error generik dari torch.load
        parent = os.path.dirname(path)
        if os.path.isdir(parent):
            existing = os.listdir(parent)
            raise FileNotFoundError(
                f"Checkpoint tidak ditemukan: '{path}'.\n"
                f"Isi folder '{parent}' saat ini: {existing}\n"
                f"Cek apakah nama file di atas ada yang cocok -- kalau ada, "
                f"gunakan nama itu persis (termasuk huruf besar/kecil)."
            )
        else:
            raise FileNotFoundError(
                f"Checkpoint tidak ditemukan: '{path}'. "
                f"Folder induknya ('{parent}') bahkan tidak ada sama sekali -- "
                f"cek apakah CHECKPOINT_DIR sudah benar."
            )
    ckpt = torch.load(path, map_location=device)
    backbone_type = ckpt["backbone_type"]
 
    if backbone_type == "trafficsignnet":
        model = TrafficSignNet(num_classes=ckpt["num_classes"], channels=tuple(ckpt["channels"]))
    elif backbone_type == "vgg":
        # ckpt lama (sebelum ada prune_vgg_classifier_fc) tidak punya
        # 'vgg_fc_widths' -- fallback ke default (4096,4096) supaya tetap
        # kompatibel dengan checkpoint VGG yang dibuat sebelum perbaikan ini.
        fc_widths = ckpt.get("vgg_fc_widths", (4096, 4096))
        model = VGGManual(num_classes=ckpt["num_classes"], cfg=ckpt["vgg_cfg"], fc_widths=fc_widths)
    else:
        raise ValueError(f"backbone_type tidak dikenal di checkpoint: '{backbone_type}'")
 
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    file_size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  Checkpoint dimuat dari: {path} (backbone_type='{backbone_type}', "
          f"{file_size_mb:.1f} MB, {count_params(model):,} params)")
    return model, ckpt
 
 
def get_channel_counts(model):
    """Ekstrak jumlah channel per layer dari model apapun yang didukung
    file ini (VGG atau TrafficSignNet/RepVGG-fused). Return: list of
    (nama_layer, jumlah_channel), urutan sesuai forward pass -- data
    mentah untuk membuat diagram gaya Fig. 12 paper (original vs pruned
    channel per layer).
 
    Untuk VGG: menyertakan LAYER CONV *dan* fc1/fc2 (classifier) --
    penting karena prune_vgg_full() memangkas keduanya, jadi perbandingan
    yang cuma lihat conv akan melewatkan sebagian besar cerita kompresi
    (fc2 sendirian ~16.8 juta parameter di VGG16 standar)."""
    if hasattr(model, "bns"):  # TrafficSignNet / RepVGG (fused)
        return [(f"layer_{i}.conv", bn.num_features) for i, bn in enumerate(model.bns)]
    else:  # VGG
        pairs = get_conv_bn_pairs(model.features)
        counts = [(f"features.{idx}.conv", bn.num_features) for _, bn, idx in pairs]
        fc1_out, fc2_out = get_vgg_fc_widths_from_model(model)
        counts.append(("classifier.fc1", fc1_out))
        counts.append(("classifier.fc2", fc2_out))
        return counts
 
 
def compare_channel_counts(original_path_or_model, pruned_path_or_model, device="cpu"):
    """Bandingkan channel per-layer antara model SEBELUM dan SESUDAH
    pruning -- persis data yang dipakai paper untuk Fig. 12 ("Number of
    original and pruned channels"). Terima path checkpoint (string) ATAU
    objek model langsung.
 
    Return: dict {'layer_names': [...], 'original': [...], 'pruned': [...]}
    -- siap dipakai langsung sebagai 2 series di grafik bar berdampingan."""
    if isinstance(original_path_or_model, str):
        original_model, _ = load_checkpoint(original_path_or_model, device=device)
    else:
        original_model = original_path_or_model
 
    if isinstance(pruned_path_or_model, str):
        pruned_model, _ = load_checkpoint(pruned_path_or_model, device=device)
    else:
        pruned_model = pruned_path_or_model
 
    original_counts = get_channel_counts(original_model)
    pruned_counts = get_channel_counts(pruned_model)
 
    n = min(len(original_counts), len(pruned_counts))
    if len(original_counts) != len(pruned_counts):
        print(f"PERINGATAN: jumlah layer beda -- original punya {len(original_counts)} layer, "
              f"pruned punya {len(pruned_counts)} layer. Membandingkan {n} layer pertama saja "
              f"(kemungkinan besar arsitekturnya beda total, cek apakah kedua checkpoint "
              f"benar-benar dari model yang sama sebelum/sesudah pruning).")
 
    layer_names = [original_counts[i][0] for i in range(n)]
    original_vals = [original_counts[i][1] for i in range(n)]
    pruned_vals = [pruned_counts[i][1] for i in range(n)]
 
    print(f"\n{'Layer':<20} {'Original':>10} {'Pruned':>10} {'Reduksi':>10}")
    for name, o, p in zip(layer_names, original_vals, pruned_vals):
        reduksi = 100 * (1 - p / o) if o > 0 else 0
        print(f"{name:<20} {o:>10} {p:>10} {reduksi:>9.1f}%")
 
    return {"layer_names": layer_names, "original": original_vals, "pruned": pruned_vals}
 
 
@torch.no_grad()
def evaluate(model, dataloader, device="cpu"):
    model.eval()
    model.to(device)
    correct, total = 0, 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct / total if total else 0.0
 
 
@torch.no_grad()
def evaluate_detailed(model, dataloader, num_classes, device="cpu", class_names=None):
    """Versi lengkap evaluate() -- selain akurasi keseluruhan, juga hitung
    CONFUSION MATRIX dan precision/recall/F1 PER-KELAS. Dibangun manual
    (tanpa sklearn) supaya tidak perlu instalasi tambahan.
 
    PENTING: panggil ini HANYA SEKALI di test set asli, di titik yang sama
    dengan evaluasi akurasi final -- jangan dipakai untuk monitoring
    berulang selama training (sama seperti evaluate() biasa, disiplin
    val/test yang sudah kita bangun tetap berlaku di sini).
 
    Return: dict berisi 'confusion_matrix' (num_classes x num_classes,
    baris=label asli, kolom=prediksi), 'per_class' (list dict berisi
    precision/recall/f1/support per kelas), 'accuracy', 'macro_f1'."""
    model.eval()
    model.to(device)
 
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(dim=1)
        for t, p in zip(y.view(-1), preds.view(-1)):
            cm[t.long(), p.long()] += 1
 
    total = cm.sum().item()
    correct = cm.diag().sum().item()
    accuracy = correct / total if total else 0.0
 
    per_class = []
    f1_scores, precision_scores, recall_scores = [], [], []
    for c in range(num_classes):
        tp = cm[c, c].item()
        fp = cm[:, c].sum().item() - tp   # diprediksi kelas c, tapi label asli beda
        fn = cm[c, :].sum().item() - tp   # label asli kelas c, tapi diprediksi beda
        support = cm[c, :].sum().item()   # jumlah sampel asli kelas c
 
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)
        precision_scores.append(precision)
        recall_scores.append(recall)
 
        name = class_names[c] if class_names is not None and c < len(class_names) else str(c)
        per_class.append({
            "class": name, "precision": precision, "recall": recall,
            "f1": f1, "support": support,
        })
 
    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    macro_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0.0
    macro_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
 
    return {
        "confusion_matrix": cm,
        "per_class": per_class,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
    }
 
 
def print_detailed_report(result, title="LAPORAN DETAIL"):
    """Cetak hasil evaluate_detailed() dengan rapi -- tabel per-kelas +
    confusion matrix (sebagai teks, bisa di-copy ke Excel/laporan)."""
    print(f"\n{'='*70}\n{title}\n{'='*70}")
    print(f"Akurasi keseluruhan: {result['accuracy']:.4f}")
    print(f"Macro F1 (rata-rata F1 semua kelas, bobot sama): {result['macro_f1']:.4f}")
 
    print(f"\n{'Kelas':<10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    for row in result["per_class"]:
        print(f"{row['class']:<10} {row['precision']:>10.4f} {row['recall']:>10.4f} "
              f"{row['f1']:>10.4f} {row['support']:>10d}")
 
    cm = result["confusion_matrix"]
    n = cm.shape[0]
    print(f"\nConfusion Matrix (baris=label asli, kolom=prediksi):")
    header = "      " + "".join(f"{i:>5d}" for i in range(n))
    print(header)
    for i in range(n):
        row_str = "".join(f"{cm[i,j].item():>5d}" for j in range(n))
        print(f"{i:>5d} {row_str}")
 
    # kelas dengan performa terendah -- kandidat untuk dibahas di paper
    sorted_by_f1 = sorted(result["per_class"], key=lambda r: r["f1"])
    print(f"\n3 kelas dengan F1 TERENDAH (kandidat pembahasan error analysis):")
    for row in sorted_by_f1[:3]:
        print(f"  Kelas {row['class']}: F1={row['f1']:.4f} "
              f"(precision={row['precision']:.4f}, recall={row['recall']:.4f}, support={row['support']})")
 
 
def save_confusion_matrix_csv(result, path, class_names=None):
    """Simpan confusion matrix + metrik per-kelas ke CSV -- siap dipakai
    langsung sebagai tabel/gambar di paper."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cm = result["confusion_matrix"].numpy()
    n = cm.shape[0]
    labels = class_names if class_names else [str(i) for i in range(n)]
 
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_path = path.replace(".csv", "_matrix.csv")
    cm_df.to_csv(cm_path)
 
    per_class_df = pd.DataFrame(result["per_class"])
    per_class_path = path.replace(".csv", "_per_class.csv")
    per_class_df.to_csv(per_class_path, index=False)
 
    print(f"  Confusion matrix disimpan ke: {cm_path}")
    print(f"  Metrik per-kelas disimpan ke: {per_class_path}")
 
 
def iterative_prune(model, train_loader, test_loader, target_rate, n_steps=5,
                     finetune_epochs_per_step=5, sparsity_lambda=0.008,
                     min_channels_per_layer=2, device="cpu", prune_fn=None):
    """Prunes gradually instead of in one shot: cuts a small fraction of
    channels, briefly fine-tunes (so the surviving weights adapt to the
    new, smaller network), then repeats until the CUMULATIVE reduction
    reaches target_rate.
 
    prune_fn: which pruning function to use each step -- defaults to
    prune_layerwise (for TrafficSignNet). Pass prune_vgg_layerwise when
    pruning a VGG model instead.
 
    Why: a single large cut (e.g. pruning_rate=0.30 in one shot) can be a
    "shock" the network never recovers from even with distillation
    afterward, especially for a shallow network with no skip connections.
    Cutting a little at a time and letting the model adapt between cuts is
    the standard mitigation for this in the pruning literature (and is
    exactly the kind of "iterative pruning" the paper itself only prunes
    once, but this fills that gap for architectures that need it).
 
    The per-step rate is computed GEOMETRICALLY so that n_steps of pruning
    at that rate multiply out to the same overall keep-fraction as a single
    cut at target_rate would -- e.g. target_rate=0.30 over 5 steps uses a
    per-step rate of about 7% each, not 6% (0.30/5), because each step's
    cut applies to an already-shrunk layer.
    """
    if prune_fn is None:
        prune_fn = prune_layerwise
 
    keep_fraction_total = 1 - target_rate
    per_step_rate = 1 - keep_fraction_total ** (1 / n_steps)
 
    history = []
    acc0 = evaluate(model, test_loader, device=device)
    params0 = count_params(model)
    print(f"Step 0 (sebelum iterative pruning): acc={acc0:.4f}, params={params0}")
    history.append({"step": 0, "acc": acc0, "params": params0, "cumulative_reduction_pct": 0.0})
 
    for step in range(1, n_steps + 1):
        model = prune_fn(model, pruning_rate=per_step_rate,
                          min_channels_per_layer=min_channels_per_layer)
        acc_after_prune = evaluate(model, test_loader, device=device)
        params_after_prune = count_params(model)
        print(f"\nStep {step}/{n_steps}: potong (rate per-langkah={per_step_rate:.3f}) "
              f"-> acc={acc_after_prune:.4f}, params={params_after_prune}")
 
        if finetune_epochs_per_step > 0:
            model, _ = sparse_train(model, train_loader, epochs=finetune_epochs_per_step,
                                     sparsity_lambda=sparsity_lambda, device=device)
        acc_after_finetune = evaluate(model, test_loader, device=device)
        print(f"          setelah fine-tune {finetune_epochs_per_step} epoch -> acc={acc_after_finetune:.4f}")
 
        cumulative_reduction = 100 * (1 - params_after_prune / params0)
        history.append({
            "step": step, "acc_after_prune": acc_after_prune,
            "acc_after_finetune": acc_after_finetune, "params": params_after_prune,
            "cumulative_reduction_pct": cumulative_reduction,
        })
 
    final_acc = evaluate(model, test_loader, device=device)
    final_params = count_params(model)
    final_reduction = 100 * (1 - final_params / params0)
    print(f"\nHasil akhir iterative pruning: {final_params} params "
          f"({final_reduction:.1f}% reduksi dari awal), akurasi = {final_acc:.4f}")
 
    return model, history
 
 
def sweep_pruning_rates(sparse_trained_model, test_loader, rates=None,
                         min_channels_per_layer=2, device="cpu", target_acc=None,
                         recalibrate_loader=None, recalibrate_batches=20,
                         prune_fn=None, num_classes=None):
    """Finds the highest pruning_rate that keeps POST-PRUNING accuracy
    (before any distillation) above target_acc, by trying several rates
    on the SAME already-sparse-trained model. This mirrors the paper's own
    Fig. 11 methodology (they swept pruning rates and plotted mAP vs. rate
    to find their 'balance point').
 
    Cheap and fast: pruning + evaluation only, no retraining per rate --
    so this can run in seconds even though sparse_train() itself is slow.
 
    Pass in the model from right after Stage 1 (sparse_train), evaluated
    on your real held-out test set, to get numbers that reflect YOUR
    actual dataset -- architecture fragility is data-dependent, so a rate
    that's safe on a synthetic task is not guaranteed to be safe on real
    traffic sign images, and vice versa.
 
    recalibrate_loader: if given (usually your TRAINING loader), each
    pruned candidate has its BatchNorm running stats refreshed via
    recalibrate_batchnorm() before evaluation. Without this, post-pruning
    accuracy can look erratically bad even at tiny pruning rates, because
    BN running stats become stale relative to the new (smaller) input
    each layer receives after upstream channels are removed -- this is
    NOT the same thing as genuine capacity loss, and recalibration is
    nearly free (a few forward passes, no backward/optimizer step).
 
    prune_fn: fungsi pruning yang dipakai tiap rate -- default
    prune_layerwise (TrafficSignNet/RepVGG). Untuk VGG, pass
    prune_fn=prune_vgg_full (atau prune_vgg_layerwise).
 
    num_classes: kalau diisi, HITUNG JUGA macro precision/recall/F1 di
    tiap rate (pakai evaluate_detailed), bukan cuma akurasi -- data ini
    persis yang dibutuhkan untuk grafik gaya Fig. 11(b) paper (P, R, F1
    vs prune rate), berdampingan dengan Fig. 11(a) (akurasi/params vs
    prune rate) yang sudah didukung fungsi ini sejak awal."""
    if rates is None:
        rates = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
    if prune_fn is None:
        prune_fn = prune_layerwise
 
    baseline_acc = evaluate(sparse_trained_model, test_loader, device=device)
    baseline_params = count_params(sparse_trained_model)
 
    print(f"Baseline (before pruning): acc={baseline_acc:.4f}, params={baseline_params}")
    if recalibrate_loader is not None:
        print(f"(BN recalibration ENABLED: {recalibrate_batches} batches per candidate)")
    header = f"{'rate':>6} | {'post-prune acc':>15} | {'params':>10} | {'reduction':>10} | {'meets target?':>14}"
    if num_classes is not None:
        header += f" | {'precision':>10} | {'recall':>10} | {'f1':>10}"
    print(header)
 
    results = []
    best_rate_meeting_target = None
    for rate in rates:
        pruned = prune_fn(sparse_trained_model, pruning_rate=rate,
                           min_channels_per_layer=min_channels_per_layer)
        if recalibrate_loader is not None:
            pruned = recalibrate_batchnorm(pruned, recalibrate_loader,
                                            num_batches=recalibrate_batches, device=device)
 
        if num_classes is not None:
            detailed = evaluate_detailed(pruned, test_loader, num_classes=num_classes, device=device)
            acc = detailed["accuracy"]
            precision, recall, f1 = detailed["macro_precision"], detailed["macro_recall"], detailed["macro_f1"]
        else:
            acc = evaluate(pruned, test_loader, device=device)
            precision = recall = f1 = None
 
        params = count_params(pruned)
        reduction = 100 * (1 - params / baseline_params)
        meets = (target_acc is not None and acc >= target_acc)
        if meets:
            best_rate_meeting_target = rate
 
        entry = {"rate": rate, "accuracy": acc, "params": params, "reduction_pct": reduction}
        if num_classes is not None:
            entry.update({"precision": precision, "recall": recall, "f1": f1})
        results.append(entry)
 
        flag = "  <- OK" if meets else ""
        row = f"{rate:>6.2f} | {acc:>15.4f} | {params:>10} | {reduction:>9.1f}% | {'yes' if meets else 'no':>14}"
        if num_classes is not None:
            row += f" | {precision:>10.4f} | {recall:>10.4f} | {f1:>10.4f}"
        print(row + flag)
 
    if target_acc is not None:
        if best_rate_meeting_target is not None:
            print(f"\nHighest rate keeping accuracy >= {target_acc:.2f}: "
                  f"pruning_rate={best_rate_meeting_target}")
        else:
            print(f"\nNo tested rate kept accuracy >= {target_acc:.2f}. "
                  f"Try rates below {min(rates)}, or increase sparse_epochs "
                  f"before pruning (better gamma separation can sometimes help), "
                  f"or accept a lower post-pruning accuracy and rely on "
                  f"distillation to recover it afterward.")
 
    return results
 
 
# =============================================================================
# BAGIAN 2: KONFIGURASI -- INI YANG PERLU ANDA ADJUST
# =============================================================================
 
# --- 1. Lokasi data ---
TRAIN_DIR = r"D:\Documents\Computer Eng\PhD\Lecture PhD\Python\Special Topic AI\data\ArabicTrafficSigns\train"
TRAIN_CSV = r"D:\Documents\Computer Eng\PhD\Lecture PhD\Python\Special Topic AI\data\ArabicTrafficSigns\Labels_Train_Arabic_Traffic_Signs_24_2200.csv"
TEST_DIR = r"D:\Documents\Computer Eng\PhD\Lecture PhD\Python\Special Topic AI\data\ArabicTrafficSigns\test"
TEST_CSV = r"D:\Documents\Computer Eng\PhD\Lecture PhD\Python\Special Topic AI\data\ArabicTrafficSigns\Labels_Test_Arabic_Traffic_Signs_24_518.csv"
 
IMAGE_SIZE = 64
BATCH_SIZE = 32
VAL_SPLIT_RATIO = 0.15   # 15% dari data TRAIN dipecah jadi validation, untuk
                          # monitoring/keep_best/iterative_prune/sweep -- TEST
                          # set asli tidak pernah dipakai untuk ini, cuma untuk
                          # laporan akhir SEKALI. Lihat diskusi soal integritas
                          # akademik/data leakage kalau test dipakai berulang.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
 
# --- 2. Arsitektur model ---
# "plain"  = TrafficSignNet biasa, langsung dari bobot acak.
# "repvgg" = latih RepVGG (multi-branch) dulu, lalu fusi -- biasanya
#            akurasi awal (sebelum sparse training) lebih tinggi.
BACKBONE = "vgg16"          # bisa huruf besar/kecil bebas: "repvgg", "plain", "vgg16", "vgg11"
REPVGG_EPOCHS = 30          # hanya dipakai kalau BACKBONE == "repvgg"
CHANNELS = (3, 64, 128, 256, 256)   # dilebarkan dari (3,32,64,128,128) -- terbukti +15pp di tes kita
VGG_PRETRAINED = True       # pakai bobot ImageNet pretrained (butuh akses internet ke download.pytorch.org)
 
# --- 2b. Simpan/muat checkpoint (penting untuk fine-tune VGG dulu SEBELUM
#     sparsity+pruning -- lihat penjelasan soal urutan yang benar secara
#     akademik) ---
# --- 2b. Simpan/muat checkpoint (penting untuk fine-tune VGG dulu SEBELUM
#     sparsity+pruning -- lihat penjelasan soal urutan yang benar secara
#     akademik). SEKARANG OTOMATIS TERISI (bukan None) -- supaya checkpoint
#     SELALU tersimpan di tiap tahap penting, tidak bergantung Anda ingat
#     mengisinya atau menangkap return value main(). Ini mencegah kejadian
#     "training 30 epoch tapi lupa simpan, lalu tidak bisa diselamatkan".
CHECKPOINT_DIR = r"D:\Documents\Computer Eng\PhD\Lecture PhD\Comprehensive\Knowledge Destilation\Disertasi\Paper\checkpoints"
SAVE_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "1_finetuned.pt")   # HANYA ditulis saat FOCUS_ACCURACY_ONLY=True
                                                                          # (hasil fine-tune MURNI -- akurasi tertinggi Anda)
SAVE_SPARSE_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "1b_sparse_trained.pt")  # ditulis setelah Stage 1
                                                                          # sparse training (FOCUS_ACCURACY_ONLY=False) --
                                                                          # nama BEDA dari SAVE_CHECKPOINT_PATH supaya
                                                                          # checkpoint fine-tune murni Anda TIDAK PERNAH
                                                                          # tertimpa oleh hasil sparse training berikutnya
LOAD_CHECKPOINT_PATH = None   # isi manual kalau mau SKIP Stage 0-1 dan muat checkpoint yang sudah ada
SAVE_PRUNED_PATH = os.path.join(CHECKPOINT_DIR, "2_pruned.pt")          # SELALU disimpan setelah Stage 2
SAVE_DISTILLED_PATH = os.path.join(CHECKPOINT_DIR, "3_distilled.pt")    # SELALU disimpan setelah Stage 3
CONFUSION_MATRIX_SAVE_PATH = os.path.join(CHECKPOINT_DIR, "confusion_matrix.csv")  # simpan confusion
                                                                          # matrix + metrik per-kelas
                                                                          # sebagai CSV (siap tabel di paper)
SPARSE_LOSS_HISTORY_SAVE_PATH = os.path.join(CHECKPOINT_DIR, "stage1b_loss_history.csv")  # loss per
                                                                          # epoch Stage 1 -- untuk grafik
GAMMA_DISTRIBUTION_SAVE_PATH = os.path.join(CHECKPOINT_DIR, "gamma_distribution.csv")  # nilai |gamma|
                                                                          # mentah tiap channel/layer --
                                                                          # untuk histogram distribusi gamma
 
# --- 0. FOKUS AKURASI DULU (sebelum pruning) ---
# Kalau True: TIDAK akan lanjut ke Stage 2 (pruning) sama sekali -- cuma
# jalankan Stage 0 + Stage 1, laporkan akurasi, lalu berhenti. Juga MATIKAN
# sparsity penalty selama fase ini (SPARSITY_LAMBDA diabaikan, dipaksa 0),
# karena penalti itu hanya diperlukan untuk PERSIAPAN pruning nanti --
# kita sudah buktikan penalti ini menahan akurasi sekitar 2pp, jadi tidak
# ada gunanya dinyalakan kalau belum mau pruning. Nyalakan augmentasi data
# juga, karena dataset training kita kecil (~90 gambar/kelas).
FOCUS_ACCURACY_ONLY = True    # AKTIFKAN untuk VGG -- fine-tune dulu sampai stabil
                                # SEBELUM sparsity penalty dinyalakan (lihat penjelasan
                                # soal urutan yang benar secara akademik untuk model pretrained)
USE_AUGMENTATION = True
 
# --- 3. Sparse training (Stage 1) -- ini yang MENENTUKAN kualitas pemisahan
#     gamma (channel penting vs tidak penting) sebelum pruning dilakukan.
#     Kalau nanti gamma diagnostics menunjukkan "LOW SEPARATION", coba
#     naikkan SPARSE_EPOCHS atau SPARSITY_LAMBDA.
#     (Diabaikan/dipaksa 0 kalau FOCUS_ACCURACY_ONLY=True, lihat di atas.)
SPARSE_EPOCHS = 30
SPARSITY_LAMBDA = 0.008     # lambda di Eq. 10 paper (default paper: 0.008)
SPARSE_LR = 1e-3
 
# --- 4. Pruning (Stage 2) -- parameter INTI yang paling menentukan
#     seberapa besar drop akurasi setelah pruning.
PRUNING_METHOD = "layerwise"   # "layerwise" (disarankan) atau "global"
MIN_CHANNELS_PER_LAYER = 8     # batas minimum channel per layer, agar
                                # tidak ada layer yang terpotong habis
 
# Mode A: uji SATU pruning_rate spesifik
SINGLE_RATE_MODE = False
SINGLE_PRUNING_RATE = 0.20
 
# Mode B: SWEEP banyak pruning_rate sekaligus, cari rate yang memenuhi
#         target akurasi (misal 0.75-0.80 sesuai concern Anda)
SWEEP_MODE = False
SWEEP_RATES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
TARGET_ACCURACY = 0.75         # akurasi minimum SETELAH pruning yang diinginkan
 
# Mode C: ITERATIVE -- potong sedikit-sedikit + fine-tune tiap langkah,
#         bukan sekali potong besar. Terbukti jauh lebih tahan drop
#         dibanding one-shot di rate/target yang sama.
ITERATIVE_MODE = True
ITERATIVE_TARGET_RATE = 0.58       # reduksi TOTAL yang ingin dicapai di akhir
ITERATIVE_N_STEPS = 10              # jumlah langkah kecil (lebih banyak = lebih halus, lebih lama)
ITERATIVE_FINETUNE_EPOCHS = 10      # epoch fine-tune di ANTARA tiap langkah potong
 
# --- 5. Knowledge Distillation (Stage 3) -- setelah pruning selesai (mode
#     apapun di atas), latih ulang model hasil pruning (student) dibimbing
#     oleh model SEBELUM pruning (teacher), untuk memulihkan/mendorong
#     akurasi lebih tinggi lagi. Eq. 11 paper.
DISTILLATION_MODE = True
KD_EPOCHS = 30
KD_LR = 1e-4              # SENGAJA lebih kecil dari SPARSE_LR (1e-3) -- distillation
                          # menyempurnakan model yang SUDAH akurat, bukan training dari
                          # nol, jadi butuh langkah kecil supaya tidak "menendang" model
                          # keluar dari solusi bagus yang sudah ditemukan sebelumnya.
KD_MU = 0.5              # bobot hard-label loss vs soft-label loss (paper: tidak spesifik, 0.5 = seimbang)
KD_TEMPERATURE = 4.0     # temperature softmax untuk soft label
 
COMPARE_KD_MODE = False  # True = latih DUA model paralel (tanpa KD vs dengan KD)
                          # untuk grafik perbandingan gaya Fig. 13(b) paper.
                          # Butuh 2x waktu training Stage 3 (dua model dilatih).
 
# =============================================================================
# END KONFIGURASI -- tidak perlu diubah di bawah ini
# =============================================================================
 
 
def main():
    print("=" * 70)
    print("PRUNING STUDY")
    print("=" * 70)
 
    # --- load data: TRAIN dipecah jadi train+val (untuk monitoring selama
    #     training/pruning/distillation -- TIDAK PERNAH pakai test asli).
    #     TEST_DIR/TEST_CSV dimuat terpisah dan HANYA dipakai SEKALI, di
    #     akhir segalanya, untuk laporan final yang jujur secara metodologis
    #     (lihat diskusi soal data leakage kalau test dipakai berulang
    #     untuk keep_best/monitoring). ---
    train_loader, val_loader, num_classes, label_to_idx = split_train_val(
        TRAIN_DIR, TRAIN_CSV, image_size=IMAGE_SIZE, val_ratio=VAL_SPLIT_RATIO,
        augment_train=USE_AUGMENTATION, batch_size=BATCH_SIZE,
    )
 
    test_ds = ArabicTrafficSignsDataset(TEST_DIR, TEST_CSV, image_size=IMAGE_SIZE,
                                         augment=False)  # JANGAN augment test set
    test_ds.label_to_idx = label_to_idx
    test_ds.num_classes = num_classes
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
 
    print(f"\nKelas terdeteksi: {num_classes}")
    print(f"Val samples: {len(val_loader.dataset)}  |  "
          f"Test samples (HANYA disentuh di akhir): {len(test_ds)}")
 
    class _TrainDsShim:
        """Kompatibilitas: kode di bawah masih baca train_ds.num_classes di
        beberapa tempat -- objek kecil ini menyediakan atribut itu tanpa
        perlu ubah semua pemanggilan satu-satu."""
        pass
    train_ds = _TrainDsShim()
    train_ds.num_classes = num_classes
    train_ds.label_to_idx = label_to_idx
 
    # --- Stage 0 (opsional): RepVGG / VGG / muat dari checkpoint ---
    backbone_normalized = BACKBONE.strip().lower()  # terima "repvgg", "REPVGG", "RepVGG", dst.
    is_vgg = backbone_normalized in ("vgg16", "vgg16_bn", "vgg11", "vgg11_bn")
 
    if LOAD_CHECKPOINT_PATH is not None:
        print(f"\n--- Stage 0: muat checkpoint dari '{LOAD_CHECKPOINT_PATH}' (Stage 0 training DILEWATI) ---")
        model, _ckpt = load_checkpoint(LOAD_CHECKPOINT_PATH, device=DEVICE)
        is_vgg = _ckpt["backbone_type"] == "vgg"
    elif backbone_normalized == "repvgg":
        print(f"\n--- Stage 0: latih RepVGG ({REPVGG_EPOCHS} epoch), lalu fusi ---")
        model = train_repvgg_and_fuse(
            num_classes=train_ds.num_classes, dataloader=train_loader,
            channels=CHANNELS, epochs=REPVGG_EPOCHS, device=DEVICE,
        )
    elif backbone_normalized == "plain":
        model = TrafficSignNet(num_classes=train_ds.num_classes, channels=CHANNELS)
        model.to(DEVICE)
    elif is_vgg:
        vgg_arch = "vgg16_bn" if "16" in backbone_normalized else "vgg11_bn"
        print(f"\n--- Stage 0: load {vgg_arch} (pretrained={VGG_PRETRAINED}) ---")
        model = build_vgg_classifier(num_classes=train_ds.num_classes,
                                      arch=vgg_arch, pretrained=VGG_PRETRAINED)
        model.to(DEVICE)
    else:
        raise ValueError(f"BACKBONE tidak dikenal: '{BACKBONE}' "
                          f"(pakai 'plain', 'repvgg', 'vgg16', atau 'vgg11')")
 
    # fungsi pruning yang dipakai tergantung arsitektur -- VGG butuh versi
    # umum (prune_vgg_layerwise) karena strukturnya beda dari TrafficSignNet
    # prune_vgg_full memangkas conv DAN fc classifier (fc1/fc2) -- jauh
    # lebih efektif untuk VGG karena classifier menyumbang ~89% total
    # parameter (fc2 sendiri 16.8 juta, tidak tersentuh kalau cuma
    # pangkas conv saja). Ganti ke prune_vgg_layerwise kalau ingin
    # kembali ke perilaku lama (cuma pangkas conv).
    prune_fn = prune_vgg_full if is_vgg else prune_layerwise
 
    params_before_sparse = count_params(model)
    acc_before_sparse = evaluate(model, val_loader, device=DEVICE)
    print(f"\nModel awal: {params_before_sparse} params, akurasi (VALIDATION) = {acc_before_sparse:.4f}")
 
    # --- SIMPAN TEACHER ASLI sebelum sparse training merusak akurasinya --
    # sparse training (Stage 1, dengan sparsity penalty) TERBUKTI menurunkan
    # akurasi model (contoh nyata: 99.03% -> 92.08%), karena tujuannya
    # memisahkan channel penting/tidak penting untuk pruning, BUKAN untuk
    # akurasi murni. Kalau model yang sudah terdegradasi ini dipakai sebagai
    # teacher di Stage 3, distillation cuma bisa mengejar akurasi teacher
    # yang sudah rendah itu -- bukan akurasi terbaik yang pernah dicapai.
    # Simpan salinan TERPISAH di sini (sebelum sparse_train dijalankan),
    # supaya Stage 3 bisa pakai teacher yang akurasinya masih maksimal.
    if not FOCUS_ACCURACY_ONLY:
        print(f"\nMenyimpan salinan model SEBELUM sparse training (akurasi {acc_before_sparse:.4f}) "
              f"untuk dipakai sebagai teacher distillation nanti...")
        best_teacher_model = copy.deepcopy(model)
        best_teacher_acc = acc_before_sparse
    else:
        best_teacher_model = None
        best_teacher_acc = None
 
    # --- Stage 1: sparse training ---
    effective_sparsity_lambda = 0.0 if FOCUS_ACCURACY_ONLY else SPARSITY_LAMBDA
    print(f"\n--- Stage 1: {'training murni (fokus akurasi, TANPA sparsity penalty)' if FOCUS_ACCURACY_ONLY else 'sparse training'} "
          f"({SPARSE_EPOCHS} epoch, lambda={effective_sparsity_lambda}) ---")
    model, sparse_loss_history = sparse_train(model, train_loader, epochs=SPARSE_EPOCHS,
                                               lr=SPARSE_LR, sparsity_lambda=effective_sparsity_lambda, device=DEVICE)
    acc_after_sparse = evaluate(model, val_loader, device=DEVICE)
    print(f"\nAkurasi setelah Stage 1 (VALIDATION): {acc_after_sparse:.4f}")
 
    if SPARSE_LOSS_HISTORY_SAVE_PATH is not None:
        os.makedirs(os.path.dirname(SPARSE_LOSS_HISTORY_SAVE_PATH) or ".", exist_ok=True)
        pd.DataFrame({"epoch": list(range(1, len(sparse_loss_history) + 1)),
                      "loss": sparse_loss_history}).to_csv(SPARSE_LOSS_HISTORY_SAVE_PATH, index=False)
        print(f"  Loss history Stage 1 disimpan ke: {SPARSE_LOSS_HISTORY_SAVE_PATH}")
 
    if GAMMA_DISTRIBUTION_SAVE_PATH is not None and not FOCUS_ACCURACY_ONLY:
        export_gamma_distribution(model, GAMMA_DISTRIBUTION_SAVE_PATH)
 
    def _save_stage1_checkpoint():
        """Selalu dipanggil setelah Stage 1 selesai, apapun nilai
        FOCUS_ACCURACY_ONLY -- supaya hasil training tidak pernah hilang
        begitu saja kalau Stage 2/3 nanti crash atau sesi terputus.
 
        PENTING: pilih path BERBEDA tergantung FOCUS_ACCURACY_ONLY, supaya
        checkpoint fine-tune MURNI (akurasi tertinggi, SAVE_CHECKPOINT_PATH)
        tidak pernah tertimpa oleh hasil SPARSE TRAINING (akurasi lebih
        rendah, SAVE_SPARSE_CHECKPOINT_PATH) -- keduanya dulu memakai nama
        file yang SAMA, sehingga checkpoint terbaik bisa hilang tertimpa
        begitu Anda lanjut ke Stage 2/3 di run berikutnya."""
        save_path = SAVE_CHECKPOINT_PATH if FOCUS_ACCURACY_ONLY else SAVE_SPARSE_CHECKPOINT_PATH
        if save_path is None:
            return
        print(f"\nMenyimpan checkpoint Stage 1 (otomatis, supaya tidak hilang)...")
        if is_vgg:
            vgg_arch_saved = "vgg16_bn" if "16" in backbone_normalized else "vgg11_bn"
            save_checkpoint(model, save_path, backbone_type="vgg",
                             num_classes=train_ds.num_classes, vgg_arch=vgg_arch_saved)
        else:
            save_checkpoint(model, save_path, backbone_type="trafficsignnet",
                             num_classes=train_ds.num_classes, channels=list(CHANNELS))
 
    _save_stage1_checkpoint()  # <-- SELALU jalan, baik FOCUS_ACCURACY_ONLY True maupun False
 
    if FOCUS_ACCURACY_ONLY:
        # Ini titik di mana model biasanya dilaporkan sebagai hasil (mis.
        # checkpoint fine-tune terbaik Anda) -- jadi test set asli
        # dievaluasi SEKALI di sini, terpisah dari angka validation di atas.
        acc_test_final = evaluate(model, test_loader, device=DEVICE)
        print("\n" + "=" * 70)
        print(f"FOCUS_ACCURACY_ONLY aktif -- BERHENTI DI SINI (Stage 2/pruning DILEWATI).")
        print(f"Akurasi validation (dipakai selama training): {acc_after_sparse:.4f}")
        print(f"Akurasi TEST ASLI (dievaluasi SEKALI, ini yang dilaporkan): {acc_test_final:.4f}")
        print(f"Checkpoint sudah tersimpan di: {SAVE_CHECKPOINT_PATH}")
        print(f"Untuk lanjut ke pruning nanti: set LOAD_CHECKPOINT_PATH="
              f"'{SAVE_CHECKPOINT_PATH}' dan FOCUS_ACCURACY_ONLY=False di run berikutnya.")
        print("=" * 70)
        return model, acc_test_final
 
    print_gamma_diagnostics(model)
 
    # --- Stage 2 & 3: dibungkus try/except -- kalau CRASH di sini (misal
    #     CUDA out-of-memory, koneksi terputus), checkpoint Stage 1 yang
    #     mahal (30 epoch) SUDAH aman tersimpan (lihat _save_stage1_checkpoint
    #     di atas). Pesan ini memberitahu cara lanjut TANPA mengulang Stage 1.
    try:
        return _run_stage2_and_3(
            model, train_ds, train_loader, val_loader, test_loader, is_vgg, backbone_normalized,
            params_before_sparse, acc_after_sparse, prune_fn, DEVICE,
            best_teacher_model, best_teacher_acc,
        )
    except Exception as e:
        print(f"\n{'!'*70}")
        print(f"ERROR di Stage 2/3: {type(e).__name__}: {e}")
        print(f"Checkpoint Stage 1 (sparse training) SUDAH AMAN tersimpan di:")
        print(f"  {SAVE_SPARSE_CHECKPOINT_PATH}")
        print(f"Untuk lanjut TANPA mengulang Stage 1, set:")
        print(f"  LOAD_CHECKPOINT_PATH = r'{SAVE_SPARSE_CHECKPOINT_PATH}'")
        print(f"  FOCUS_ACCURACY_ONLY = False")
        print(f"lalu jalankan main() lagi -- akan langsung lompat ke Stage 2.")
        print(f"{'!'*70}")
        raise
 
 
def _run_stage2_and_3(model, train_ds, train_loader, val_loader, test_loader, is_vgg,
                       backbone_normalized, params_before_sparse, acc_after_sparse,
                       prune_fn, DEVICE, best_teacher_model=None, best_teacher_acc=None):
    # PENTING: val_loader dipakai untuk SEMUA monitoring/keep_best/iterative
    # pruning/sweep di bawah -- test_loader (test set asli) BARU disentuh
    # SATU KALI, di paling akhir tiap jalur, untuk laporan final yang jujur.
 
    # teacher untuk distillation: pakai model SEBELUM sparse training kalau
    # ada (akurasinya lebih tinggi), fallback ke `model` biasa kalau tidak
    teacher_for_kd = best_teacher_model if best_teacher_model is not None else model
    if best_teacher_model is not None:
        print(f"\n(Distillation nanti akan pakai teacher akurasi {best_teacher_acc:.4f} "
              f"-- SEBELUM sparse training, bukan model yang sudah terdegradasi "
              f"ke {acc_after_sparse:.4f})")
 
    # --- Stage 2: pruning study ---
    if ITERATIVE_MODE:
        print(f"\n--- Stage 2: ITERATIVE pruning (target={ITERATIVE_TARGET_RATE}, "
              f"{ITERATIVE_N_STEPS} langkah, fine-tune {ITERATIVE_FINETUNE_EPOCHS} epoch/langkah) ---")
        pruned_model, history = iterative_prune(
            model, train_loader, val_loader,
            target_rate=ITERATIVE_TARGET_RATE, n_steps=ITERATIVE_N_STEPS,
            finetune_epochs_per_step=ITERATIVE_FINETUNE_EPOCHS,
            sparsity_lambda=SPARSITY_LAMBDA, min_channels_per_layer=MIN_CHANNELS_PER_LAYER,
            device=DEVICE, prune_fn=prune_fn,
        )
        print(f"\n[Ukuran model SETELAH pruning] {count_params(pruned_model):,} params "
              f"({100*(1-count_params(pruned_model)/params_before_sparse):.1f}% reduksi dari sebelum pruning)")
 
        if SAVE_PRUNED_PATH is not None:
            if is_vgg:
                save_checkpoint(pruned_model, SAVE_PRUNED_PATH, backbone_type="vgg",
                                 num_classes=train_ds.num_classes)
            else:
                pruned_channels = [pruned_model.convs[0].in_channels] + \
                                   [bn.num_features for bn in pruned_model.bns]
                save_checkpoint(pruned_model, SAVE_PRUNED_PATH, backbone_type="trafficsignnet",
                                 num_classes=train_ds.num_classes, channels=pruned_channels)
 
        if DISTILLATION_MODE:
            # teacher_for_kd = model SEBELUM sparse training (akurasi lebih
            # tinggi) kalau tersedia, bukan `model` yang sudah terdegradasi
            # sparsity penalty -- lihat penjelasan di atas fungsi ini.
 
            if COMPARE_KD_MODE:
                model_no_kd, model_kd, kd_comparison = compare_kd_vs_no_kd(
                    pruned_model, teacher_for_kd, train_loader, val_loader,
                    epochs=KD_EPOCHS, lr=KD_LR, mu=KD_MU, temperature=KD_TEMPERATURE,
                    device=DEVICE,
                )
                pruned_model = model_kd  # yang dipakai untuk save/ringkasan selanjutnya = versi DENGAN KD
                acc_before_kd = kd_comparison["accuracy_kd"][0] if kd_comparison["accuracy_kd"] else None
                acc_after_kd = kd_comparison["accuracy_kd"][-1] if kd_comparison["accuracy_kd"] else None
                kd_history = kd_comparison
            else:
                print(f"\n--- Stage 3: knowledge distillation "
                      f"(mu={KD_MU}, T={KD_TEMPERATURE}, {KD_EPOCHS} epoch) ---")
                acc_before_kd = evaluate(pruned_model, val_loader, device=DEVICE)
                pruned_model, kd_history = kd_train(pruned_model, teacher_for_kd, train_loader,
                                                     epochs=KD_EPOCHS, lr=KD_LR, mu=KD_MU, temperature=KD_TEMPERATURE,
                                                     device=DEVICE, test_loader=val_loader)
                acc_after_kd = evaluate(pruned_model, val_loader, device=DEVICE)
                print(f"\nAkurasi sebelum distillation (VALIDATION): {acc_before_kd:.4f}")
                print(f"Akurasi SETELAH distillation (VALIDATION):  {acc_after_kd:.4f}")
                print(f"[Ukuran model SETELAH distillation] {count_params(pruned_model):,} params "
                      f"(sama seperti setelah pruning -- distillation tidak mengubah ukuran, hanya bobot)")
 
            if SAVE_DISTILLED_PATH is not None:
                if is_vgg:
                    save_checkpoint(pruned_model, SAVE_DISTILLED_PATH, backbone_type="vgg",
                                     num_classes=train_ds.num_classes)
                else:
                    distilled_channels = [pruned_model.convs[0].in_channels] + \
                                          [bn.num_features for bn in pruned_model.bns]
                    save_checkpoint(pruned_model, SAVE_DISTILLED_PATH, backbone_type="trafficsignnet",
                                     num_classes=train_ds.num_classes, channels=distilled_channels)
 
        # --- LAPORAN FINAL YANG JUJUR: test set asli dievaluasi SATU KALI,
        #     di sini, setelah SEMUA keputusan (pruning rate, keep_best,
        #     dst) sudah selesai dibuat berdasarkan validation. Angka ini
        #     -- bukan angka validation manapun di atas -- yang seharusnya
        #     dilaporkan sebagai akurasi final di skripsi/paper Anda. ---
        detailed_result = evaluate_detailed(pruned_model, test_loader,
                                             num_classes=train_ds.num_classes, device=DEVICE)
        acc_final_test = detailed_result["accuracy"]
        print_detailed_report(detailed_result, title="HASIL AKHIR DI TEST SET ASLI (dievaluasi SEKALI)")
        if CONFUSION_MATRIX_SAVE_PATH is not None:
            save_confusion_matrix_csv(detailed_result, CONFUSION_MATRIX_SAVE_PATH)
 
        print(f"\n{'='*70}\nRINGKASAN UKURAN MODEL DI SETIAP TAHAP\n{'='*70}")
        print(f"  Sebelum pruning (Stage 1): {params_before_sparse:,} params")
        print(f"  Setelah pruning (Stage 2): {count_params(pruned_model):,} params "
              f"({100*(1-count_params(pruned_model)/params_before_sparse):.1f}% reduksi)")
        if DISTILLATION_MODE:
            print(f"  Setelah distillation (Stage 3): {count_params(pruned_model):,} params (ukuran sama dgn Stage 2)")
            return pruned_model, history, kd_history  # kd_history: riwayat hard_loss/soft_loss/accuracy per epoch,
                                                        # untuk membuat grafik gaya Fig. 13 paper
 
        return pruned_model, history
 
    if SWEEP_MODE:
        print(f"\n--- Stage 2: SWEEP pruning_rate (target akurasi >= {TARGET_ACCURACY}, di VALIDATION) ---")
        print("(Catatan: mode SWEEP untuk eksplorasi -- tidak menghasilkan satu model "
              "final, jadi DISTILLATION_MODE diabaikan di sini. Pakai SINGLE_RATE_MODE "
              "atau ITERATIVE_MODE kalau sudah tahu rate yang mau dipakai + distillation.)")
        results = sweep_pruning_rates(
            model, val_loader, rates=SWEEP_RATES,
            min_channels_per_layer=MIN_CHANNELS_PER_LAYER,
            device=DEVICE, target_acc=TARGET_ACCURACY,
        )
        return results
 
    if SINGLE_RATE_MODE:
        print(f"\n--- Stage 2: pruning tunggal (rate={SINGLE_PRUNING_RATE}, method='{PRUNING_METHOD}') ---")
        pruning_method_normalized = PRUNING_METHOD.strip().lower()  # terima "Layerwise", "GLOBAL", dst.
        if is_vgg:
            if pruning_method_normalized == "global":
                print("Catatan: metode 'global' belum didukung untuk VGG -- pakai 'layerwise' (conv+fc).")
            pruned = prune_vgg_full(model, pruning_rate=SINGLE_PRUNING_RATE,
                                     min_channels_per_layer=MIN_CHANNELS_PER_LAYER)
        elif pruning_method_normalized == "layerwise":
            pruned = prune_layerwise(model, pruning_rate=SINGLE_PRUNING_RATE,
                                      min_channels_per_layer=MIN_CHANNELS_PER_LAYER)
        elif pruning_method_normalized == "global":
            threshold = compute_global_threshold(model, pruning_rate=SINGLE_PRUNING_RATE)
            pruned = prune_sequential_model(model, threshold,
                                             min_channels_per_layer=MIN_CHANNELS_PER_LAYER)
        else:
            raise ValueError(f"PRUNING_METHOD tidak dikenal: '{PRUNING_METHOD}'")
 
        params_after = count_params(pruned)
        acc_after = evaluate(pruned, val_loader, device=DEVICE)
        reduction = 100 * (1 - params_after / params_before_sparse)
 
        print(f"\nHasil pruning tunggal:")
        print(f"  Params: {params_before_sparse} -> {params_after} ({reduction:.1f}% reduksi)")
        print(f"  Akurasi (VALIDATION): {acc_after_sparse:.4f} -> {acc_after:.4f}")
        if hasattr(pruned, "bns"):
            channels_retained = [bn.num_features for bn in pruned.bns]
            print(f"  Channel tersisa per layer: {channels_retained}")
 
        if DISTILLATION_MODE:
            print(f"\n--- Stage 3: knowledge distillation "
                  f"(mu={KD_MU}, T={KD_TEMPERATURE}, {KD_EPOCHS} epoch) ---")
            pruned, kd_history = kd_train(pruned, teacher_for_kd, train_loader,
                                           epochs=KD_EPOCHS, lr=KD_LR, mu=KD_MU, temperature=KD_TEMPERATURE,
                                           device=DEVICE, test_loader=val_loader)
            acc_after_kd = evaluate(pruned, val_loader, device=DEVICE)
            print(f"\nAkurasi SETELAH distillation (VALIDATION): {acc_after_kd:.4f}")
 
            detailed_result = evaluate_detailed(pruned, test_loader,
                                                 num_classes=train_ds.num_classes, device=DEVICE)
            acc_test_final = detailed_result["accuracy"]
            print_detailed_report(detailed_result, title="HASIL AKHIR DI TEST SET ASLI (dievaluasi SEKALI)")
            if CONFUSION_MATRIX_SAVE_PATH is not None:
                save_confusion_matrix_csv(detailed_result, CONFUSION_MATRIX_SAVE_PATH)
            return pruned, kd_history
 
        detailed_result = evaluate_detailed(pruned, test_loader,
                                             num_classes=train_ds.num_classes, device=DEVICE)
        acc_test_final = detailed_result["accuracy"]
        print_detailed_report(detailed_result, title="HASIL AKHIR DI TEST SET ASLI (dievaluasi SEKALI)")
        if CONFUSION_MATRIX_SAVE_PATH is not None:
            save_confusion_matrix_csv(detailed_result, CONFUSION_MATRIX_SAVE_PATH)
        return pruned
 
    print("\nTidak ada mode yang aktif -- set SINGLE_RATE_MODE atau SWEEP_MODE ke True.")
    return None

if __name__ == "__main__":
    LOAD_CHECKPOINT_PATH = r"D:\Documents\Computer Eng\PhD\Lecture PhD\Comprehensive\Knowledge Destilation\Disertasi\Paper\checkpoints\hasil_bagus\1_finetuned.pt"  # sesuaikan path pastinya dari hasil dir /s /b
    FOCUS_ACCURACY_ONLY = False
    model, history, kd_history = main()
# ============================================================================
# CATATAN PENTING UNTUK PENGGUNA JUPYTER:
#
# Baris "if __name__ == '__main__': main()" SENGAJA DIHAPUS dari sini.
#
# Di Jupyter, __name__ selalu "__main__" di dalam sel apapun -- jadi kalau
# baris itu ada, main() akan OTOMATIS terpanggil setiap kali sel INI
# (yang berisi seluruh definisi file) dijalankan/dijalankan ulang, memakai
# nilai konfigurasi yang tertanam DI DALAM PASTE INI -- bukan nilai yang
# sudah Anda edit di sel lain. Ini menyebabkan training/download terulang
# tanpa sengaja setiap kali sel definisi ini di-run ulang.
#
# CARA PAKAI YANG BENAR DI JUPYTER:
#   Sel 1 (sel ini): definisi semua class/fungsi/konfigurasi -- jalankan
#                     SEKALI saja setelah kernel start/restart.
#   Sel 2 (sel BARU, terpisah): edit config yang perlu diubah (misal
#                     LOAD_CHECKPOINT_PATH, FOCUS_ACCURACY_ONLY), lalu
#                     panggil main() secara EKSPLISIT:
#
#       LOAD_CHECKPOINT_PATH = r"...\1_finetuned.pt"
#       FOCUS_ACCURACY_ONLY = False
#       model, hasil = main()
#
#   Untuk run berikutnya, CUKUP edit & jalankan ulang Sel 2 -- JANGAN
#   jalankan ulang Sel 1 kecuali kernel benar-benar di-restart, karena
#   itu akan mendefinisikan ulang variabel config ke nilai default di
#   file ini (menimpa perubahan yang sudah Anda buat di Sel 2).
# ============================================================================
