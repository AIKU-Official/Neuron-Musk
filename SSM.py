#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S4-based spike clustering (Real S4Block, models/s4/s4.py import):
- Input: spike windows (N, C, T) cut around detected peaks
- Model: Conv1d stem -> S4Block stack -> global pooling -> embedding
- Train: self-supervised (InfoNCE); optionally supervised-contrastive if GT labels given
- Cluster: HDBSCAN (if available) / KMeans
- Eval: ARI / NMI vs. GT neuron IDs

If you already have spike windows from SpikeInterface:
  waveforms_sorted: np.ndarray, shape (N, C, T)
  labels_sorted   : np.ndarray, shape (N,) neuron IDs  (if available)
  timestamps_sorted: np.ndarray, shape (N,) sample idx  (optional)
Plug them into `train_and_cluster()` instead of dummy generation.

Requires: numpy, torch, scikit-learn
Optional: hdbscan
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================
# Real S4 import
#   레포 트리: models/s4/s4.py
# =========================================================
from models.s4.s4 import S4Block  # ★ 고정 임포트

# ---------------------------
# Optional clustering deps
# ---------------------------
HDBSCAN_AVAILABLE = True
try:
    import hdbscan
except Exception:
    HDBSCAN_AVAILABLE = False
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# ---------------------------
# SpikeInterface / SpikeForest
# ---------------------------
import spikeforest as sf
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as pre
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.peak_localization import PeakRetriever, ExtractDenseWaveforms

# ---------------------------
# Utils
# ---------------------------
def seed_all(seed=42):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
seed_all(42)

def l2norm(x, dim=-1, eps=1e-8):
    return x / (x.norm(dim=dim, keepdim=True) + eps)

def compute_cluster_accuracy(pred, labels):
    """
    Compare clustering results (pred) with GT labels (labels) to compute accuracy  
    - pred: (N,) cluster IDs  
    - labels: (N,) GT labels (filled with -99 if GT is missing)
    """
    if labels is None or (isinstance(labels, np.ndarray) and np.all(labels == -99)):
        print("[Info] No GT → Skip Accuracy Calculation.")
        return None

    # 유효한 인덱스만 추출 (라벨이 -99가 아닌 경우만)
    valid_mask = labels != -99
    valid_pred = pred[valid_mask]
    valid_labels = labels[valid_mask]

    # 클러스터 라벨과 GT 라벨을 매핑 (majority voting)
    mapping = {}
    for cluster_id in np.unique(valid_pred):
        cluster_mask = valid_pred == cluster_id
        if cluster_mask.sum() == 0:
            continue
        cluster_labels = valid_labels[cluster_mask]
        if len(cluster_labels) == 0:
            mapping[cluster_id] = -1
        else:
            vals, counts = np.unique(cluster_labels, return_counts=True)
            mapping[cluster_id] = vals[np.argmax(counts)]

    # 매핑 적용
    mapped_preds = np.array([mapping.get(cid, -1) for cid in valid_pred])
    acc = accuracy_score(valid_labels, mapped_preds)
    return acc

# ---------------------------
# S4 Spike Encoder (Real S4Block)
# ---------------------------
class SpikeS4Net(nn.Module):
    """
    C: channels, T: samples per window
    Stem: Conv1d → S4Block stack (expects [B, D, L]) → global pool → embedding/projection
    """
    def __init__(self,
                 C: int,
                 d_model: int = 128,
                 n_layers: int = 4,
                 emb_dim: int = 128,
                 proj_dim: int = 64,
                 dropout: float = 0.1,
                 l_max: int | None = None,  
                 s4_kwargs: dict | None = None):
        super().__init__()
        self.d_model = d_model
        self.stem = nn.Sequential(
            nn.Conv1d(C, 64, 7, padding=3), nn.GELU(),
            nn.Conv1d(64, d_model, 3, padding=1), nn.GELU(),
        )
        if s4_kwargs is None:
            s4_kwargs = {}
        if l_max is not None:
            s4_kwargs.setdefault("l_max", l_max)

        # S4Block은 대부분 [B, D, L] 입력을 기대 → stem 출력 [B,d,T] 그대로 전달
        self.s4_layers = nn.ModuleList([S4Block(d_model=d_model, **s4_kwargs) for _ in range(n_layers)])

        self.norm = nn.GroupNorm(num_groups=min(8, d_model), num_channels=d_model)
        self.drop = nn.Dropout(dropout)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.to_emb = nn.Linear(d_model, emb_dim)
        self.to_proj = nn.Linear(emb_dim, proj_dim)

    def forward(self, x):  # x: [B, C, T]
        h = self.stem(x)    # [B, d, T]
        for s4 in self.s4_layers:
            out = s4(h)                 # expect [B, d, T]
            h = out[0] if isinstance(out, tuple) else out
        h = self.drop(self.norm(h))
        z = self.pool(h).squeeze(-1)          # [B, d]
        z = l2norm(self.to_emb(z), dim=-1)    # [B, emb_dim]
        p = l2norm(self.to_proj(z), dim=-1)   # [B, proj_dim]
        return z, p

# ---------------------------
# Losses: InfoNCE & Supervised Contrastive
# ---------------------------
def info_nce(p1, p2, temp=0.2):
    # p1, p2: [B, d] (unit-normalized)
    logits = (p1 @ p2.t()) / temp
    labels = torch.arange(p1.size(0), device=p1.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))

def supcon_loss(feats, labels, temp=0.2):
    """
    Supervised contrastive: for each anchor, positives = same-label samples (excluding itself)
    feats : [B, d], normalized
    labels: [B]
    """
    B, d = feats.shape
    sim = (feats @ feats.t()) / temp  # [B,B]
    mask = torch.eye(B, dtype=torch.bool, device=feats.device)
    sim = sim.masked_fill(mask, -1e9)

    labels = labels.view(-1, 1)
    pos_mask = (labels == labels.t()) & (~mask)  # [B,B]
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    denom = pos_mask.sum(dim=1).clamp_min(1)
    loss = -(pos_mask * log_prob).sum(dim=1) / denom
    return loss.mean()

# ---------------------------
# Dataset w/ simple augmentations
# ---------------------------
def augment_wave_np(x):
    """
    Basic spike-friendly augmentations:
    - small time shift, amp scale, light Gaussian noise, optional channel drop
    """
    x = x.copy()
    C, T = x.shape
    # shift
    s = np.random.randint(-2, 3)
    x = np.roll(x, s, axis=-1)
    # amp scale
    x *= (1.0 + 0.15 * (2*np.random.rand() - 1))
    # noise
    x += 0.02 * np.random.randn(*x.shape).astype(np.float32)
    # drop channel
    if C > 1 and np.random.rand() < 0.15:
        ch = np.random.randint(0, C)
        x[ch, :] = 0.0
    return x

class SpikeContrastiveDS(torch.utils.data.Dataset):
    def __init__(self, waves: np.ndarray, labels: np.ndarray | None = None):
        """
        waves : (N, C, T)
        labels: (N,) neuron id (optional)
        """
        self.waves = waves.astype(np.float32)
        self.labels = None if labels is None else labels.astype(np.int64)

    def __len__(self): return self.waves.shape[0]

    def __getitem__(self, idx):
        w = self.waves[idx]
        v1 = torch.from_numpy(augment_wave_np(w)).float()
        v2 = torch.from_numpy(augment_wave_np(w)).float()
        if self.labels is None:
            return v1, v2
        else:
            return v1, v2, torch.tensor(self.labels[idx], dtype=torch.long)

def extract_peak_sample_indices(peaks) -> np.ndarray:
    """
    SpikeInterface 버전에 따라 detect_peaks의 필드명이 다르다.
    가능한 후보를 순서대로 탐색해 sample index를 int64 배열로 반환한다.
    """
    import numpy as np

    if hasattr(peaks, "dtype") and hasattr(peaks.dtype, "names") and peaks.dtype.names:
        candidates = (
            "sample_ind",      # 구버전
            "sample_index",    # 신버전
            "index",
            "frame",
            "sample",
        )
        for k in candidates:
            if k in peaks.dtype.names:
                return peaks[k].astype(np.int64)

        # 디버깅용: 가용 필드 보여주기
        avail = ", ".join(peaks.dtype.names)
        raise RuntimeError(
            f"detect_peaks 결과에서 sample index 필드를 찾지 못했습니다. 가용 필드: {avail}"
        )

    if isinstance(peaks, np.ndarray) and peaks.ndim == 2 and peaks.shape[1] >= 1 and \
       np.issubdtype(peaks.dtype, np.integer):
        return peaks[:, 0].astype(np.int64)
    
    if isinstance(peaks, dict):
        for k in ("sample_ind", "sample_index", "index", "frame", "sample"):
            if k in peaks:
                arr = np.asarray(peaks[k])
                return arr.astype(np.int64)
        raise RuntimeError(f"detect_peaks 결과(dict)에서 sample index 키를 찾지 못했습니다. 키: {list(peaks.keys())}")

    raise RuntimeError("detect_peaks 결과 형식을 해석할 수 없습니다.")

# =========================
# GT label matching: peaks ↔ sorting_true
# =========================
def build_gt_from_sorting(peaks_sample_idx: np.ndarray,
                          sorting_true,
                          tol_ms: float,
                          fs: float) -> np.ndarray:
    tol_samples = int(round(tol_ms * 1e-3 * fs))
    all_times, all_units = [], []
    for uid in sorting_true.get_unit_ids():
        st = sorting_true.get_unit_spike_train(unit_id=uid)  # sample index
        all_times.append(st); all_units.append(np.full_like(st, uid))
    if len(all_times) == 0:
        return np.full(peaks_sample_idx.shape, -1, dtype=np.int64)

    gt_times  = np.concatenate(all_times).astype(np.int64)
    gt_units  = np.concatenate(all_units).astype(np.int64)
    order = np.argsort(gt_times)
    gt_times = gt_times[order]; gt_units = gt_units[order]

    labels = np.full(peaks_sample_idx.shape, -1, dtype=np.int64)
    pos = np.searchsorted(gt_times, peaks_sample_idx)
    for i, p in enumerate(peaks_sample_idx):
        cands = []
        if pos[i] > 0: cands.append(pos[i]-1)
        if pos[i] < len(gt_times): cands.append(pos[i])
        best = -1; best_dt = 1e18
        for j in cands:
            dt = abs(int(gt_times[j]) - int(p))
            if dt < best_dt and dt <= tol_samples:
                best_dt = dt; best = j
        labels[i] = gt_units[best] if best != -1 else -1
    return labels

# ---------------------------
# Training / Inference
# ---------------------------
def train_s4_encoder(waves, labels=None, *,
                     epochs=30, batch_size=512,
                     d_model=128, n_layers=4,
                     emb_dim=128, proj_dim=64, lr=1e-3,
                     use_supcon_ratio=0.5,  # fraction of batches mixing SupCon (if labels given)
                     device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Train S4 encoder with InfoNCE; optionally mix Supervised Contrastive when labels are provided.
    Uses Real S4Block stack (no pseudo fallback).
    """
    C, T = waves.shape[1], waves.shape[2]
    model = SpikeS4Net(
        C=C, d_model=256, n_layers=6,
        emb_dim=256, proj_dim=128,
        dropout=0.1, l_max=T, s4_kwargs=None
    ).to(device)

    ds = SpikeContrastiveDS(waves, labels)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True,
                                     pin_memory=True, num_workers=0)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for ep in range(1, epochs+1):
        total = 0.0
        for batch in dl:
            if labels is None:
                v1, v2 = batch
                lab = None
            else:
                if len(batch) == 3:
                    v1, v2, lab = batch
                else:
                    v1, v2 = batch; lab = None

            # ✅ 이미 Tensor라면 from_numpy 금지
            if isinstance(v1, np.ndarray):
                v1 = torch.from_numpy(v1).float()
                v2 = torch.from_numpy(v2).float()
            else:
                v1 = v1.float()
                v2 = v2.float()

            v1 = v1.to(device)
            v2 = v2.to(device)

            z1, p1 = model(v1)   # [B, emb], [B, proj]
            z2, p2 = model(v2)

            loss = info_nce(p1, p2, temp=0.2)
            if (labels is not None) and (lab is not None) and (np.random.rand() < use_supcon_ratio):
                if isinstance(lab, np.ndarray):
                    lab = torch.as_tensor(lab, dtype=torch.long)
                else:
                    lab = lab.long()
                lab = lab.to(device)
                
                loss = loss + 0.5 * supcon_loss(torch.cat([z1, z2], dim=0),
                                            torch.cat([lab, lab], dim=0),
                                            temp=0.2)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        print(f"[epoch {ep:02d}] loss={total/len(dl):.4f}")
    return model

@torch.no_grad()
def embed_all(model, waves, device=None, batch=2048):
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    Z = []
    for i in range(0, len(waves), batch):
        w = torch.from_numpy(waves[i:i+batch]).float().to(device)
        z, _ = model(w)
        Z.append(z.cpu().numpy())
    return np.concatenate(Z, axis=0)

def cluster_embeddings(Z, labels_gt=None):
    if HDBSCAN_AVAILABLE:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size= round(0.3 * np.sqrt(len(Z))), 
            min_samples= 5,                                   # 5~20 
            cluster_selection_epsilon=0.00,                    # 0.0 ~ 0.1 
            metric="euclidean",                               # cosine 
            cluster_selection_method="eom"                    # eom / leaf 
        )
        pred = clusterer.fit_predict(Z)
    else:
        K = len(np.unique(labels_gt)) if labels_gt is not None else 20
        pred = KMeans(n_clusters=K, n_init=10, random_state=0).fit_predict(Z)

    metrics = {}
    if labels_gt is not None:
        metrics["ARI"] = float(adjusted_rand_score(labels_gt, pred))
        metrics["NMI"] = float(normalized_mutual_info_score(labels_gt, pred))
    return pred, metrics

# =========================
# Spike_Data Load / Filtering / Feature Extraction 
# =========================
def load_real_data_and_windows():
    """
    - SpikeForest synthetic monotrode에서 하나의 recording을 선택
    - band-pass filtering
    - detect_peaks()
    - ExtractDenseWaveforms()로 (N, C, T) 파형 추출
    - sorting_true로부터 GT 라벨은 peak timestamp 근접 매칭으로 생성
    """
    # 1) Loading SpikeForest monotrode dataset
    synth_monotrode_uri = 'sha1://3b265eced5640c146d24a3d39719409cceccc45b?synth-monotrode-spikeforest-recordings.json'
    all_recordings = sf.load_spikeforest_recordings(synth_monotrode_uri)
    syn_mono = all_recordings[11]  # Data selection

    # 2) Recording / Sorting(GT)
    recording_sf = syn_mono.get_recording_extractor()
    sorting_true = syn_mono.get_sorting_true_extractor()   # GT spike trains

    traces = recording_sf.get_traces()
    arr = np.asarray(traces)
    if arr.ndim != 2:
        raise ValueError(f"Unexpected traces ndim: {arr.ndim}, shape={arr.shape}")
    if arr.shape[0] < arr.shape[1]:  # (C, T) → (T, C)
        arr = arr.T
    fs = float(recording_sf.get_sampling_frequency())
    recording = se.NumpyRecording(traces_list=arr, sampling_frequency=fs)

    # 3) band-pass 300-3000 Hz
    recording_filtered = pre.bandpass_filter(recording, freq_min=300, freq_max=3000, dtype="float32")

    # 4) peak detection
    job_kwargs = dict(chunk_duration='1s', n_jobs=8, progress_bar=False)
    peaks = detect_peaks(
        recording=recording,
        peak_sign='neg',
        detect_threshold=5,
        method='by_channel',
        gather_mode='memory',
        **job_kwargs
    )

    # 5) Waveform extraction (N, C, T)
    retriever = PeakRetriever(recording_filtered, peaks)
    wf_node = ExtractDenseWaveforms(
        recording_filtered,
        parents=[retriever],
        ms_before=0.5,  #0.5
        ms_after=1.0,   #1.0
        return_output=True
    )
    waveforms = wf_node.compute(traces=recording_filtered.get_traces(), peaks=peaks)
    wf = np.asarray(waveforms)
    if wf.ndim != 3:
        raise ValueError(f"Unexpected waveforms shape: {wf.shape}")
    N, A, B = wf.shape
    if A < B:  # (N, T, C) → (N, C, T)
        wf = np.transpose(wf, (0, 2, 1))
    waves = wf.astype(np.float32)

    # 6) per-sample regularization
    mu = waves.mean(axis=(0, 2), keepdims=True)
    sd = waves.std(axis=(0, 2), keepdims=True) + 1e-6
    waves = ((waves - mu) / sd).astype(np.float32)

    # 7) timestamps & GT labels (approximate matching)
    try:
        timestamps = extract_peak_sample_indices(peaks)
        
        if sorting_true is not None:
            labels = build_gt_from_sorting(timestamps, sorting_true, tol_ms=0.6, fs=fs)
        else:
            # Fill labels with -99 if GT is missing
            labels = np.full_like(timestamps, -99, dtype=np.int64)
    except Exception as e:
        if hasattr(peaks, "dtype") and hasattr(peaks.dtype, "names"):
            print("[DEBUG] peaks.dtype.names:", peaks.dtype.names)
        elif isinstance(peaks, dict):
            print("[DEBUG] peaks.keys():", list(peaks.keys()))
        raise
    labels = build_gt_from_sorting(timestamps, sorting_true, tol_ms=0.6, fs=fs)

    return recording, peaks, waves, sorting_true, labels, timestamps

def main():
    # ---- loading real data ----
    recording, peaks, waves, sorting_true, labels, timestamps = load_real_data_and_windows()

    # ---- training/clustering ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Input] waves: {waves.shape} | labels: {None if labels is None else labels.shape} | times: {timestamps.shape}")

    model = train_s4_encoder(
        waves=waves,
        labels=labels,             # Self-supervised mode when GT is not provided
        epochs=40,
        batch_size=512,
        d_model=256, n_layers=6,
        emb_dim=256, proj_dim=128, lr=1e-3,
        use_supcon_ratio=0.7 if labels is not None else 0.0,
        device=device
    )

    Z = embed_all(model, waves, device=device, batch=2048)
    pred, metrics = cluster_embeddings(Z, labels_gt=labels)
    print("[Cluster] metrics:", metrics)
    
    # ---- Accuracy Calculation ----
    acc = compute_cluster_accuracy(pred, labels)
    if acc is not None:
        print(f"[Cluster] Accuracy: {acc:.4f}")

    uniq, cnt = np.unique(pred, return_counts=True)
    print("[Summary] clusters (id:count) — first 12:", dict(zip(uniq.tolist()[:12], cnt.tolist()[:12])))
    
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    Z_2d = TSNE(n_components=2, random_state=0).fit_transform(Z)
    plt.figure(figsize=(8,6))
    plt.scatter(Z_2d[:,0], Z_2d[:,1], c=labels, cmap='tab20', s=2)
    plt.title('Spike Embedding Visualization')

    plt.savefig("spike_embedding_tsne.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()