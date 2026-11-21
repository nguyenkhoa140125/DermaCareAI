"""
hg1_masi.py -- Huong 1, GIAI DOAN CUOI: tinh chi so dien tich va muc do nang MASI tu mask vung nam
cua tung ROI.  KHONG can PyTorch.

MASI (Kimbrough-Green va cs., 1994; Pandya va cs., 2011):
  MASI = 0.3*A_tran*(D+H) + 0.3*A_ma_phai*(D+H) + 0.3*A_ma_trai*(D+H) + 0.1*A_cam*(D+H)     (0 .. 48)
  A: diem dien tich 0-6 (0: 0% | 1: <10% | 2: 10-29% | 3: 30-49% | 4: 50-69% | 5: 70-89% | 6: 90-100%)
  D: do dam 0-4,  H: do dong deu 0-4  (bac si cham bang mat)

CANH BAO QUAN TRONG: A tinh duoc truc tiep tu mask. D va H do bac si cham, o day chi duoc UOC LUONG
bang quy tac thu cong (xem darkness_score, homogeneity_score); cac nguong la GIA DINH, CHUA duoc hieu
chinh hay kiem chung voi diem bac si. Vi vay MASI tu dong o day la chi so THAM KHAO, khong phai MASI lam sang.
"""
import cv2
import numpy as np

from hg1_roi import MASI_ROIS

MASI_WEIGHTS = {"forehead": 0.3, "right_malar": 0.3, "left_malar": 0.3, "chin": 0.1}

# Nguong uoc luong (GIA DINH, can hieu chinh neu co diem bac si)
DARK_BINS = (4.0, 8.0, 12.0)          # delta L* (da thuong - nam): <4 ->1, <8 ->2, <12 ->3, con lai ->4
HOMO_BINS = (0.75, 0.55, 0.35)        # do dac (lesion / bao loi): >=.75 ->1, >=.55 ->2, >=.35 ->3, con lai ->4
MIN_SKIN_FRACTION = 0.30              # ROI co it hon 30% la da thi canh bao
RING_MIN_PIXELS = 300                # vanh da sat vung nam can it nhat chung nay pixel
MIN_CONTRAST_DE = 3.0                 # delta E (Lab) giua vung nam va da sat ben < nguong nay -> nghi duong tinh gia
LOCAL_REF_MIN_FRAC = 0.10             # moc cuc bo can it nhat 10% da khong nam trong ROI
MIN_COMPONENT_FRAC = 0.002            # bo mang nam nho hon 0.2% dien tich ROI (nhieu)


def area_score(pct):
    if pct <= 0:
        return 0
    for score, upper in ((1, 10), (2, 30), (3, 50), (4, 70), (5, 90)):
        if pct < upper:
            return score
    return 6


def skin_mask(rgb):
    """Mat na da don gian bang nguong YCrCb (+ dong lo nho). Khong hoan hao voi da rat toi hoac anh lech mau."""
    ycc = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    m = cv2.inRange(ycc, (0, 133, 77), (255, 180, 135)) > 0
    k = max(3, int(0.02 * min(rgb.shape[:2])) | 1)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    return m.astype(bool)


def clean_lesion(lesion, min_frac=MIN_COMPONENT_FRAC):
    """Bo cac thanh phan lien thong rat nho (< min_frac dien tich ROI)."""
    les = lesion.astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(les, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_frac * les.size
    return keep[lab]


def lab_L(rgb):
    """L* trong thang 0-100."""
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[:, :, 0].astype(np.float32) * (100.0 / 255.0)


def normal_skin_pixels(rgb, lesion, skin):
    """Pixel da binh thuong: la da, khong phai nam, cach nam mot khoang an toan."""
    k = max(3, int(0.03 * min(rgb.shape[:2])) | 1)
    near = cv2.dilate(lesion.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    return skin & ~near


def ring_pixels(rgb, lesion, skin):
    """Vanh da ngay sat ben ngoai vung nam (cach mep nam mot khoang nho): so sanh voi vanh nay it bi anh huong
    boi anh sang giam dan tu giua mat ra mep mat hon la so voi 'da binh thuong' o xa."""
    h, w = lesion.shape
    k1 = max(3, int(0.02 * min(h, w))) | 1
    k2 = max(k1 + 4, int(0.08 * min(h, w))) | 1
    les8 = lesion.astype(np.uint8)
    outer = cv2.dilate(les8, np.ones((k2, k2), np.uint8)) > 0
    inner = cv2.dilate(les8, np.ones((k1, k1), np.uint8)) > 0
    return outer & ~inner & skin & ~lesion


def darkness_score(lesion_L, ref_L):
    """Do dam D (1-4) tu chenh lech do sang trung vi giua da binh thuong va vung nam. 0 neu khong co nam."""
    if lesion_L is None or ref_L is None:
        return None
    delta = ref_L - lesion_L
    for score, upper in zip((1, 2, 3), DARK_BINS):
        if delta < upper:
            return score
    return 4


def homogeneity_score(lesion):
    """Do dong deu H (1-4): vung nam dac, lien mach -> thap; rai rac, loang lo -> cao. Dua tren
    lesion_area / dien_tich_bao_loi cua toan bo cac mang nam."""
    cnts, _ = cv2.findContours(lesion.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0, None
    hull = cv2.convexHull(np.concatenate(cnts))
    hull_area = max(1.0, cv2.contourArea(hull))
    solidity = min(1.0, float(lesion.sum()) / hull_area)
    for score, lo in zip((1, 2, 3), HOMO_BINS):
        if solidity >= lo:
            return score, solidity
    return 4, solidity


def global_reference_L(items):
    """items: list (rgb_crop, lesion, skin) -> trung vi L* cua da binh thuong tren tat ca ROI (hoac None)."""
    vals = []
    for rgb, lesion, skin in items:
        n = normal_skin_pixels(rgb, lesion, skin)
        if n.sum() > 0:
            vals.append(lab_L(rgb)[n])
    if not vals:
        return None
    v = np.concatenate(vals)
    return float(np.median(v)) if v.size >= 200 else None


def roi_metrics(rgb, lesion, ref_L, use_skin=True):
    """Chi so cho MOT ROI. rgb: crop ROI (do phan giai goc); lesion: mask nam cung kich thuoc (0/1).
    ref_L: do sang da binh thuong TOAN MAT (du phong). Thu tu moc: (1) VANH da sat ben ngoai vung nam, (2) da khong nam
    trong ROI, (3) toan mat (chi khi ROI gan nhu toan nam). Moc cang gan cang it bi anh sang khong deu anh huong."""
    warnings = []
    skin = skin_mask(rgb) if use_skin else np.ones(lesion.shape, bool)
    skin_frac = float(skin.mean())
    if use_skin and skin_frac < MIN_SKIN_FRACTION:
        warnings.append(f"chi {100 * skin_frac:.0f}% ROI la da; dien tich % co the khong dang tin")
        skin = np.ones(lesion.shape, bool)  # khong co mat na da dang tin -> dung ca ROI
    les = clean_lesion(lesion.astype(bool) & skin) if lesion.any() else np.zeros(lesion.shape, bool)
    denom = int(skin.sum())
    pct = 100.0 * float(les.sum()) / denom if denom else 0.0
    A = area_score(pct)
    if A == 0:
        return {"area_percent": round(pct, 2), "A": 0, "D": 0, "H": 0, "skin_fraction": round(skin_frac, 3),
                "solidity": None, "delta_L": None, "contrast_dE": None, "ref_source": None, "d_reliable": True, "warnings": warnings}
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[:, :, 0] *= 100.0 / 255.0
    lab[:, :, 1:] -= 128.0
    L = lab[:, :, 0]
    lesion_L = float(np.median(L[les]))
    ring = ring_pixels(rgb, les, skin)
    normal = normal_skin_pixels(rgb, les, skin)
    contrast = None
    if ring.sum() >= RING_MIN_PIXELS:
        ref, source = float(np.median(L[ring])), "ring"
        contrast = float(np.linalg.norm(np.median(lab[ring], axis=0) - np.median(lab[les], axis=0)))
    elif normal.sum() >= max(300, LOCAL_REF_MIN_FRAC * denom):
        ref, source = float(np.median(L[normal])), "local"
    else:
        ref, source = ref_L, "global"
    reliable = True
    if ref is None:
        warnings.append("khong co da binh thuong de lam moc -> D tam dat = 1")
        D, delta, reliable = 1, None, False
    else:
        delta = ref - lesion_L
        D = darkness_score(lesion_L, ref)
        if delta <= 0:
            warnings.append(f"vung nam KHONG toi hon da tham chieu (delta L*={delta:.1f}, moc {source}) "
                            "-> D khong dang tin (anh sang khong deu hoac phan vung sai?); D tam = 1")
            reliable = False
        if contrast is not None and contrast < MIN_CONTRAST_DE:
            warnings.append(f"vung duoc phan vung gan nhu KHONG khac da sat ben (delta E={contrast:.1f} < {MIN_CONTRAST_DE:g}) "
                            "-> nghi duong tinh gia hoac nam rat nhat; D khong dang tin")
            reliable = False
    H, solidity = homogeneity_score(les)
    return {"area_percent": round(pct, 2), "A": A, "D": D, "H": H, "skin_fraction": round(skin_frac, 3),
            "solidity": None if solidity is None else round(solidity, 3),
            "delta_L": None if delta is None else round(delta, 2),
            "contrast_dE": None if contrast is None else round(contrast, 2), "ref_source": source,
            "d_reliable": reliable, "warnings": warnings}


def compute_masi(metrics):
    """metrics: dict ten_roi -> ket qua roi_metrics. Tra ve MASI va dong gop tung vung.
    Thieu vung MASI nao (bi cat/khong thay) thi ghi vao 'missing' va MASI chi cong cac vung co du lieu,
    nen co the THAP HON that. 'unreliable_D' liet ke vung co D khong dang tin (MASI cua anh do chi la can duoi)."""
    contrib, missing, unreliable = {}, [], []
    for name in MASI_ROIS:
        m = metrics.get(name)
        if m is None:
            missing.append(name)
            continue
        contrib[name] = round(MASI_WEIGHTS[name] * m["A"] * (m["D"] + m["H"]), 3)
        if m["A"] > 0 and not m.get("d_reliable", True):
            unreliable.append(name)
    return {"masi": round(sum(contrib.values()), 2), "contributions": contrib, "missing_rois": missing,
            "unreliable_D": unreliable, "complete": not missing, "reliable": not unreliable and not missing,
            "scale": "0-48"}
