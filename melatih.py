import json
import re #re itu modul untuk regular expresssion gunanya untuk mencari mencocokan 
import time
import unicodedata #menangani data teks dengan data unicode
import numpy as np
import pandas as pd
import torch
from pathlib import Path #digunakan untuk bekerja dengan pathfile dan direktori pendekatan objek

# =========================
# Konfigurasi & device
# =========================
import os #digunakan untuk berinteraksi engan sistem operasi seperti mengambil data dari folder tertentu
BASE_DIR = Path(os.environ.get("TASK2_DATA_DIR", r"D:\coding stuff\pandasenv\task2\dataset-task2"))
SCREENSHOT_DIR = BASE_DIR / "screenshots"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device dipakai: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# =========================
# load data
# =========================
def load_data(base_dir: Path = BASE_DIR):
    articles = pd.read_csv(base_dir / "articles.csv")
    categories = pd.read_csv(base_dir / "categories.csv")
    states_train = pd.read_csv(base_dir / "states_train.csv")
    states_test = pd.read_csv(base_dir / "states_test.csv")
    sample_submission = pd.read_csv(base_dir / "sample_submission.csv")

    data = {
        "articles": articles,
        "categories": categories,
        "states_train": states_train,
        "states_test": states_test,
        "sample_submission": sample_submission,
    }
    return data


#eda
def _compute_woe_iv(df: pd.DataFrame, feature: str, target: str) -> tuple[pd.DataFrame, float]: #mengembalikan data frame yang sudah di proses
    """WoE & IV untuk fitur biner/kategorikal terhadap target biner (0/1)."""
    g = df.groupby(feature)[target].agg(["count", "sum"]) #berfungsi untuk mengelompokan data berdasarkan nilai unik dalam fitur yang pilih
    g.columns = ["total", "events"] #baris `g.columns = ["total", "events"]`. Ini tujuannya untuk mengganti nama kolom hasil 
    #agregasi tadi. Kolom `count` menjadi `total`, dan kolom `sum` menjadi `events`.
    g["non_events"] = g["total"] - g["events"]
    total_events = g["events"].sum()
    total_non_events = g["non_events"].sum()
    eps = 1e-6
    g["dist_events"] = (g["events"] + eps) / (total_events + eps)
    g["dist_non_events"] = (g["non_events"] + eps) / (total_non_events + eps)
    g["woe"] = np.log(g["dist_events"] / g["dist_non_events"])
    g["iv"] = (g["dist_events"] - g["dist_non_events"]) * g["woe"]
    return g, g["iv"].sum()


def _gini_impurity(counts: pd.Series) -> float:
    p = counts / counts.sum()
    return 1 - (p ** 2).sum()


def _entropy(counts: pd.Series) -> float:
    p = counts / counts.sum()
    return -(p * np.log2(p)).sum()


def run_eda(data: dict):
    import matplotlib
    matplotlib.use("Agg")  # aman dijalankan tanpa display, langsung save ke file
    import matplotlib.pyplot as plt
    import seaborn as sns

    out_dir = Path("eda_outputs")
    out_dir.mkdir(exist_ok=True)

    articles = data["articles"]
    categories = data["categories"]
    states_train = data["states_train"]
    states_test = data["states_test"]

    print("\n" + "=" * 50)
    print("EDA: categories")
    print("=" * 50)
    top_level = categories["category"].str.split(".").str[1]
    print("Distribusi top-level category:")
    print(top_level.value_counts())

    n_cat_per_article = categories.groupby("article_id").size()
    print(f"\nArtikel dengan >1 kategori: {(n_cat_per_article > 1).sum()} / {len(n_cat_per_article)}")
    print(f"Artikel di articles.csv TANPA kategori sama sekali: "
          f"{len(set(articles.article_id) - set(categories.article_id))}")

    print("\n" + "=" * 50)
    print("EDA: states_train - struktur dasar")
    print("=" * 50)
    pair_next_nunique = states_train.groupby(
        ["current_article_id", "target_article_id"]
    )["next_article_id"].nunique()
    print(f"Pasangan (current,target) unik: {len(pair_next_nunique)} dari {len(states_train)} baris")
    print(f"Pasangan dengan next_article_id berbeda-beda (label ambigu): {(pair_next_nunique > 1).sum()}")

    reach_directly = (states_train["next_article_id"] == states_train["target_article_id"]).sum()
    print(f"Kasus next == target langsung (1 klik sampai tujuan): {reach_directly} "
          f"({reach_directly/len(states_train):.2%})")

    print("\nArtikel yang paling sering jadi 'current':")
    print(states_train["current_article_id"].value_counts().head(10))

    print("\n" + "=" * 50)
    print("EDA: overlap train vs test")
    print("=" * 50)
    cur_overlap = set(states_train.current_article_id) & set(states_test.current_article_id)
    print(f"current_article_id unik di train: {states_train.current_article_id.nunique()}")
    print(f"current_article_id unik di test : {states_test.current_article_id.nunique()}")
    print(f"overlap current_article_id train/test: {len(cur_overlap)}")

    print("\n" + "=" * 50)
    print("EDA: dimensi screenshot (sampel)")
    print("=" * 50)
    from PIL import Image
    import random
    sample_ids = random.sample(list(articles.article_id), k=min(50, len(articles)))
    sizes = []
    for aid in sample_ids:
        p = SCREENSHOT_DIR / f"{aid}.png"
        if p.exists():
            with Image.open(p) as im:
                sizes.append(im.size)
    sizes_df = pd.DataFrame(sizes, columns=["width", "height"])
    print(sizes_df.describe())

    # =====================================================================
    # A. Target Class Distribution (Ketimpangan Kelas)
    #    target = next_article_id (label yang diprediksi)
    # =====================================================================
    print("\n" + "=" * 50)
    print("EDA-A: Target Class Distribution (next_article_id)")
    print("=" * 50)
    next_counts = states_train["next_article_id"].value_counts()
    print(f"Jumlah kelas unik (artikel yang pernah jadi next): {len(next_counts)} dari {len(states_train)} baris")
    print(f"Top-1 class share: {next_counts.iloc[0] / len(states_train):.4%}")
    print(f"Top-10 class share (kumulatif): {next_counts.head(10).sum() / len(states_train):.4%}")
    print(f"Gini impurity distribusi next_article_id: {_gini_impurity(next_counts):.4f}")
    print(f"Entropy distribusi next_article_id: {_entropy(next_counts):.4f} bit "
          f"(maks kalau uniform = {np.log2(len(next_counts)):.2f} bit)")

    fig, ax = plt.subplots(figsize=(10, 5))
    next_counts.head(30).plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title("Top 30 artikel paling sering jadi next_article_id (train)")
    ax.set_xlabel("article_id")
    ax.set_ylabel("frekuensi")
    plt.tight_layout()
    plt.savefig(out_dir / "A_target_class_distribution.png", dpi=120)
    plt.close(fig)
    print(f"-> plot disimpan: {out_dir / 'A_target_class_distribution.png'}")

    # =====================================================================
    # Feature engineering ringan untuk analisis B, C, D, E:
    # kategori top-level current/target/next + flag direct_hit & cat_match
    # =====================================================================
    cat_first = (
        categories.sort_values("article_id")
        .drop_duplicates("article_id", keep="first")
        .assign(top_level=lambda d: d["category"].str.split(".").str[1])
        .set_index("article_id")["top_level"]
    )
    df = states_train.copy()
    df["cur_top"] = df["current_article_id"].map(cat_first)
    df["tgt_top"] = df["target_article_id"].map(cat_first)
    df["next_top"] = df["next_article_id"].map(cat_first)
    df["direct_hit"] = (df["next_article_id"] == df["target_article_id"]).astype(int)
    df["cat_match_cur_tgt"] = (df["cur_top"] == df["tgt_top"]).astype(int)
    df["cat_match_next_tgt"] = (df["next_top"] == df["tgt_top"]).astype(int)

    # =====================================================================
    # B. Segmented Bar Plot: direct-hit rate per top-level category (current)
    # =====================================================================
    print("\n" + "=" * 50)
    print("EDA-B: Segmented Bar Plot - direct_hit rate per kategori current")
    print("=" * 50)
    hit_rate_by_cat = df.groupby("cur_top")["direct_hit"].agg(["mean", "count"]).sort_values("mean", ascending=False)
    print(hit_rate_by_cat)

    fig, ax = plt.subplots(figsize=(10, 6))
    hit_rate_by_cat["mean"].plot(kind="barh", ax=ax, color="darkorange")
    ax.set_title("Direct-hit rate (next==target) per kategori top-level artikel current")
    ax.set_xlabel("direct-hit rate")
    plt.tight_layout()
    plt.savefig(out_dir / "B_segmented_bar_direct_hit_rate.png", dpi=120)
    plt.close(fig)
    print(f"-> plot disimpan: {out_dir / 'B_segmented_bar_direct_hit_rate.png'}")

    # =====================================================================
    # C. Crosstab Heatmap: transisi kategori current -> target, current -> next
    # =====================================================================
    print("\n" + "=" * 50)
    print("EDA-C: Crosstab Heatmap kategori")
    print("=" * 50)
    ct_cur_tgt = pd.crosstab(df["cur_top"], df["tgt_top"])
    ct_cur_next = pd.crosstab(df["cur_top"], df["next_top"])

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    sns.heatmap(ct_cur_tgt, cmap="viridis", ax=axes[0], cbar_kws={"label": "frekuensi"})
    axes[0].set_title("Crosstab: kategori current vs target")
    sns.heatmap(ct_cur_next, cmap="viridis", ax=axes[1], cbar_kws={"label": "frekuensi"})
    axes[1].set_title("Crosstab: kategori current vs next (label)")
    plt.tight_layout()
    plt.savefig(out_dir / "C_crosstab_heatmap_category.png", dpi=120)
    plt.close(fig)
    print(f"-> plot disimpan: {out_dir / 'C_crosstab_heatmap_category.png'}")
    print("\nProporsi cat_match_next_tgt (next artikel setopik dgn target):")
    print(df["cat_match_next_tgt"].value_counts(normalize=True))

    # =====================================================================
    # D. Point Plot: rata-rata direct-hit rate per kategori, dgn interval
    # =====================================================================
    print("\n" + "=" * 50)
    print("EDA-D: Point Plot direct-hit rate per kategori (dgn CI)")
    print("=" * 50)
    fig, ax = plt.subplots(figsize=(10, 6))
    order = hit_rate_by_cat.index.tolist()
    sns.pointplot(data=df, x="direct_hit", y="cur_top", order=order, ax=ax,
                  errorbar=("ci", 95), linestyle="none")
    ax.set_title("Direct-hit rate per kategori current (point estimate + 95% CI)")
    plt.tight_layout()
    plt.savefig(out_dir / "D_pointplot_direct_hit_rate.png", dpi=120)
    plt.close(fig)
    print(f"-> plot disimpan: {out_dir / 'D_pointplot_direct_hit_rate.png'}")

    # =====================================================================
    # E. Weight of Evidence (WoE) & Information Value (IV)
    #    target biner: direct_hit (next == target)
    #    fitur yg diuji: cat_match_cur_tgt (current & target setopik?)
    # =====================================================================
    print("\n" + "=" * 50)
    print("EDA-E: Weight of Evidence & Information Value")
    print("=" * 50)
    woe_table, iv_total = _compute_woe_iv(df, "cat_match_cur_tgt", "direct_hit")
    print("Tabel WoE untuk fitur 'cat_match_cur_tgt' terhadap target 'direct_hit':")
    print(woe_table[["total", "events", "non_events", "woe", "iv"]])
    print(f"\nTotal Information Value (IV): {iv_total:.5f}")
    print("Interpretasi IV umum: <0.02 tidak berguna | 0.02-0.1 lemah | "
          "0.1-0.3 sedang | 0.3-0.5 kuat | >0.5 mencurigakan (kebocoran data)")

    # =====================================================================
    # F. Validasi fitur OCR (pytesseract): apakah judul target BENAR-BENAR
    #    ketemu sbg teks di screenshot current -> bandingkan predictive power-nya
    #    (via IV) dengan cat_match_cur_tgt yang cuma berbasis kategori topik.
    # =====================================================================
    print("\n" + "=" * 50)
    print("EDA-F: Validasi fitur OCR (target_title_in_cur_ocr) vs direct_hit")
    print("=" * 50)
    ocr_df = extract_screenshot_ocr(articles)
    ocr_feat, title_map = build_ocr_features(articles, ocr_df)
    df_ocr = add_ocr_derived_features(df, ocr_feat, title_map)

    woe_ocr, iv_ocr = _compute_woe_iv(df_ocr, "target_title_in_cur_ocr", "direct_hit")
    print("Tabel WoE untuk fitur 'target_title_in_cur_ocr' terhadap target 'direct_hit':")
    print(woe_ocr[["total", "events", "non_events", "woe", "iv"]])
    print(f"\nTotal Information Value (IV) fitur OCR: {iv_ocr:.5f}  "
          f"(pembanding: IV cat_match_cur_tgt = {iv_total:.5f})")

    # =====================================================================
    # G. Candidate extraction: cocokkan teks OCR current thd SEMUA judul artikel
    #    -> proxy daftar link nyata yg tersedia di halaman (bukan cuma target).
    #    Tujuan: cek dulu CAKUPANnya sebelum dipakai bikin model ranking-kandidat.
    # =====================================================================
    print("\n" + "=" * 50)
    print("EDA-G: Candidate extraction (judul artikel di teks OCR current)")
    print("=" * 50)
    title_lookup = build_title_lookup(articles)
    matcher = build_candidate_matcher(title_lookup)
    candidates = extract_candidates_from_ocr(ocr_feat, matcher)
    validate_candidate_coverage(states_train, candidates)

    print("\nSemua plot EDA tersimpan di folder:", out_dir.resolve())

#preprocessing
CACHE_DIR = Path("cache")
FEATURES_DIR = Path("features")


def extract_screenshot_dims(articles: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    """Cache dimensi (width, height) tiap screenshot -> hindari re-scan 4604 gambar tiap run."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache_path = CACHE_DIR / "screenshot_dims.csv"
    if cache_path.exists() and not force:
        print(f"[cache] load dimensi screenshot dari {cache_path}")
        return pd.read_csv(cache_path)

    from PIL import Image
    print(f"[cache] scanning {len(articles)} screenshot untuk ambil dimensi...")
    rows = []
    for aid in articles["article_id"]:
        p = SCREENSHOT_DIR / f"{aid}.png"
        if p.exists():
            with Image.open(p) as im:
                w, h = im.size
        else:
            w, h = np.nan, np.nan
        rows.append({"article_id": aid, "width": w, "height": h})
    dims = pd.DataFrame(rows)
    dims.to_csv(cache_path, index=False)
    print(f"[cache] disimpan ke {cache_path}")
    return dims


def _normalize_text(s: str) -> str:
    """Lowercase, buang aksen & karakter non alfanumerik -> mempermudah substring match."""
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _ocr_single_image(reader, path: Path, max_height: int = 1500) -> str:
    """
    OCR satu screenshot dengan EasyOCR (jalan di GPU/CUDA kalau tersedia).
    Screenshot Wikipedia di dataset ini bisa sangat panjang (>20000px height),
    jadi dibatasi max_height (ambil dari bagian atas halaman) karena infobox +
    paragraf pembuka + link-link awal biasanya paling relevan utk fitur ini,
    dan supaya waktu OCR tetap terkendali untuk ~4600 gambar.
    Naikkan max_height kalau mau meng-cover isi halaman lebih dalam (lebih lambat).
    """
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None  # screenshot legit & besar, matikan guard decompression-bomb PIL

    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        h_used = min(h, max_height)
        im = im.crop((0, 0, w, h_used))
        arr = np.array(im)

    # batch_size: sebelumnya default=1 -> tiap kotak teks yg kedeteksi di-recognize
    # SATU-SATU (overhead kernel launch numpuk kalau halaman padat teks). Dinaikkan
    # supaya beberapa crop di-batch sekaligus ke GPU -> jauh lebih cepat.
    # canvas_size: default 2560 (network deteksi resize ke ukuran ini) -> diturunkan
    # ke 1280 karena screenshot sudah dipotong kecil (max_height), tidak perlu
    # kanvas sebesar itu.
    result = reader.readtext(
        arr, detail=0, paragraph=False,
        batch_size=16, canvas_size=1280,
    )
    return " ".join(result)


def extract_screenshot_ocr(articles: pd.DataFrame, force: bool = False,
                            max_height: int = 1500) -> pd.DataFrame:
    """
    Jalankan EasyOCR (GPU/CUDA kalau tersedia) ke semua screenshot & cache hasilnya
    -> hindari OCR ulang ~4600 gambar tiap kali script dijalankan.
    Ditulis INCREMENTAL per 100 gambar supaya kalau proses berhenti di tengah jalan
    (mis. run_eda dites terpisah, atau Ctrl+C), run berikutnya tinggal lanjut dari
    artikel yang belum ke-OCR, tidak mulai dari nol lagi.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    cache_path = CACHE_DIR / "screenshot_ocr.csv"

    if cache_path.exists() and not force:
        cached = pd.read_csv(cache_path, keep_default_na=False)
    else:
        cached = pd.DataFrame(columns=["article_id", "ocr_text"])

    done_ids = set(cached["article_id"])
    todo = articles.loc[~articles["article_id"].isin(done_ids), "article_id"].tolist()

    if not todo:
        print(f"[cache] load hasil OCR dari {cache_path} ({len(cached)} artikel)")
        return cached

    use_gpu = torch.cuda.is_available()
    print(f"[OCR] menjalankan EasyOCR untuk {len(todo)} screenshot "
          f"({len(done_ids)} sudah ada di cache) | gpu={use_gpu}...")

    try:
        import easyocr
    except ImportError as e:
        raise RuntimeError(
            "Package 'easyocr' belum terinstall. Jalankan:\n"
            "    pip install easyocr\n"
            "(pytesseract yang lama CPU-only & tidak bisa pakai GPU/CUDA, makanya diganti "
            "ke EasyOCR yang berbasis PyTorch dan otomatis pakai CUDA yang sama dgn training "
            "MLP-mu kalau tersedia.)"
        ) from e

    # reader di-init SEKALI di luar loop (load model ke GPU sekali saja, bukan per-gambar)
    reader = easyocr.Reader(["en"], gpu=use_gpu)

    rows = []
    t0 = time.time()
    for i, aid in enumerate(todo, 1):
        p = SCREENSHOT_DIR / f"{aid}.png"
        text = _ocr_single_image(reader, p, max_height=max_height) if p.exists() else ""
        rows.append({"article_id": aid, "ocr_text": text})

        if i % 100 == 0 or i == len(todo):
            elapsed = time.time() - t0
            print(f"  [OCR] {i}/{len(todo)} selesai ({elapsed:.0f}s, "
                  f"~{elapsed / i:.2f}s/gambar, sisa estimasi ~{elapsed / i * (len(todo) - i):.0f}s)")
            cached = pd.concat([cached, pd.DataFrame(rows)], ignore_index=True)
            cached.to_csv(cache_path, index=False)
            rows = []

    print(f"[OCR] selesai, hasil disimpan ke {cache_path}")
    return cached


def build_ocr_features(articles: pd.DataFrame, ocr_df: pd.DataFrame):
    """
    Ubah teks OCR mentah -> fitur & lookup siap pakai:
      - ocr_feat: DataFrame ter-index article_id berisi teks ternormalisasi,
        panjang teks, dan jumlah kata (proxy kepadatan konten yang ter-render).
      - title_map: Series article_id -> judul artikel ternormalisasi (dipakai utk
        cek apakah judul suatu artikel muncul di teks OCR artikel lain).
    """
    ocr_text = ocr_df.set_index("article_id")["ocr_text"].fillna("").astype(str)
    ocr_text_norm = ocr_text.map(_normalize_text)

    ocr_feat = pd.DataFrame({
        "ocr_text_norm": ocr_text_norm,
        "ocr_len": ocr_text_norm.str.len(),
        "ocr_word_count": ocr_text_norm.str.split().apply(len),
    })

    title_map = articles.set_index("article_id")["title"].map(_normalize_text)
    return ocr_feat, title_map


def add_ocr_derived_features(df: pd.DataFrame, ocr_feat: pd.DataFrame, title_map: pd.Series) -> pd.DataFrame:
    """
    Tambah fitur turunan OCR ke tabel states (current/target):
      - cur_ocr_len_log / cur_ocr_word_count_log: seberapa 'padat' teks yang
        ter-render di screenshot current (log transform, sama pola dgn cur_height_log).
      - target_title_in_cur_ocr: apakah judul target_article_id benar-benar
        ketemu sebagai teks di screenshot current_article_id -> sinyal langsung
        kalau link menuju target kemungkinan terlihat/ada di halaman current
        (lebih kuat dari cat_match_cur_tgt yang cuma menyamakan kategori topik).
    """
    df = df.copy()
    cur_ocr_norm = df["current_article_id"].map(ocr_feat["ocr_text_norm"]).fillna("")

    df["cur_ocr_len_log"] = np.log1p(
        df["current_article_id"].map(ocr_feat["ocr_len"]).fillna(0)
    )
    df["cur_ocr_word_count_log"] = np.log1p(
        df["current_article_id"].map(ocr_feat["ocr_word_count"]).fillna(0)
    )

    tgt_title_norm = df["target_article_id"].map(title_map).fillna("")
    df["target_title_in_cur_ocr"] = [
        int(bool(t) and t in txt) for t, txt in zip(tgt_title_norm, cur_ocr_norm)
    ]
    return df


def build_title_lookup(articles: pd.DataFrame) -> pd.DataFrame:
    """Normalisasi semua judul artikel jadi 'kamus' pencarian.
    Judul <=2 karakter dibuang -> terlalu generik, rawan false-positive match
    (mis. singkatan 2-huruf yg kebetulan nongol di teks manapun)."""
    lut = articles.copy()
    lut["title_norm"] = lut["title"].map(_normalize_text)
    before = len(lut)
    lut = lut[lut["title_norm"].str.len() > 2].reset_index(drop=True)
    dropped = before - len(lut)
    if dropped:
        print(f"[candidate] {dropped} judul artikel dibuang dari matcher (<=2 karakter, terlalu generik)")
    return lut


def build_candidate_matcher(title_lookup: pd.DataFrame):
    """Bangun keyword matcher (flashtext, trie-based) utk cari SEMUA judul artikel
    yg muncul di suatu teks sekali index -> jauh lebih cepat drpd 4604x4604
    substring check manual, dan otomatis match berbasis word-boundary (bukan
    substring mentah di tengah kata, beda dgn cek 't in txt' yg dipakai sebelumnya)."""
    from flashtext import KeywordProcessor
    kp = KeywordProcessor()
    for aid, title_norm in zip(title_lookup["article_id"], title_lookup["title_norm"]):
        kp.add_keyword(title_norm, str(aid))  # payload = article_id (sbg string)
    return kp


def extract_candidates_from_ocr(ocr_feat: pd.DataFrame, matcher) -> dict:
    """Utk tiap artikel (sbg current), cari semua article_id LAIN yg judulnya
    muncul di teks OCR halaman itu -> proxy daftar link yg tersedia di halaman.
    Urutan hasil MENGIKUTI urutan kemunculan pertama di teks (bukan diurutkan
    ulang) -> dipakai sbg fitur 'candidate_position' (link di awal halaman
    kemungkinan lebih sering diklik drpd link di bagian bawah)."""
    candidates = {}
    for aid, text in ocr_feat["ocr_text_norm"].items():
        if not text:
            candidates[aid] = []
            continue
        found = matcher.extract_keywords(text)  # urut sesuai posisi kemunculan
        seen, seen_set = [], set()
        for x in found:
            xi = int(x)
            if xi != aid and xi not in seen_set:
                seen_set.add(xi)
                seen.append(xi)
        candidates[aid] = seen
    return candidates


def validate_candidate_coverage(states_train: pd.DataFrame, candidates: dict):
    """Diagnostik: seberapa sering next_article_id / target_article_id BENAR-BENAR
    ketemu di candidate list current -> plafon akurasi kalau nanti prediksi
    dibatasi hanya ke kandidat ini (candidate-ranking approach)."""
    next_in_cand = [
        row.next_article_id in candidates.get(row.current_article_id, [])
        for row in states_train.itertuples()
    ]
    tgt_in_cand = [
        row.target_article_id in candidates.get(row.current_article_id, [])
        for row in states_train.itertuples()
    ]
    next_hit = np.mean(next_in_cand)
    tgt_hit = np.mean(tgt_in_cand)

    n_cand = pd.Series({aid: len(v) for aid, v in candidates.items()})
    print(f"Rata-rata jumlah kandidat per halaman: {n_cand.mean():.1f} "
          f"(median={n_cand.median():.0f}, min={n_cand.min()}, max={n_cand.max()})")
    print(f"Halaman TANPA kandidat sama sekali (0 match): {(n_cand == 0).sum()} / {len(n_cand)}")
    print(f"\nCoverage: next_article_id ada di candidate list current -> "
          f"{next_hit:.2%} ({sum(next_in_cand)}/{len(states_train)})")
    print(f"Coverage: target_article_id ada di candidate list current -> "
          f"{tgt_hit:.2%} ({sum(tgt_in_cand)}/{len(states_train)})")
    print("\n[interpretasi] Ini PLAFON akurasi kalau prediksi dibatasi ke candidate list saja.")
    print("Kalau coverage next_article_id rendah, kemungkinan besar karena max_height=1500")
    print("terlalu pendek dibanding tinggi halaman rata-rata (~6800-8800px) -> link yg")
    print("dituju ada di bagian halaman yg belum ke-OCR.")
    return next_hit, tgt_hit


def build_category_map(categories: pd.DataFrame) -> pd.Series:
    """article_id -> top-level category (ambil kategori pertama kalau >1)."""
    return (
        categories.sort_values("article_id")
        .drop_duplicates("article_id", keep="first")
        .assign(top_level=lambda d: d["category"].str.split(".").str[1])
        .set_index("article_id")["top_level"]
    )


def build_popularity_features(states_train: pd.DataFrame) -> dict:
    """Fitur popularitas/hub HANYA dihitung dari train (prior global, bukan per-baris label)."""
    return {
        "as_current": states_train["current_article_id"].value_counts(),
        "as_target": states_train["target_article_id"].value_counts(),
        "as_next": states_train["next_article_id"].value_counts(),  # proxy in-degree hub
    }


def build_tfidf_index(ocr_feat: pd.DataFrame):
    """TF-IDF dari SELURUH teks OCR (isi asli tiap halaman, bukan cuma flag biner
    spt sebelumnya). Dipakai utk hitung cosine similarity antar halaman -> model
    sekarang benar2 'membaca' konten teksnya, bukan cuma tau ada/tidaknya 1 kata."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    aids = ocr_feat.index.to_numpy()
    texts = ocr_feat["ocr_text_norm"].fillna("").tolist()
    vectorizer = TfidfVectorizer(max_features=5000, min_df=2, sublinear_tf=True)
    X = vectorizer.fit_transform(texts)
    aid_to_row = {int(aid): i for i, aid in enumerate(aids)}
    print(f"[tfidf] index dibangun: {X.shape[0]} artikel x {X.shape[1]} kata "
          f"(vocab dibatasi max_features=5000)")
    return vectorizer, X, aid_to_row


def paired_tfidf_similarity(X_tfidf, aid_to_row: dict, ids_a: pd.Series, ids_b: pd.Series) -> np.ndarray:
    """Cosine similarity BERPASANGAN: baris ke-i dari ids_a vs baris ke-i dari ids_b
    (bukan matriks similarity penuh -> jauh lebih hemat memori utk ratusan ribu pasangan)."""
    from sklearn.metrics.pairwise import paired_cosine_distances

    rows_a = np.array([aid_to_row.get(int(a), -1) for a in ids_a])
    rows_b = np.array([aid_to_row.get(int(b), -1) for b in ids_b])
    sims = np.zeros(len(ids_a), dtype=np.float32)
    valid = (rows_a >= 0) & (rows_b >= 0)
    if valid.any():
        Xa = X_tfidf[rows_a[valid]]
        Xb = X_tfidf[rows_b[valid]]
        sims[valid] = 1.0 - paired_cosine_distances(Xa, Xb)
    return sims


def add_candidate_features(pairs: pd.DataFrame, cat_map: pd.Series, pop: dict,
                            dims: pd.DataFrame, height_median: float,
                            X_tfidf=None, aid_to_row: dict = None) -> pd.DataFrame:
    """Fitur per pasangan (current, target, KANDIDAT) -> ini yg dilihat model,
    bukan lagi 1 baris/state spt sebelumnya."""
    df = pairs.copy()
    df["cur_top"] = df["current_article_id"].map(cat_map).fillna("Unknown")
    df["tgt_top"] = df["target_article_id"].map(cat_map).fillna("Unknown")
    df["cand_top"] = df["candidate_article_id"].map(cat_map).fillna("Unknown")

    df["cat_match_cur_tgt"] = (df["cur_top"] == df["tgt_top"]).astype(int)
    df["cat_match_cand_tgt"] = (df["cand_top"] == df["tgt_top"]).astype(int)
    df["cat_match_cur_cand"] = (df["cur_top"] == df["cand_top"]).astype(int)

    # fitur paling kuat (mirip target_title_in_cur_ocr, IV=2.44): apakah kandidat
    # ini PERSIS artikel target?
    df["candidate_is_target"] = (df["candidate_article_id"] == df["target_article_id"]).astype(int)

    dims_idx = dims.set_index("article_id")["height"]
    df["cand_height_log"] = np.log1p(df["candidate_article_id"].map(dims_idx).fillna(height_median))

    df["cand_popularity"] = df["candidate_article_id"].map(pop["as_current"]).fillna(0)
    df["cand_hub_indegree"] = df["candidate_article_id"].map(pop["as_next"]).fillna(0)
    df["tgt_hub_indegree"] = df["target_article_id"].map(pop["as_next"]).fillna(0)

    # posisi kemunculan kandidat di teks OCR current (0=paling awal) -> proxy
    # "link di atas halaman lebih sering diklik drpd link di bawah"
    df["candidate_position_norm"] = df["candidate_position"] / df["n_candidates"].clip(lower=1)
    df["n_candidates_log"] = np.log1p(df["n_candidates"])

    # --- TF-IDF cosine similarity: fitur yg BENAR2 pakai isi teks OCR ---
    if X_tfidf is not None and aid_to_row is not None:
        df["tfidf_sim_cand_tgt"] = paired_tfidf_similarity(
            X_tfidf, aid_to_row, df["candidate_article_id"], df["target_article_id"])
        df["tfidf_sim_cur_cand"] = paired_tfidf_similarity(
            X_tfidf, aid_to_row, df["current_article_id"], df["candidate_article_id"])
        df["tfidf_sim_cur_tgt"] = paired_tfidf_similarity(
            X_tfidf, aid_to_row, df["current_article_id"], df["target_article_id"])
        # inti gagasan: apakah klik kandidat ini bikin "isi halaman" makin mirip
        # target drpd posisi sekarang -> proxy greedy-navigation manusia
        df["tfidf_sim_delta"] = df["tfidf_sim_cand_tgt"] - df["tfidf_sim_cur_tgt"]

    return df


def build_candidate_pairs(states: pd.DataFrame, candidates: dict, fallback_article_id: int,
                           force_true_positive: bool = False, random_state: int = 42) -> pd.DataFrame:
    """
    Ubah 1 baris/state -> 1 baris/(state, kandidat). Kandidat diambil dari hasil
    extract_candidates_from_ocr(current_article_id) (EDA-G: rata2 19.3 kandidat/halaman,
    0/4604 halaman kosong).

    force_true_positive: KHUSUS dipakai utk TRAIN. Kalau next_article_id ternyata
    TIDAK ke-detect sbg kandidat organik (42.72% kasus, krn max_height=1500 kepotong
    atau anchor text beda dari judul persis), tetap dipaksa masuk sbg 1 kandidat
    tambahan -> supaya ada minimal 1 contoh positif yg bisa dipelajari model dari
    tiap state. TIDAK dipakai utk val/test (val/test HARUS pakai kandidat organik
    apa adanya, biar angka akurasi yg keluar jujur & realistis, termasuk ikut
    kena penalti kalau kandidatnya memang tidak ke-cover OCR).

    PENTING: kandidat yg dipaksa masuk disisipkan di POSISI ACAK (bukan selalu
    di ujung list). Kalau selalu ditaruh di akhir, `candidate_position` jadi
    fitur BOCOR/artifak -> model belajar "posisi terakhir = sering positif",
    padahal itu murni efek cara kita nyuntik data, bukan pola klik manusia asli.
    Pola itu juga tidak ada di val/test (krn tidak di-force), jadi bisa bikin
    model salah kalibrasi antara train vs val/test.
    """
    has_next = "next_article_id" in states.columns
    rng = np.random.default_rng(random_state)
    records = []
    for row in states.itertuples():
        cur, tgt = row.current_article_id, row.target_article_id
        true_next = row.next_article_id if has_next else None

        cand_list = list(candidates.get(cur, []))
        if force_true_positive and true_next is not None and true_next not in cand_list:
            insert_at = int(rng.integers(0, len(cand_list) + 1))  # posisi ACAK, bukan selalu akhir
            cand_list = cand_list[:insert_at] + [true_next] + cand_list[insert_at:]
        if not cand_list:
            cand_list = [fallback_article_id]  # jaga2, EDA-G blng ini harusnya 0 kasus

        n = len(cand_list)
        for pos, cand in enumerate(cand_list):
            rec = {
                "state_id": row.state_id,
                "current_article_id": cur,
                "target_article_id": tgt,
                "candidate_article_id": cand,
                "candidate_position": pos,
                "n_candidates": n,
            }
            if has_next:
                rec["is_chosen"] = int(cand == true_next)
            records.append(rec)
    return pd.DataFrame(records)


def analyze_feature_correlation(train_pairs: pd.DataFrame, numeric_cols: list):
    """Korelasi (point-biserial, krn target biner) tiap fitur numerik vs is_chosen
    -> analisis data BERDASARKAN preprocessing baru (fitur TF-IDF dkk), sebelum
    masuk ke pemilihan model."""
    print("\n" + "=" * 50)
    print("ANALISIS KORELASI: fitur numerik vs is_chosen (preprocessing baru)")
    print("=" * 50)
    corr = train_pairs[numeric_cols + ["is_chosen"]].corr(numeric_only=True)["is_chosen"].drop("is_chosen")
    corr = corr.sort_values(key=lambda s: s.abs(), ascending=False)
    print(corr.to_string())

    # IV utk fitur TF-IDF kontinu -> di-binning dulu jadi kuartil (biar sebanding
    # dgn analisis WoE/IV yg sudah dipakai di EDA-E/EDA-F sebelumnya)
    if "tfidf_sim_delta" in train_pairs.columns:
        binned = pd.qcut(train_pairs["tfidf_sim_delta"], q=5, duplicates="drop")
        tmp = train_pairs[["is_chosen"]].copy()
        tmp["tfidf_sim_delta_bin"] = binned.astype(str)
        woe_tbl, iv = _compute_woe_iv(tmp, "tfidf_sim_delta_bin", "is_chosen")
        print("\nWoE/IV untuk tfidf_sim_delta (dibinning 5 kuantil) terhadap is_chosen:")
        print(woe_tbl[["total", "events", "non_events", "woe", "iv"]])
        print(f"Total IV tfidf_sim_delta: {iv:.5f}")
    print()


def preprocess_candidate_data(data: dict):
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.model_selection import train_test_split
    import joblib

    FEATURES_DIR.mkdir(exist_ok=True)

    articles = data["articles"]
    categories = data["categories"]
    states_train = data["states_train"].copy()
    states_test = data["states_test"].copy()

    cat_map = build_category_map(categories)
    dims = extract_screenshot_dims(articles)
    height_median = dims["height"].median()
    pop = build_popularity_features(states_train)

    ocr_df = extract_screenshot_ocr(articles)
    ocr_feat, title_map = build_ocr_features(articles, ocr_df)

    print("\n" + "=" * 50)
    print("PREPROCESSING: TF-IDF index dari teks OCR (model 'membaca' isi teks asli)")
    print("=" * 50)
    tfidf_vectorizer, X_tfidf, aid_to_row = build_tfidf_index(ocr_feat)

    title_lookup = build_title_lookup(articles)
    matcher = build_candidate_matcher(title_lookup)
    candidates = extract_candidates_from_ocr(ocr_feat, matcher)

    fallback_article_id = int(states_train["next_article_id"].value_counts().idxmax())
    print(f"[fallback] artikel hub paling populer (dipakai kalau candidate list kosong): "
          f"{fallback_article_id}")

    print("\n" + "=" * 50)
    print("PREPROCESSING: Split train/val per-state (SEBELUM expand ke candidate-pairs)")
    print("=" * 50)
    train_states, val_states = train_test_split(states_train, test_size=0.15, random_state=42)
    print(f"train_states={len(train_states)} | val_states={len(val_states)}")

    print("\n" + "=" * 50)
    print("PREPROCESSING: Build candidate pairs (1 baris/state -> 1 baris/(state,kandidat))")
    print("=" * 50)
    train_pairs = build_candidate_pairs(train_states, candidates, fallback_article_id,
                                         force_true_positive=True)
    val_pairs = build_candidate_pairs(val_states, candidates, fallback_article_id,
                                       force_true_positive=False)
    test_pairs = build_candidate_pairs(states_test, candidates, fallback_article_id,
                                        force_true_positive=False)
    print(f"train_pairs={len(train_pairs)} (positif={train_pairs['is_chosen'].sum()}) | "
          f"val_pairs={len(val_pairs)} (positif={val_pairs['is_chosen'].sum()}) | "
          f"test_pairs={len(test_pairs)}")

    train_pairs = add_candidate_features(train_pairs, cat_map, pop, dims, height_median,
                                          X_tfidf, aid_to_row)
    val_pairs = add_candidate_features(val_pairs, cat_map, pop, dims, height_median,
                                        X_tfidf, aid_to_row)
    test_pairs = add_candidate_features(test_pairs, cat_map, pop, dims, height_median,
                                         X_tfidf, aid_to_row)

    categorical_cols = ["cur_top", "tgt_top", "cand_top"]
    numeric_cols = ["cat_match_cur_tgt", "cat_match_cand_tgt", "cat_match_cur_cand",
                     "candidate_is_target", "cand_height_log", "cand_popularity",
                     "cand_hub_indegree", "tgt_hub_indegree",
                     "candidate_position_norm", "n_candidates_log",
                     "tfidf_sim_cand_tgt", "tfidf_sim_cur_cand", "tfidf_sim_cur_tgt",
                     "tfidf_sim_delta"]

    analyze_feature_correlation(train_pairs, numeric_cols)

    print("\n" + "=" * 50)
    print("PREPROCESSING: One-Hot Encoding + Standard Scaling (level pasangan kandidat)")
    print("=" * 50)
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ("num", StandardScaler(), numeric_cols),
    ])
    X_train = preprocessor.fit_transform(train_pairs)
    X_val = preprocessor.transform(val_pairs)
    X_test = preprocessor.transform(test_pairs)
    print(f"Shape X_train={X_train.shape} | X_val={X_val.shape} | X_test={X_test.shape}")

    y_train = train_pairs["is_chosen"].values
    y_val = val_pairs["is_chosen"].values
    print(f"Positive rate -> train={y_train.mean():.4f} | val={y_val.mean():.4f}")

    joblib.dump(preprocessor, FEATURES_DIR / "preprocessor_candidate.joblib")
    joblib.dump(fallback_article_id, FEATURES_DIR / "fallback_article_id.joblib")
    joblib.dump(tfidf_vectorizer, FEATURES_DIR / "tfidf_vectorizer.joblib")
    states_test[["state_id"]].to_csv(FEATURES_DIR / "test_state_ids.csv", index=False)
    print(f"\nArtefak preprocessing disimpan ke: {FEATURES_DIR.resolve()}")

    return {
        "X_train": X_train, "y_train": y_train, "train_pairs": train_pairs,
        "X_val": X_val, "y_val": y_val, "val_pairs": val_pairs, "val_states": val_states,
        "X_test": X_test, "test_pairs": test_pairs, "states_test": states_test,
        "preprocessor": preprocessor, "fallback_article_id": fallback_article_id,
    }

#model ml
import time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.ensemble import RandomForestClassifier
import joblib

MODEL_DIR = Path("models")


def compare_binary_classifiers(X_train, y_train, X_val, y_val, val_pairs, val_states):
    """RESET model: jangan langsung pakai RandomForest, tapi banding beberapa
    model sklearn yg cocok utk data ini (fitur campuran kategorikal one-hot +
    numerik + similarity teks) -> pilih yg PALING SELARAS scr empiris, bukan
    asumsi di awal."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.utils.class_weight import compute_sample_weight

    print("\n" + "=" * 50)
    print("MODEL: reset & bandingkan beberapa classifier sklearn (level pasangan kandidat)")
    print("=" * 50)

    sample_weight = compute_sample_weight("balanced", y_train)

    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=2000),
        "GaussianNB": GaussianNB(),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=20, n_jobs=-1, random_state=42),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=300, random_state=42),
    }

    results, fitted = {}, {}
    for name, clf in candidates.items():
        t0 = time.time()
        clf.fit(X_train, y_train, sample_weight=sample_weight)
        val_scores = clf.predict_proba(X_val)[:, 1]
        acc = state_level_accuracy(val_pairs, val_scores, val_states)
        results[name] = acc
        fitted[name] = clf
        print(f"  {name:22s} | val_acc(state-level)={acc:.4f} | waktu latih={time.time() - t0:.1f}s")

    best_name = max(results, key=results.get)
    print(f"\n>>> Model tree/linear terbaik: {best_name} (val_acc={results[best_name]:.4f})")
    return fitted[best_name], best_name, results


class CandidateScorer(nn.Module):
    """MLP biner (dilatih via backpropagation): output = 1 logit skor
    'seberapa mungkin KANDIDAT INI yg diklik berikutnya'."""

    def __init__(self, input_dim: int, hidden=(128, 64), dropout: float = 0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_neural_net_binary(X_train, y_train, X_val, y_val, device,
                             epochs=60, batch_size=512, lr=1e-3, patience=10):
    print("\n" + "=" * 50)
    print(f"MODEL: Neural Network biner (MLP, backpropagation) di device={device}")
    print("=" * 50)

    model = CandidateScorer(X_train.shape[1]).to(device)

    Xtr_t = torch.tensor(X_train, dtype=torch.float32)
    ytr_t = torch.tensor(y_train, dtype=torch.float32)
    Xval_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    yval_t = torch.tensor(y_val, dtype=torch.float32).to(device)

    train_loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=batch_size, shuffle=True)

    n_pos = max((y_train == 1).sum(), 1)
    n_neg = (y_train == 0).sum()
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_loss, best_state, epochs_no_improve = float("inf"), None, 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()  # backpropagation
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        train_loss = total_loss / len(train_loader.dataset)

        model.eval()
        with torch.no_grad():
            val_logits = model(Xval_t)
            val_loss = nn.functional.binary_cross_entropy_with_logits(val_logits, yval_t).item()
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"epoch {epoch:3d}/{epochs} | train_loss={train_loss:.4f} | "
                  f"val_loss={val_loss:.4f} | best_val_loss={best_val_loss:.4f}")

        if epochs_no_improve >= patience:
            print(f"[early stopping] tidak ada perbaikan {patience} epoch berturut-turut, stop di epoch {epoch}")
            break

    model.load_state_dict(best_state)
    print("MLP biner selesai dilatih")
    return model


def predict_proba_binary_mlp(model, X, device, batch_size=4096):
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i + batch_size], dtype=torch.float32).to(device)
            probs.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(probs, axis=0)


def pick_best_candidate(pairs: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    """Utk tiap state, ambil KANDIDAT dgn skor tertinggi -> 1 prediksi/state."""
    tmp = pairs[["state_id", "candidate_article_id"]].copy()
    tmp["score"] = scores
    idx = tmp.groupby("state_id")["score"].idxmax()
    best = tmp.loc[idx, ["state_id", "candidate_article_id"]].rename(
        columns={"candidate_article_id": "predicted_next_article_id"})
    return best.reset_index(drop=True)


def state_level_accuracy(pairs: pd.DataFrame, scores: np.ndarray, true_states: pd.DataFrame) -> float:
    """Metrik yg SEBENARNYA dipakai kompetisi: exact-match per state, BUKAN
    akurasi biner per baris kandidat."""
    best = pick_best_candidate(pairs, scores)
    truth = true_states.set_index("state_id")["next_article_id"]
    best["true_next"] = best["state_id"].map(truth)
    return (best["predicted_next_article_id"] == best["true_next"]).mean()


def find_best_ensemble_weight_ranking(val_pairs, tree_scores, mlp_scores, val_states):
    print("\nPencarian bobot ensemble (alpha=bobot model tree/linear terpilih, 1-alpha=bobot MLP), "
          "dievaluasi di LEVEL STATE:")
    best_alpha, best_acc, results = 0.5, -1.0, []
    for alpha in np.arange(0.0, 1.01, 0.1):
        combined = alpha * tree_scores + (1 - alpha) * mlp_scores
        acc = state_level_accuracy(val_pairs, combined, val_states)
        results.append((round(alpha, 2), acc))
        if acc > best_acc:
            best_acc, best_alpha = acc, alpha
    for a, acc in results:
        marker = "  <== best" if a == round(best_alpha, 2) else ""
        print(f"  alpha={a:.1f} | val_acc(state-level)={acc:.4f}{marker}")
    return best_alpha, best_acc


def train_candidate_ranking_model(prep: dict, device: torch.device):
    MODEL_DIR.mkdir(exist_ok=True)

    X_train, y_train = prep["X_train"], prep["y_train"]
    X_val, y_val = prep["X_val"], prep["y_val"]
    val_pairs, val_states = prep["val_pairs"], prep["val_states"]

    best_tree_model, best_tree_name, all_results = compare_binary_classifiers(
        X_train, y_train, X_val, y_val, val_pairs, val_states)
    tree_val_scores = best_tree_model.predict_proba(X_val)[:, 1]

    mlp = train_neural_net_binary(X_train, y_train, X_val, y_val, device)
    mlp_val_scores = predict_proba_binary_mlp(mlp, X_val, device)
    mlp_state_acc = state_level_accuracy(val_pairs, mlp_val_scores, val_states)
    print(f"[MLP] val_acc (state-level, standalone) = {mlp_state_acc:.4f}")

    best_alpha, best_ens_acc = find_best_ensemble_weight_ranking(
        val_pairs, tree_val_scores, mlp_val_scores, val_states)
    print(f"\n>>> Ensemble terbaik: alpha={best_alpha:.1f} ({best_tree_name}) / "
          f"{1 - best_alpha:.1f} (MLP) | val_acc(state-level)={best_ens_acc:.4f}")

    naive_acc = (val_states["next_article_id"] == prep["fallback_article_id"]).mean()
    print(f"\n[sanity check] Baseline naif (selalu tebak artikel hub {prep['fallback_article_id']}) "
          f"-> val_acc = {naive_acc:.4f}")
    print("(Plafon realistis pendekatan ini dibatasi coverage kandidat dari EDA-G: ~57.28% utk next_article_id)")

    joblib.dump(best_tree_model, MODEL_DIR / "tree_candidate.joblib")
    joblib.dump(best_tree_name, MODEL_DIR / "tree_candidate_name.joblib")
    torch.save(mlp.state_dict(), MODEL_DIR / "mlp_candidate_state_dict.pt")
    joblib.dump({"input_dim": X_train.shape[1]}, MODEL_DIR / "mlp_candidate_config.joblib")
    joblib.dump({"alpha": best_alpha}, MODEL_DIR / "ensemble_weight_candidate.joblib")
    print(f"\nModel disimpan ke folder: {MODEL_DIR.resolve()}")

    return best_tree_model, mlp, best_alpha


#submission
def generate_submission(tree_model, mlp, alpha: float, prep: dict, device: torch.device,
                         out_path: Path = Path("submission.csv")):
    print("\n" + "=" * 50)
    print("SUBMISSION: generate prediksi (candidate-ranking) untuk test set")
    print("=" * 50)

    X_test, test_pairs = prep["X_test"], prep["test_pairs"]

    tree_scores = tree_model.predict_proba(X_test)[:, 1]
    mlp_scores = predict_proba_binary_mlp(mlp, X_test, device)
    combined = alpha * tree_scores + (1 - alpha) * mlp_scores

    best = pick_best_candidate(test_pairs, combined)

    states_test = prep["states_test"]
    submission = states_test[["state_id"]].merge(best, on="state_id", how="left")
    missing = submission["predicted_next_article_id"].isna().sum()
    if missing:
        print(f"[warning] {missing} state_id tanpa prediksi -> fallback ke artikel hub populer")
        submission["predicted_next_article_id"] = submission["predicted_next_article_id"].fillna(
            prep["fallback_article_id"])
    submission["predicted_next_article_id"] = submission["predicted_next_article_id"].astype(int)

    submission.to_csv(out_path, index=False)
    print(f"Submission disimpan ke: {out_path.resolve()}")
    print(f"Shape: {submission.shape}")
    print(submission.head())
    return submission


if __name__ == "__main__":
    data = load_data()

    for name, df in data.items():
        print(f"\n=== {name} | shape={df.shape} ===")
        print(df.head())

    n_screens = len(list(SCREENSHOT_DIR.glob("*.png")))
    print(f"\nJumlah file screenshot: {n_screens}")
    print(f"Jumlah artikel di articles.csv: {len(data['articles'])}")

    run_eda(data)

    prep = preprocess_candidate_data(data)

    rf, mlp, best_alpha = train_candidate_ranking_model(prep, device)

    generate_submission(rf, mlp, best_alpha, prep, device)