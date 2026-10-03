"""
Payment Screenshot Analyzer (UPI / bill receipts) - risk-score based.
Run:  streamlit run app.py
"""
import hashlib
import io
import re
import sqlite3
from datetime import datetime

try:
    import pymupdf as fitz  # PyMuPDF (naya naam)
except ImportError:
    import fitz  # purane versions
import numpy as np
import streamlit as st
from PIL import Image, ImageChops
from rapidocr_onnxruntime import RapidOCR

DB = "seen.db"
EDITOR_WORDS = ["photoshop", "canva", "picsart", "snapseed", "gimp", "pixlr",
                "lightroom", "illustrator", "ilovepdf", "smallpdf", "sejda",
                "editor", "fotor", "paint.net", "coreldraw"]
MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


# ---------------------------------------------------------------- loading
MAX_PAGES = 20  # cloud memory/time bachane ke liye
MON_RE = "Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec"
AMOUNT_LABELS = ("amount", "total", "paid", "fee", "charge", "gst", "debited", "bill",
                 "sent", "received", "balance")


@st.cache_resource
def get_ocr():
    return RapidOCR()


def ocr_text(img):
    """RapidOCR chalao aur boxes ko line-wise (upar se neeche, left se right) text me jodo."""
    result, _ = get_ocr()(np.asarray(img))  # RGB array (BGR par chhota text miss ho raha tha)
    items = []
    for box, txt, _score in result or []:
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        items.append((sum(ys) / 4, min(xs), max(ys) - min(ys), txt))
    items.sort()
    lines, cur, cy, ch = [], [], None, 1
    for y, x, h, txt in items:
        if cy is not None and abs(y - cy) <= 0.6 * ch:
            cur.append((x, txt))
        else:
            if cur:
                lines.append(cur)
            cur, cy, ch = [(x, txt)], y, max(h, 1)
    if cur:
        lines.append(cur)
    return "\n".join("  ".join(t for _, t in sorted(ln)) for ln in lines)


def load_file(data: bytes, name: str):
    """Return ([(PIL image, text), ...] one entry per page, metadata dict, kind)."""
    meta = {}
    if name.lower().endswith(".pdf"):
        doc = fitz.open(stream=data, filetype="pdf")
        meta = {k: v for k, v in (doc.metadata or {}).items() if v}
        pages = []
        for p in doc:
            if len(pages) >= MAX_PAGES:
                break
            pix = p.get_pixmap(dpi=200)
            img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            text = p.get_text()
            if len(text.strip()) < 30:  # scanned / image-only page
                text = ocr_text(img)
            pages.append((img, text))
        return pages, meta, "pdf"
    img = Image.open(io.BytesIO(data))
    try:
        exif = img.getexif()
        meta = {str(k): str(v) for k, v in exif.items()}
        meta.update({k: str(v) for k, v in img.info.items() if isinstance(v, (str, bytes))})
    except Exception:
        pass
    img = img.convert("RGB")
    text = ocr_text(img)
    return [(img, text)], meta, "image"


# ---------------------------------------------------------------- helpers
def parse_dates(text):
    out = []

    def add(y, m, d):
        try:
            out.append(datetime(int(y), int(m), int(d)).date())
        except ValueError:
            pass

    # 02 May 2026 / 02May2026 / 2-May-2026 (OCR kabhi spaces hata deta hai)
    for d, m, y in re.findall(
            rf"(?<!\d)(\d{{1,2}})\s*[\-/]?\s*({MON_RE})[a-z]*\.?,?\s*[\-/]?\s*(\d{{4}})(?!\d)", text, re.I):
        add(y, MONTHS[m.lower()], d)
    # May 02, 2026
    for m, d, y in re.findall(
            rf"(?<![A-Za-z])({MON_RE})[a-z]*\.?\s*(\d{{1,2}}),?\s*(\d{{4}})(?!\d)", text, re.I):
        add(y, MONTHS[m.lower()], d)
    # 02/05/2026
    for d, m, y in re.findall(r"(?<!\d)(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})(?!\d)", text):
        add(y, m, d)
    return out


def parse_times(text):
    out = []
    for h, m, ap in re.findall(r"(?<!\d)(\d{1,2}):(\d{2})(?::\d{2})?\s*([AaPp][Mm])?", text):
        h, m = int(h), int(m)
        if ap:
            h = h % 12 + (12 if ap.lower() == "pm" else 0)
        if h < 24 and m < 60:
            out.append(h * 60 + m)
    return out


def parse_amounts(text):
    """OCR kabhi ₹ hata deta hai, isliye 3 tarike: currency ke saath, comma format, label wali line."""
    vals = []
    for line in text.splitlines():
        low = line.lower()
        # 1) ₹12,833 / Rs 500 / INR 1,200.50
        vals += re.findall(r"(?<![a-z])(?:₹|rs\.?|inr)\s*([\d,]+(?:\.\d{1,2})?)", low)
        # 2) comma wale amounts: 12,833
        vals += re.findall(r"(?<![\d,.])(\d{1,3}(?:,\d{2,3})+(?:\.\d{1,2})?)(?![\d,])", line)
        # 3) amount-type label wali line ke chhote numbers: "fee + 3", "Total 500"
        if any(w in low for w in AMOUNT_LABELS):
            clean = re.sub(r"\d{1,2}:\d{2}(?::\d{2})?\s*(?:[ap]m)?", " ", line, flags=re.I)
            clean = re.sub(rf"\d{{1,2}}\s*[\-/]?\s*(?:{MON_RE})[a-z]*\.?,?\s*[\-/]?\s*\d{{4}}", " ", clean, flags=re.I)
            clean = re.sub(r"\d{1,2}[/\-]\d{1,2}[/\-]\d{4}", " ", clean)
            vals += re.findall(r"(?<![\d,.:/\-])(\d{1,7}(?:\.\d{1,2})?)(?![\d,:/\-])", clean)
    out = []
    for v in vals:
        try:
            out.append(float(v.replace(",", "")))
        except ValueError:
            pass
    return out


def embedded_date_from_id(token):
    """Tokens like NX251014124825... / T2510141248... -> (date, minutes or None)."""
    m = re.match(r"^[A-Za-z]{1,4}(20)?(2[3-7])(\d\d)(\d\d)(\d\d)?(\d\d)?", token)
    if not m:
        return None
    yy, mm, dd = int(m.group(2)), int(m.group(3)), int(m.group(4))
    try:
        d = datetime(2000 + yy, mm, dd).date()
    except ValueError:
        return None
    mins = None
    if m.group(5) and m.group(6):
        h, mi = int(m.group(5)), int(m.group(6))
        if h < 24 and mi < 60:
            mins = h * 60 + mi
    return d, mins


def ela_score(img):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    diff = np.asarray(ImageChops.difference(img, Image.open(buf)).convert("L"), dtype=float)
    h, w = diff.shape
    bs = 16
    blocks = diff[:h // bs * bs, :w // bs * bs].reshape(h // bs, bs, w // bs, bs).mean(axis=(1, 3))
    return float(np.percentile(blocks, 99.5) / (blocks.mean() + 1e-6))


def sha(data):
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------- checks
# each check returns (name, status, weight, detail); status: pass/warn/fail/info
def run_checks(img, text, meta, kind, data, page=1):
    R = []
    add = lambda n, s, w, d: R.append((n, s, w, d))
    today = datetime.now().date()

    # 1. metadata / editing software (file-level, sirf pehle page ke saath)
    if page == 1:
        blob = " ".join(f"{k}={v}" for k, v in meta.items()).lower()
        hit = [w for w in EDITOR_WORDS if w in blob]
        if hit:
            add("Metadata: editing software", "fail", 35, f"Editor ka naam mila: {', '.join(hit)}")
        else:
            add("Metadata: editing software", "pass" if meta else "info", 0,
                "Koi editor nahi mila" if meta else "Metadata nahi mila (WhatsApp se hat jata hai)")
        if kind == "pdf" and meta.get("creationDate") and meta.get("modDate") \
                and meta["creationDate"] != meta["modDate"]:
            add("PDF: creation vs modified time", "warn", 15, "Dono time alag hain (file baad me modify hui)")

    # 2. dates
    dates = parse_dates(text)
    times = parse_times(text)
    if dates:
        if any(d > today for d in dates):
            add("Date: future", "fail", 40, "Screenshot me aane wali date hai")
        else:
            add("Date: future", "pass", 0, f"Dates: {', '.join(str(d) for d in sorted(set(dates)))}")
    else:
        add("Date mila?", "warn", 10, "OCR se date nahi mili")

    # 3. ID embedded date vs displayed date
    tokens = re.findall(r"\b[A-Za-z]{1,4}[A-Za-z0-9]{12,}\b", text)
    found = False
    for t in tokens:
        e = embedded_date_from_id(t)
        if not e:
            continue
        found = True
        ed, emin = e
        if dates and ed not in dates:
            add("ID date vs displayed date", "fail", 45,
                f"ID {t} ki date {ed} hai, par screenshot me {', '.join(str(d) for d in sorted(set(dates)))}")
        elif dates:
            add("ID date vs displayed date", "pass", 0, f"{t} ki date match hai")
        if emin is not None and times:
            ok = any(min(abs(emin - x), 1440 - abs(emin - x)) <= 5 for x in times)
            add("ID time vs displayed time", "pass" if ok else "warn", 0 if ok else 15,
                "Time match" if ok else f"ID me time {emin // 60:02d}:{emin % 60:02d}, screenshot me alag")
        break
    if not found:
        add("ID date vs displayed date", "info", 0, "ID me date pattern nahi mila (app-specific rule chahiye)")

    # 4. UTR / 12 digit ref
    utrs = re.findall(r"(?<!\d)\d{12}(?!\d)", text)
    if not utrs:
        add("UTR / Ref number", "warn", 10, "12 digit UTR nahi mila")
    else:
        bad = [u for u in utrs if len(set(u)) <= 3 or u in "01234567890123456789"
               or u in "98765432109876543210"]
        add("UTR / Ref number", "fail" if bad else "pass", 30 if bad else 0,
            f"Suspicious pattern: {bad[0]}" if bad else f"UTR: {utrs[0]}")

    # 5. amounts consistency
    amts = parse_amounts(text)
    u = sorted(set(amts))
    if len(u) <= 1:
        add("Amount consistency", "pass" if u else "warn", 0 if u else 10,
            f"Amount: {u[0]:,.0f}" if u else "Amount OCR se nahi mila")
    else:
        ok = any(a + b == c or abs(a + b - c) < 0.01 for a in u for b in u for c in u if a != c and b != c)
        add("Amount consistency / math", "pass" if ok else "warn", 0 if ok else 20,
            "Bill + fee = total sahi" if ok else f"Amounts mismatch: {u}")

    # 6. ELA (weak signal, sirf image files par; PDF render par ELA ka matlab nahi)
    try:
        if kind != "image":
            raise ValueError("ELA skip for PDF pages")
        s = ela_score(img)
        add("Image forensics (ELA)", "warn" if s > 6 else "pass", 15 if s > 6 else 0,
            f"Score {s:.1f} - kuch region alag compress hue" if s > 6 else f"Score {s:.1f} normal")
    except Exception:
        pass

    # 7. duplicate (record baad me hota hai, taaki ek hi PDF ke pages aapas me duplicate na dikhen)
    file_key = sha(data) if page == 1 else f"{sha(data)}-p{page}"
    keys = [("file", file_key)] + [("utr", x) for x in utrs[:1]]
    con = sqlite3.connect(DB)
    con.execute("CREATE TABLE IF NOT EXISTS seen(kind TEXT, val TEXT, ts TEXT, UNIQUE(kind,val))")
    dup = [k for k, v in keys if con.execute("SELECT 1 FROM seen WHERE kind=? AND val=?", (k, v)).fetchone()]
    con.close()
    if dup:
        add("Duplicate check", "fail", 40, "Ye screenshot/UTR pehle bhi use ho chuka hai")
    else:
        add("Duplicate check", "pass", 0, "Pehle nahi dikha")
    return R, keys


def record_seen(keys):
    con = sqlite3.connect(DB)
    con.execute("CREATE TABLE IF NOT EXISTS seen(kind TEXT, val TEXT, ts TEXT, UNIQUE(kind,val))")
    for k, v in keys:
        con.execute("INSERT OR IGNORE INTO seen VALUES(?,?,?)", (k, v, datetime.now().isoformat()))
    con.commit()
    con.close()


def verdict(R):
    score = sum(w if s == "fail" else w * 0.5 if s == "warn" else 0 for _, s, w, _ in R)
    score = min(100, int(score))
    if score >= 45:
        return score, "❌ Likely FAKE / edited", "error"
    if score >= 15:
        return score, "⚠️ Suspicious - bank SMS/app me credit confirm karo", "warning"
    return score, "✅ Likely ORIGINAL (phir bhi credit confirm karna best hai)", "success"


# ---------------------------------------------------------------- UI
st.set_page_config(page_title="Payment Screenshot Analyzer", page_icon="🔍")
st.title("🔍 Payment Screenshot Analyzer")
f = st.file_uploader("Screenshot ya PDF upload karo", type=["png", "jpg", "jpeg", "webp", "pdf"])
record = st.checkbox("Duplicate check ke liye history me save karo", value=True)

if f:
    data = f.getvalue()
    with st.spinner("Check ho raha hai (saare pages)..."):
        pages, meta, kind = load_file(data, f.name)
        results = [run_checks(img, text, meta, kind, data, i + 1)
                   for i, (img, text) in enumerate(pages)]
        if record:
            record_seen([k for _, keys in results for k in keys])
    verdicts = [verdict(R) for R, _ in results]
    worst = max(range(len(verdicts)), key=lambda i: verdicts[i][0])
    score, msg, level = verdicts[worst]
    suffix = f"  \n(Sabse zyada risk Page {worst + 1} par, total {len(pages)} pages scan hue)" if len(pages) > 1 else ""
    getattr(st, level)(f"**{msg}**  \nRisk score: {score}/100{suffix}")
    if kind == "pdf" and len(pages) >= MAX_PAGES:
        st.info(f"Sirf pehle {MAX_PAGES} pages scan kiye gaye.")
    icons = {"pass": "✅", "warn": "⚠️", "fail": "❌", "info": "ℹ️"}
    for i, ((img, text), (R, _), (sc, vmsg, _)) in enumerate(zip(pages, results, verdicts)):
        title = f"Page {i + 1}: {vmsg} (risk {sc}/100)" if len(pages) > 1 else "Details"
        with st.expander(title, expanded=(len(pages) == 1 or i == worst)):
            st.image(img, width=300)
            st.table([{"": icons[s], "Check": n, "Detail": d} for n, s, _, d in R])
            st.text_area("OCR text", text, height=150, key=f"ocr{i}")
    st.caption("Ye sirf risk estimate hai, 100% proof nahi. Asli confirmation bank SMS/app me credit dekhkar hi hota hai.")
