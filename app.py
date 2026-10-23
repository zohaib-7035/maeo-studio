"""
MAEO Studio - showcase app (AIEF International AI Championship 2026)

Live on this laptop (CPU):
  * Secure Video Vault : AES-256-GCM + scrypt encryption into a .maeosec container, and decryption back to the video
  * Traffic Analyzer   : token-level motion analysis on MAEO's 16x16 grid, annotated preview + CSV report
  * Fleet Planner      : bandwidth / storage / cost plan from MAEO's measured bits per pixel
Replayed (real recorded results from Kaggle T4 runs, the neural codec needs a GPU):
  * codec results, step-by-step walkthrough

Start:  python -m streamlit run app.py
"""
import hashlib
import io
import json
import os
import struct
import tempfile
import time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
NAVY, BLUE, ORANGE, PAPER, GREY = "#0F1B2D", "#2457C5", "#D9772B", "#F6F5F0", "#9AA5B4"
TOK = 16

# ======================================================================= custom icon set (MAEO line icons)
_IC = {
    "logo": '<rect x="2.6" y="2.6" width="5.2" height="5.2" rx="1.4" fill="currentColor" stroke="none"/>'
            '<rect x="9.4" y="2.6" width="5.2" height="5.2" rx="1.4"/>'
            '<rect x="16.2" y="2.6" width="5.2" height="5.2" rx="1.4" fill="currentColor" stroke="none"/>'
            '<rect x="2.6" y="9.4" width="5.2" height="5.2" rx="1.4"/>'
            '<rect x="9.4" y="9.4" width="5.2" height="5.2" rx="1.4" fill="currentColor" stroke="none"/>'
            '<rect x="16.2" y="9.4" width="5.2" height="5.2" rx="1.4"/>'
            '<rect x="2.6" y="16.2" width="5.2" height="5.2" rx="1.4" fill="currentColor" stroke="none"/>'
            '<rect x="9.4" y="16.2" width="5.2" height="5.2" rx="1.4"/>'
            '<rect x="16.2" y="16.2" width="5.2" height="5.2" rx="1.4" fill="currentColor" stroke="none"/>',
    "overview": '<rect x="3" y="3" width="7.5" height="9" rx="2"/><rect x="13.5" y="3" width="7.5" height="5" rx="2"/>'
                '<rect x="13.5" y="11" width="7.5" height="10" rx="2"/><rect x="3" y="15" width="7.5" height="6" rx="2"/>',
    "vault": '<path d="M12 2.8l7.5 3v5.4c0 4.6-3.1 8.5-7.5 10-4.4-1.5-7.5-5.4-7.5-10V5.8z"/>'
             '<circle cx="12" cy="10.6" r="2.1"/><path d="M12 12.7v3.4"/>',
    "storage": '<ellipse cx="12" cy="5.5" rx="7.5" ry="2.7"/><path d="M4.5 5.5v6.2c0 1.5 3.4 2.7 7.5 2.7s7.5-1.2 7.5-2.7V5.5"/>'
               '<path d="M4.5 11.7v6.6c0 1.5 3.4 2.7 7.5 2.7s7.5-1.2 7.5-2.7v-6.6"/><path d="M10 16.2l2 1.6 2-1.6"/>',
    "traffic": '<rect x="8" y="2.5" width="8" height="19" rx="3.2"/><circle cx="12" cy="7.2" r="1.6"/>'
               '<circle cx="12" cy="12" r="1.6"/><circle cx="12" cy="16.8" r="1.6"/>'
               '<path d="M8 7.2H5M8 12H5M16 7.2h3M16 12h3"/>',
    "fleet": '<circle cx="12" cy="5" r="2.3"/><circle cx="4.8" cy="18.5" r="2.3"/><circle cx="19.2" cy="18.5" r="2.3"/>'
             '<path d="M10.9 7.1L6 16.4M13.1 7.1l4.9 9.3M7.1 18.5h9.8"/><circle cx="12" cy="14" r="1.3"/>',
    "steps": '<rect x="2.5" y="5" width="10" height="14" rx="2"/><path d="M5.5 9h4M5.5 12h4M5.5 15h2.5"/>'
             '<rect x="16" y="9" width="5.5" height="6" rx="1.6"/><path d="M13 12h2.6M14.3 10.6l1.4 1.4-1.4 1.4"/>',
    "results": '<path d="M3 20.5h18"/><rect x="5" y="11" width="3.2" height="9.5" rx="1"/>'
               '<rect x="10.4" y="5.5" width="3.2" height="15" rx="1"/><rect x="15.8" y="13.5" width="3.2" height="7" rx="1"/>',
}


def icon(name, size=24, color="currentColor", sw=1.8):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="{color}" stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round" '
            f'style="color:{color};display:block">{_IC[name]}</svg>')


def icon_uri(name):
    import urllib.parse
    return "data:image/svg+xml;utf8," + urllib.parse.quote(icon(name, 24, "#000", 1.9))


NAV = [("Overview", "overview"), ("Secure Video Vault", "vault"), ("Compressed Storage", "storage"),
       ("Traffic Analyzer", "traffic"), ("Fleet Planner", "fleet"), ("How MAEO compresses", "steps"),
       ("Research results", "results")]
NAV_CSS = "".join(
    f'[data-testid="stSidebar"] [data-testid="stRadioOption"]:has(input[value="{i}"]) [data-testid="stMarkdownContainer"]::before'
    f'{{-webkit-mask-image:url("{icon_uri(k)}");mask-image:url("{icon_uri(k)}")}}\n' for i, (_, k) in enumerate(NAV))

st.set_page_config(page_title="MAEO Studio", page_icon=os.path.join(DATA, "logo.png"), layout="wide", initial_sidebar_state="expanded")

st.markdown(
    f"""
    <style>
    #MainMenu, footer {{visibility:hidden}}
    header[data-testid="stHeader"] {{background:transparent}}
    .block-container {{padding-top:1.4rem}}
    div[data-testid="stMetric"] {{background:#FDFCF9;border:1px solid #D9DEE6;border-radius:12px;padding:10px 14px}}
    section[data-testid="stSidebar"] {{background:{NAVY}}}
    section[data-testid="stSidebar"] * {{color:#E6ECF5}}
    .hero {{background:linear-gradient(120deg,{NAVY} 0%,#1A2F55 100%);padding:24px 30px;border-radius:16px;margin-bottom:10px}}
    .hero h1 {{color:{PAPER};margin:0;font-size:40px}}
    .hero p {{color:#C5D3E3;margin:4px 0 0 0;font-size:18px}}
    .eyebrow {{color:#8FB3F0;font-size:13px;letter-spacing:2px;font-weight:600}}
    .card {{background:#FDFCF9;border:1px solid #D9DEE6;border-radius:14px;padding:16px 18px;height:100%}}
    .card h4 {{margin:0 0 6px 0;color:{NAVY}}}
    .badge {{display:inline-block;padding:3px 10px;border-radius:999px;font-size:12px;font-weight:600;margin-right:6px}}
    .live {{background:#DDF3E4;color:#1B6B35}}
    .replay {{background:#E6ECF5;color:#1E3F8A}}
    .step-chip {{display:inline-block;padding:6px 12px;margin:2px 4px 2px 0;border-radius:999px;font-size:14px;background:#E6ECF5;color:#1E3F8A}}
    .step-chip.on {{background:{BLUE};color:{PAPER};font-weight:600}}
    .step-chip.done {{background:#D6E2F7;color:#1E3F8A}}
    .hero-row {{display:flex;gap:18px;align-items:center}}
    .hero-ic {{flex:0 0 auto;width:64px;height:64px;border-radius:16px;background:rgba(143,179,240,.14);
              border:1px solid rgba(143,179,240,.35);display:flex;align-items:center;justify-content:center;color:#8FB3F0}}
    .card-top {{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px}}
    .ic-tile {{width:44px;height:44px;border-radius:12px;background:#E6ECF5;color:{BLUE};display:flex;align-items:center;justify-content:center}}
    .brand {{display:flex;gap:12px;align-items:center}}
    .brand-ic {{width:44px;height:44px;border-radius:12px;background:{BLUE};color:{PAPER};display:flex;align-items:center;justify-content:center}}
    [data-testid="stSidebar"] [data-testid="stRadioOption"] {{padding:9px 12px;border-radius:10px;margin:0 0 4px 0;width:100%;cursor:pointer;transition:background .15s}}
    [data-testid="stSidebar"] [data-testid="stRadioGroup"], [data-testid="stSidebar"] [data-testid="stRadioGroup"] > div {{width:100%}}
    [data-testid="stSidebar"] [data-testid="stRadioOption"]:hover {{background:rgba(143,179,240,.12)}}
    [data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div:first-child {{display:none !important}}
    [data-testid="stSidebar"] [data-testid="stRadioOption"] [data-testid="stMarkdownContainer"] {{display:flex;align-items:center;gap:12px}}
    [data-testid="stSidebar"] [data-testid="stRadioOption"] [data-testid="stMarkdownContainer"] p {{margin:0;font-size:15px}}
    [data-testid="stSidebar"] [data-testid="stRadioOption"] [data-testid="stMarkdownContainer"]::before {{content:"";flex:0 0 20px;width:20px;height:20px;background-color:#8FB3F0;
              -webkit-mask-repeat:no-repeat;mask-repeat:no-repeat;-webkit-mask-size:contain;mask-size:contain}}
    [data-testid="stSidebar"] [data-testid="stRadioOption"]:has(input:checked) {{background:{BLUE}}}
    [data-testid="stSidebar"] [data-testid="stRadioOption"]:has(input:checked) [data-testid="stMarkdownContainer"]::before {{background-color:#FFFFFF}}
    [data-testid="stSidebar"] [data-testid="stRadioOption"]:has(input:checked) p {{color:#FFFFFF !important;font-weight:600}}
    {NAV_CSS}
    </style>
    """,
    unsafe_allow_html=True,
)


# ======================================================================= helpers
def show_image(img, caption=None):
    for kw in ({"width": "stretch"}, {"use_container_width": True}, {"use_column_width": True}):
        try:
            st.image(img, caption=caption, **kw)
            return
        except Exception:
            continue
    st.image(img, caption=caption)


def show_table(df):
    for kw in ({"width": "stretch", "hide_index": True}, {"use_container_width": True, "hide_index": True}, {}):
        try:
            st.dataframe(df, **kw)
            return
        except Exception:
            continue


def rerun():
    (st.rerun if hasattr(st, "rerun") else st.experimental_rerun)()


def hero(eyebrow, title, sub, badge=None, ic=None):
    b = ""
    if badge == "live":
        b = "<span class='badge live'>● LIVE · REAL PROCESSING</span>"
    elif badge == "replay":
        b = "<span class='badge replay'>RECORDED RESULTS</span>"
    tile = f"<div class='hero-ic'>{icon(ic, 34)}</div>" if ic else ""
    st.markdown(f"<div class='hero'><div class='hero-row'>{tile}<div><div class='eyebrow'>{eyebrow}</div><h1>{title}</h1>"
                f"<p>{sub}</p></div></div><div style='margin-top:10px'>{b}</div></div>", unsafe_allow_html=True)


@st.cache_data
def load_results():
    with open(os.path.join(DATA, "results.json"), "r", encoding="utf-8") as f:
        r = json.load(f)
    return r, pd.DataFrame(r["mean"]), pd.DataFrame(r["per_sequence"])


R, MEAN, SEQ = load_results()
LAMS = sorted(MEAN["lam"].unique())
LAM_LABEL = {0.005: "Economy (λ 0.005)", 0.015: "Standard (λ 0.015)", 0.045: "Balanced (λ 0.045)", 0.12: "High quality (λ 0.120)"}
CODEC_NAME = {"P4-I3": "Intra codec (each frame alone)", "P4-V3": "Temporal codec (GOP-8: 1 I + 7 P)"}


def row(exp, lam):
    return MEAN[(MEAN.exp == exp) & (np.isclose(MEAN.lam, lam))].iloc[0]


def mbps(bpp, w, h, fps):
    return bpp * w * h * fps / 1e6


def lattice_mask(gh, gw, retention=0.45):
    """MAEO's real uniform token rule (R2 low-discrepancy lattice) - same code as the codec."""
    k = max(1, int(round(retention * gw * gh)))
    r = np.arange(gh, dtype=np.float64)[:, None]
    c = np.arange(gw, dtype=np.float64)[None, :]
    v = np.mod(r * 0.7548776662466927 + c * 0.5698402909980532, 1.0).ravel()
    keep = np.zeros(gh * gw, dtype=bool)
    keep[np.argsort(v, kind="stable")[:k]] = True
    return keep.reshape(gh, gw)


def draw_grid(img, tok=TOK, keep=None, color=(36, 87, 197)):
    base = img.convert("RGB")
    w, h = base.size
    gw, gh = w // tok, h // tok
    if keep is None:
        out = base.copy()
        d = ImageDraw.Draw(out)
        for j in range(gw + 1):
            d.line([(j * tok, 0), (j * tok, gh * tok)], fill=color, width=1)
        for i in range(gh + 1):
            d.line([(0, i * tok), (gw * tok, i * tok)], fill=color, width=1)
        return out
    out = Image.blend(base, Image.new("RGB", base.size, (15, 27, 45)), 0.6)
    d = ImageDraw.Draw(out)
    for i in range(gh):
        for j in range(gw):
            if keep[i, j]:
                b = (j * tok, i * tok, (j + 1) * tok, (i + 1) * tok)
                out.paste(base.crop(b), b[:2])
                d.rectangle(b, outline=color)
    return out


def fmt_bytes(n):
    for unit, div in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if n >= div:
            return f"{n/div:.2f} {unit}"
    return f"{n} B"


# ======================================================================= sidebar
with st.sidebar:
    st.markdown(f"<div class='brand'><div class='brand-ic'>{icon('logo', 28)}</div><div>"
                f"<div style='font-size:24px;font-weight:800;color:{PAPER};line-height:1.1'>MAEO Studio</div>"
                "<div style='color:#8FB3F0;font-size:12.5px'>Meaning-aware video compression</div></div></div>",
                unsafe_allow_html=True)
    st.markdown("---")
    PAGES = [n for n, _ in NAV]
    page = st.radio("Navigate", PAGES, label_visibility="collapsed")
    st.markdown("---")
    st.markdown("<div style='font-size:13px;line-height:1.5'>"
                "<b>Team</b><br>Mian Afzal Saeed<br>Muhammad Zohaib Shahid<br><br>"
                "<b>Supervisor</b><br>Dr. Asma Ahmad Farhan<br><br>FAST NUCES Lahore<br>"
                "4th AIEF International AI Championship 2026</div>", unsafe_allow_html=True)


# ======================================================================= page: overview
def pipeline_figure():
    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.set_xlim(0, 13); ax.set_ylim(0, 4.2); ax.axis("off")
    fig.patch.set_facecolor(PAPER)

    def box(x, y, w, h, title, sub, fc="#FDFCF9", ec=GREY, tc=NAVY, dashed=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                                    fc=fc, ec=ec, lw=2, ls="--" if dashed else "-"))
        ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", fontsize=12, weight="bold", color=tc)
        ax.text(x + w / 2, y + h * 0.3, sub, ha="center", va="center", fontsize=9.5, color=tc)

    def arrow(x1, y1, x2, y2, c="#5B6675", ls="-"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color=c, lw=2, ls=ls))

    box(0.1, 0.9, 1.7, 2.2, "Input", "video frame")
    box(2.3, 2.15, 2.2, 1.05, "Meaning", "frozen DINO, 64 ch")
    box(2.3, 0.75, 2.2, 1.05, "Pixels", "neural encoder, 128 ch")
    box(5.0, 2.15, 2.2, 1.05, "Free mask", "keep 45%, 0 bits")
    box(5.0, 0.75, 2.2, 1.05, "Entropy coding", "every bit billed")
    box(7.7, 0.75, 2.0, 2.45, "One file", "MAE4 / MAE5\nSHA-256 model ID", fc=BLUE, ec=BLUE, tc=PAPER)
    box(10.2, 0.9, 2.6, 2.2, "Decoder", "file only →\nrebuilt video")
    box(2.3, 3.45, 4.9, 0.65, "AI control layer (next phase)", "agents decide where bits go",
        fc="#FDF3EA", ec=ORANGE, tc="#8A4310", dashed=True)
    for y in (2.65, 1.3):
        arrow(1.8, y, 2.3, y); arrow(4.5, y, 5.0, y); arrow(7.2, y, 7.7, y)
    arrow(9.7, 1.98, 10.2, 1.98, c=BLUE)
    arrow(3.4, 3.45, 3.4, 3.2, c=ORANGE, ls="--"); arrow(6.1, 3.45, 6.1, 3.2, c=ORANGE, ls="--")
    fig.tight_layout()
    return fig


def page_overview():
    hero("AI VIDEO COMPRESSION · AIEF 2026", "MAEO Studio",
         "Spend bits on what the receiver needs, not just on pixels. Built for cameras that are watched by AI.", ic="logo")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mask cost", "0 bits", "decoder rebuilds it")
    c2.metric("Semantic bits saved", "−72%", "Wyner–Ziv prior")
    c3.metric("Temporal vs intra", "−52.6 to −60.6%", "BD-rate, 3 seeds")
    c4.metric("AI still agrees", f"{row('P4-V3', 0.045).seg_acc*100:.1f}%", "segmentation, balanced")
    st.markdown("### Products")
    a, s_, b, c = st.columns(4)
    s_.markdown("<div class='card'><div class='card-top'><div class='ic-tile'>" + icon("storage", 24) + "</div><span class='badge live'>LIVE</span></div><h4>Compressed Storage</h4>"
                "Upload footage, compress it, check the quality and keep it in a searchable library. "
                "Play back or download any stored video.</div>", unsafe_allow_html=True)
    a.markdown("<div class='card'><div class='card-top'><div class='ic-tile'>" + icon("vault", 24) + "</div><span class='badge live'>LIVE</span></div><h4>Secure Video Vault</h4>"
               "Encrypt a video into a sealed <b>.maeosec</b> file with AES-256-GCM. Upload it with the password to get the "
               "original video back. Any change to the file is detected and refused.</div>", unsafe_allow_html=True)
    b.markdown("<div class='card'><div class='card-top'><div class='ic-tile'>" + icon("traffic", 24) + "</div><span class='badge live'>LIVE</span></div><h4>Traffic Analyzer</h4>"
               "Upload a traffic camera video. MAEO’s 16×16 token grid finds where vehicles move, frame by frame, "
               "and exports an annotated preview and a CSV report.</div>", unsafe_allow_html=True)
    c.markdown("<div class='card'><div class='card-top'><div class='ic-tile'>" + icon("fleet", 24) + "</div><span class='badge live'>LIVE</span></div><h4>Fleet Planner</h4>"
               "Plan a city camera network: uplink bandwidth, storage and monthly cost from MAEO’s measured "
               "bits per pixel. Download the plan as CSV.</div>", unsafe_allow_html=True)
    st.markdown("### How it works")
    st.pyplot(pipeline_figure())
    st.caption("Codec numbers are real recorded results from our Kaggle T4 GPU runs (UVG 1080p, 7 sequences × 96 frames, "
               "every frame decoded from a real file). MAEO is a research prototype and still behind x265 on pixel quality per bit.")


# ======================================================================= page: secure vault
MAGIC = b"MAEOSEC1"


class WrongPassword(Exception):
    pass


class Tampered(Exception):
    pass


def key_check(key: bytes) -> str:
    # short public check value derived from the key: tells "wrong password" apart from "file was changed"
    return hashlib.sha256(key + b"MAEO-vault-key-check").hexdigest()[:16]


def vault_encrypt(data: bytes, filename: str, password: str) -> bytes:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
    salt, nonce = os.urandom(16), os.urandom(12)
    key = Scrypt(salt=salt, length=32, n=2 ** 14, r=8, p=1).derive(password.encode("utf-8"))
    header = {"product": "MAEO Secure Video Vault", "version": 2, "cipher": "AES-256-GCM", "kdf": "scrypt(n=16384,r=8,p=1)",
              "filename": filename, "size": len(data), "sha256": hashlib.sha256(data).hexdigest(),
              "created": datetime.now().isoformat(timespec="seconds"), "salt": salt.hex(), "nonce": nonce.hex(),
              "key_check": key_check(key)}
    hb = json.dumps(header, separators=(",", ":")).encode("utf-8")
    ct = AESGCM(key).encrypt(nonce, data, hb)            # header is authenticated too
    return MAGIC + struct.pack("<I", len(hb)) + hb + ct


def vault_read_header(blob: bytes):
    if len(blob) < 12 or blob[:8] != MAGIC:
        raise ValueError("Not a MAEO Secure Video file (.maeosec).")
    n = struct.unpack("<I", blob[8:12])[0]
    hb = blob[12:12 + n]
    return json.loads(hb.decode("utf-8")), hb, blob[12 + n:]


def vault_decrypt(blob: bytes, password: str):
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
    from cryptography.exceptions import InvalidTag
    try:
        header, hb, ct = vault_read_header(blob)
    except ValueError:
        raise
    except Exception:
        raise Tampered("The file header is damaged.")
    key = Scrypt(salt=bytes.fromhex(header["salt"]), length=32, n=2 ** 14, r=8, p=1).derive(password.encode("utf-8"))
    if "key_check" in header and key_check(key) != header["key_check"]:
        raise WrongPassword("Wrong password. The file stays locked.")
    try:
        data = AESGCM(key).decrypt(bytes.fromhex(header["nonce"]), ct, hb)
    except InvalidTag:
        if "key_check" in header:
            raise Tampered("The file was changed after it was encrypted.")
        raise WrongPassword("Wrong password, or the file was changed after encryption.")
    if hashlib.sha256(data).hexdigest() != header["sha256"]:
        raise Tampered("Integrity check failed.")
    return header, data


def page_vault():
    hero("SECURE VIDEO · AES-256-GCM", "Secure Video Vault",
         "Seal a video into one encrypted file. Only the right password opens it, and any tampering is refused.", "live", ic="vault")
    t1, t2 = st.tabs(["① Encrypt a video", "② Decrypt a .maeosec file"])
    with t1:
        c1, c2 = st.columns([1.1, 1])
        with c1:
            up = st.file_uploader("Video to protect", type=["mp4", "mov", "avi", "mkv", "webm", "m4v"], key="v_up")
            use_sample = st.checkbox("Use the sample traffic clip instead", value=up is None, key="v_sample")
            pw = st.text_input("Password", type="password", key="v_pw", help="At least 6 characters. Share it separately.")
            go = st.button("Encrypt and seal", type="primary", key="v_go", icon=":material/lock:")
        if up is not None and not use_sample:
            data, name = up.getvalue(), up.name
        else:
            data, name = open(os.path.join(DATA, "sample_clip.mp4"), "rb").read(), "sample_traffic_clip.mp4"
        with c2:
            st.video(data)
            st.caption(f"{name} · {fmt_bytes(len(data))}")
        if go:
            if len(pw) < 6:
                st.error("Please use a password of at least 6 characters.")
            else:
                t0 = time.perf_counter()
                blob = vault_encrypt(data, name, pw)
                st.session_state.vault_out = (blob, name, (time.perf_counter() - t0) * 1000)
        if "vault_out" in st.session_state:
            blob, oname, ms = st.session_state.vault_out
            hdr, _, _ = vault_read_header(blob)
            st.success(f"Sealed {oname} in {ms:.0f} ms.")
            m1, m2, m3 = st.columns(3)
            m1.metric("Original", fmt_bytes(hdr["size"]))
            m2.metric("Encrypted file", fmt_bytes(len(blob)))
            m3.metric("Cipher", "AES-256-GCM", "scrypt key")
            st.code(f"SHA-256 of original : {hdr['sha256']}\nEncrypted (last 32 bytes): {blob[-32:].hex()}", language="text")
            st.download_button("Download encrypted file", icon=":material/download:", data= blob, file_name=os.path.splitext(oname)[0] + ".maeosec",
                               mime="application/octet-stream", key="v_dl")
            st.caption("Open this file in tab ② with the same password. The video inside cannot be played without it.")
    with t2:
        c1, c2 = st.columns([1, 1.1])
        with c1:
            enc = st.file_uploader("Encrypted .maeosec file", type=None, key="d_up")
            if enc is None and "vault_out" in st.session_state:
                st.info("Tip: no file chosen, so the file you just sealed in tab ① will be used.")
            dpw = st.text_input("Password", type="password", key="d_pw")
            tamper = st.checkbox("Attack demo: secretly change one byte of the file before opening", key="d_tamper",
                                 help="Leave this OFF for a normal decrypt. Turn it ON to show that a changed file is refused.")
            dgo = st.button("Verify and decrypt", type="primary", key="d_go", icon=":material/lock_open:")
        if dgo:
            if enc is not None:
                blob = enc.getvalue()
            elif "vault_out" in st.session_state:
                blob = st.session_state.vault_out[0]
            else:
                blob = None
                st.error("Choose a .maeosec file first.")
            if blob is not None:
                if tamper:
                    b = bytearray(blob); b[-10] ^= 0x01; blob = bytes(b)
                try:
                    t0 = time.perf_counter()
                    hdr, data = vault_decrypt(blob, dpw)
                    st.session_state.vault_dec = (hdr, data, (time.perf_counter() - t0) * 1000)
                    st.session_state.vault_err = None
                except WrongPassword as e:
                    st.session_state.vault_dec = None
                    st.session_state.vault_err = ("password", str(e))
                except Tampered as e:
                    st.session_state.vault_dec = None
                    st.session_state.vault_err = ("tamper", str(e))
                except Exception as e:
                    st.session_state.vault_dec = None
                    st.session_state.vault_err = ("other", str(e))
        with c2:
            err = st.session_state.get("vault_err")
            dec = st.session_state.get("vault_dec")
            if err:
                kind, msg = err if isinstance(err, tuple) else ("other", str(err))
                if kind == "tamper":
                    st.warning("**Attack blocked.** " + msg + " The vault refused to open it, so nothing was shown. "
                               "This is the protection working.", icon=":material/gpp_good:")
                    st.caption("To decrypt normally, untick the attack demo box and press Verify and decrypt again.")
                elif kind == "password":
                    st.error(msg, icon=":material/key_off:")
                    st.caption("Use the same password you typed when sealing the file.")
                else:
                    st.error(msg, icon=":material/block:")
            elif dec:
                hdr, data, ms = dec
                st.success(f"Verified and decrypted in {ms:.0f} ms. SHA-256 matches the original.", icon=":material/verified:")
                st.video(data)
                a, b = st.columns(2)
                a.metric("File", hdr["filename"])
                b.metric("Sealed on", hdr["created"].replace("T", " "))
                st.download_button("Download decrypted video", icon=":material/download:", data= data, file_name=hdr["filename"], key="d_dl")
    st.caption("MAEO is a codec, not a cipher: in the full pipeline the compressed MAEO file is sealed this way. "
               "Inside MAEO, each file also carries a SHA-256 fingerprint of the decoder model, so a wrong model is refused.")


# ======================================================================= page: traffic analyzer
def analyze_video(path, max_frames, stride, width, thr):
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("Could not open this video.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    rows, previews, prev, idx, used = [], [], None, 0, 0
    heat = None
    while used < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride:
            idx += 1
            continue
        h0, w0 = frame.shape[:2]
        w = min(width, w0 - w0 % TOK)
        h = int(round(h0 * w / w0))
        h -= h % TOK
        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        g = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)
        gh, gw = h // TOK, w // TOK
        if prev is not None and prev.shape == g.shape:
            # camera-shake compensation: estimate the global shift and align the previous frame before comparing
            (dx, dy), _ = cv2.phaseCorrelate(prev, g)
            aligned = cv2.warpAffine(prev, np.float32([[1, 0, dx], [0, 1, dy]]), (g.shape[1], g.shape[0]),
                                     borderMode=cv2.BORDER_REPLICATE)
            d = np.abs(g - aligned)[: gh * TOK, : gw * TOK].reshape(gh, TOK, gw, TOK).mean(axis=(1, 3))
            moving = d > thr
            heat = moving.astype(np.float32) if heat is None else heat + moving
            img = Image.fromarray(rgb)
            over = img.copy()
            dr = ImageDraw.Draw(over)
            ys, xs = np.nonzero(moving)
            for i, j in zip(ys, xs):
                dr.rectangle((j * TOK, i * TOK, (j + 1) * TOK - 1, (i + 1) * TOK - 1), fill=(217, 119, 43))
            out = Image.blend(img, over, 0.45)
            dr2 = ImageDraw.Draw(out)
            for i, j in zip(ys, xs):
                dr2.rectangle((j * TOK, i * TOK, (j + 1) * TOK - 1, (i + 1) * TOK - 1), outline=(217, 119, 43))
            dr2.rectangle((0, 0, 330, 30), fill=(15, 27, 45))
            dr2.text((8, 8), f"t={idx/fps:5.2f}s  moving tokens: {int(moving.sum())}/{gh*gw}", fill=(246, 245, 240))
            previews.append(out)
            rows.append({"frame": idx, "time_s": round(idx / fps, 3), "moving_tokens": int(moving.sum()),
                         "total_tokens": gh * gw, "moving_pct": round(100.0 * moving.mean(), 2),
                         "mean_change": round(float(d.mean()), 3)})
        prev = g
        used += 1
        idx += 1
    cap.release()
    return pd.DataFrame(rows), previews, heat, fps, total, (w, h) if rows else (0, 0)


def to_gif(frames, fps):
    buf = io.BytesIO()
    if frames:
        small = [f.resize((f.size[0] * 480 // f.size[0], f.size[1] * 480 // f.size[0])) for f in frames]
        small[0].save(buf, format="GIF", save_all=True, append_images=small[1:], duration=int(1000 / max(1, fps)),
                      loop=0, optimize=True)
    return buf.getvalue()


def mp4_frame_sizes(path):
    """Real per-frame sizes (bytes) of the first track, read from the MP4 'stsz' box."""
    with open(path, "rb") as fh:
        blob = fh.read()
    i = blob.find(b"stsz")
    if i < 4:
        return []
    fixed, count = struct.unpack(">II", blob[i + 8:i + 16])
    if fixed:
        return [fixed] * count
    return list(struct.unpack(f">{count}I", blob[i + 16:i + 16 + 4 * count]))


def token_motion_masks(frames_gray, thr):
    """Per-frame boolean masks of moving 16x16 tokens (camera-shake compensated frame differencing)."""
    import cv2
    masks, prev = [], None
    for g in frames_gray:
        gh, gw = g.shape[0] // TOK, g.shape[1] // TOK
        if prev is None:
            masks.append(np.zeros((gh, gw), dtype=bool))
        else:
            (dx, dy), _ = cv2.phaseCorrelate(prev, g)
            al = cv2.warpAffine(prev, np.float32([[1, 0, dx], [0, 1, dy]]), (g.shape[1], g.shape[0]),
                                borderMode=cv2.BORDER_REPLICATE)
            d = np.abs(g - al)[: gh * TOK, : gw * TOK].reshape(gh, TOK, gw, TOK).mean(axis=(1, 3))
            masks.append(d > thr)
        prev = g
    if len(masks) > 1:
        masks[0] = masks[1].copy()
    return masks


def importance_maps(masks):
    """Important tokens = moving tokens, grown by one token and held over neighbouring frames (no flicker)."""
    import cv2
    k = np.ones((3, 3), np.uint8)
    grown = [cv2.dilate(m.astype(np.uint8), k) > 0 for m in masks]
    out = []
    for t in range(len(grown)):
        m = grown[t].copy()
        for s in (t - 2, t - 1, t + 1, t + 2):
            if 0 <= s < len(grown):
                m |= grown[s]
        out.append(m)
    return out


def encode_frames(frames_bgr, fps, crf, dst):
    import subprocess
    h, w = frames_bgr[0].shape[:2]
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
           "-r", f"{fps:.3f}", "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", dst]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        for f in frames_bgr:
            p.stdin.write(np.ascontiguousarray(f).tobytes())
        p.stdin.close()
    except BrokenPipeError:
        pass
    err = p.stderr.read().decode(errors="ignore")
    if p.wait() != 0:
        raise RuntimeError(err[-400:] or "ffmpeg failed")


def read_frames(path, max_frames, width):
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("Could not open this video.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while len(frames) < max_frames:
        ok, f = cap.read()
        if not ok:
            break
        h0, w0 = f.shape[:2]
        w = min(width, w0); w -= w % TOK
        h = int(round(h0 * w / w0)); h -= h % TOK
        frames.append(cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA))
    cap.release()
    return frames, fps


def decode_all(path):
    import cv2
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    cap.release()
    return out


def stream_quality(path, ref, masks, keep_idx):
    """PSNR on important tokens and on the background, decoding the file one frame at a time (low memory)."""
    import cv2
    cap = cv2.VideoCapture(path)
    num_in = den_in = num_out = den_out = 0.0
    kept, t = None, 0
    while t < len(ref):
        ok, b = cap.read()
        if not ok:
            break
        a, m = ref[t], masks[t]
        px = np.kron(m, np.ones((TOK, TOK), dtype=bool))[: a.shape[0], : a.shape[1]]
        e = ((a.astype(np.float32) - b.astype(np.float32)) ** 2).mean(axis=2)
        num_in += e[px].sum(); den_in += px.sum(); num_out += e[~px].sum(); den_out += (~px).sum()
        if t == keep_idx:
            kept = b.copy()
        t += 1
    cap.release()
    f = lambda n_, d_: (10 * np.log10(255.0 ** 2 / max(n_ / d_, 1e-6)) if d_ else float("nan"))
    return (f(num_in, den_in), f(num_out, den_out)), (kept if kept is not None else ref[keep_idx]), t


def region_psnr(ref, test, masks):
    num_in = den_in = num_out = den_out = 0.0
    for a, b, m in zip(ref, test, masks):
        px = np.kron(m, np.ones((TOK, TOK), dtype=bool))[: a.shape[0], : a.shape[1]]
        e = ((a.astype(np.float32) - b.astype(np.float32)) ** 2).mean(axis=2)
        num_in += e[px].sum(); den_in += px.sum()
        num_out += e[~px].sum(); den_out += (~px).sum()
    f = lambda n, d: (10 * np.log10(255.0 ** 2 / max(n / d, 1e-6)) if d else None)
    return f(num_in, den_in), f(num_out, den_out)


def smart_compress(path, max_frames, width, thr, crf, bg_strength):
    import cv2
    frames, fps = read_frames(path, max_frames, width)
    if len(frames) < 3:
        raise RuntimeError("The video is too short.")
    gray = [cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32) for f in frames]
    masks = token_motion_masks(gray, thr)
    imp = importance_maps(masks)
    smart = []
    for f, m in zip(frames, imp):
        px = np.kron(m, np.ones((TOK, TOK), dtype=np.uint8))[: f.shape[0], : f.shape[1]].astype(np.float32)
        px = cv2.GaussianBlur(px, (0, 0), 4)[..., None]               # soft edge between important / background
        bg = cv2.GaussianBlur(f, (0, 0), bg_strength)                 # background keeps shape, loses fine detail
        smart.append((f * px + bg * (1 - px)).astype(np.uint8))
    tmpd = tempfile.mkdtemp(prefix="maeo_traffic_")
    std_p, smt_p = os.path.join(tmpd, "standard.mp4"), os.path.join(tmpd, "maeo_smart.mp4")
    t0 = time.perf_counter()
    encode_frames(frames, fps, crf, std_p)
    t_std = time.perf_counter() - t0
    t0 = time.perf_counter()
    encode_frames(smart, fps, max(18, crf - 2), smt_p)            # a little extra quality on the vehicles
    del smart
    t_smt = time.perf_counter() - t0
    n = len(frames)
    busiest = int(np.argmax([m.mean() for m in masks]))
    q_std, f_std, n_std = stream_quality(std_p, frames, imp, busiest)
    q_smt, f_smt, n_smt = stream_quality(smt_p, frames, imp, busiest)
    n = min(n, n_std, n_smt)
    sz_std, sz_smt = mp4_frame_sizes(std_p), mp4_frame_sizes(smt_p)
    heat = np.mean([m.astype(np.float32) for m in imp], axis=0)
    df = pd.DataFrame({"frame": np.arange(n), "time_s": np.round(np.arange(n) / fps, 3),
                       "moving_tokens_pct": [round(100 * float(m.mean()), 2) for m in masks[:n]],
                       "important_tokens_pct": [round(100 * float(m.mean()), 2) for m in imp[:n]],
                       "standard_bytes": (sz_std + [np.nan] * n)[:n], "maeo_smart_bytes": (sz_smt + [np.nan] * n)[:n]})
    h, w = frames[0].shape[:2]
    return dict(fps=fps, w=w, h=h, n=n, raw_bytes=w * h * 3 * n, std=open(std_p, "rb").read(), smt=open(smt_p, "rb").read(),
                q_std=q_std, q_smt=q_smt, df=df, heat=heat, imp_area=float(np.mean([m.mean() for m in imp])),
                busiest=busiest, f_orig=frames[busiest], f_std=f_std, f_smt=f_smt,
                m_busy=imp[busiest], t_std=t_std, t_smt=t_smt)


def bgr2pil(f):
    import cv2
    return Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))


def importance_overlay(frame_bgr, heat):
    img = bgr2pil(frame_bgr).convert("RGB")
    gh, gw = heat.shape
    cmap = plt.get_cmap("inferno")
    rgba = (cmap(np.clip(heat / max(1e-6, heat.max()), 0, 1)) * 255).astype(np.uint8)
    hm = Image.fromarray(rgba[..., :3]).resize((gw * TOK, gh * TOK), Image.NEAREST).resize(img.size, Image.NEAREST)
    return Image.blend(img, hm, 0.6)


def crop_box(mask, w, h, size=192):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        cx, cy = w // 2, h // 2
    else:
        cy, cx = int(np.median(ys) * TOK + TOK / 2), int(np.median(xs) * TOK + TOK / 2)
    x0 = int(np.clip(cx - size // 2, 0, max(0, w - size))); y0 = int(np.clip(cy - size // 2, 0, max(0, h - size)))
    return (x0, y0, x0 + size, y0 + size)


def page_traffic():
    hero("SMART-CITY TRAFFIC · BITS WHERE THE AI NEEDS THEM", "Traffic Analyzer",
         "Give it a traffic camera video. ① Analyse the movement on MAEO’s 16×16 token grid. "
         "② Compress it so the vehicles keep their quality and the empty road costs almost nothing.", "live", ic="traffic")
    try:
        import cv2  # noqa: F401
    except Exception:
        st.error("OpenCV is missing. Close the app and run run.bat again to install it.")
        return
    if not ffmpeg_exe():
        st.error("The video engine is missing. Close the app and run run.bat again to install it.")
        return

    # ---------------- input
    st.markdown("### Input video")
    c1, c2 = st.columns([1, 1.2])
    with c1:
        up = st.file_uploader("Traffic camera video", type=["mp4", "mov", "avi", "mkv", "m4v"], key="t_up")
        st.caption("No file? The sample intersection clip is used.")
        with st.expander("Settings", expanded=False):
            max_frames = st.slider("Frames to process", 48, 300, 144, 24, key="t_nf")
            width = st.select_slider("Processing width (px)", options=[480, 640, 720, 960], value=640, key="t_w")
            thr = st.slider("Motion sensitivity (lower = more sensitive)", 2.0, 20.0, 9.0, 0.5, key="t_thr")
            crf = st.select_slider("Compression level (CRF)", options=[23, 26, 28, 30, 32], value=28, key="t_crf")
            bg = st.slider("Background saving strength", 1.0, 8.0, 4.0, 0.5, key="t_bg",
                           help="How much fine detail the static background gives up")
        b1, b2 = st.columns(2)
        go_a = b1.button("① Analyse movement", type="primary", key="t_go")
        go_c = b2.button("② Compress for traffic AI", type="primary", key="t_comp")
    if up is not None:
        key = (up.name, up.size)
        if st.session_state.get("t_src_key") != key:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(up.name)[1] or ".mp4")
            tmp.write(up.getvalue()); tmp.close()
            st.session_state.t_src = (tmp.name, up.name)
            st.session_state.t_src_key = key
            st.session_state.pop("traffic", None); st.session_state.pop("t_comp_res", None)
        src_path, src_name = st.session_state.t_src
    else:
        src_path, src_name = os.path.join(DATA, "sample_clip.mp4"), "sample_traffic_clip.mp4"
    with c2:
        st.video(src_path)
        st.caption(f"{src_name} · input file {fmt_bytes(os.path.getsize(src_path))}")

    # ---------------- ① analyse
    if go_a:
        with st.spinner("Analysing movement on MAEO’s token grid…"):
            t0 = time.perf_counter()
            try:
                df, prev, heat, fps, total, wh = analyze_video(src_path, min(120, max_frames), 2, min(width, 800), thr)
                st.session_state.traffic = dict(df=df, gif=to_gif(prev, fps / 2), heat=heat, fps=fps, wh=wh,
                                                secs=time.perf_counter() - t0,
                                                busiest=prev[int(df.moving_tokens.idxmax())] if len(df) else None)
            except Exception as e:
                st.error(f"Could not analyse this video: {e}")
    res = st.session_state.get("traffic")
    if res and not res["df"].empty:
        st.markdown("### ① Movement analysis")
        df = res["df"]
        a1, a2 = st.columns([1.2, 1])
        with a1:
            show_image(res["gif"], caption="Orange = moving tokens (vehicles, people). Animated preview.")
        with a2:
            m1, m2 = st.columns(2)
            m1.metric("Frames analysed", f"{len(df)}", f"{res['secs']:.1f} s to analyse")
            m2.metric("Tokens per frame", f"{int(df.total_tokens.iloc[0]):,}")
            m1.metric("Moving tokens", f"{df.moving_pct.mean():.1f}%", f"peak {df.moving_pct.max():.1f}%")
            m2.metric("Static tokens", f"{100 - df.moving_pct.mean():.1f}%", "can be predicted")
            if res["heat"] is not None:
                fig, ax = plt.subplots(figsize=(4.6, 3.0))
                ax.imshow(res["heat"] / max(1, len(df)), cmap="Oranges")
                ax.set_title("Where traffic moves (lanes)"); ax.axis("off")
                fig.tight_layout(); st.pyplot(fig)
        st.download_button("Download movement report (CSV)", icon=":material/download:", data= df.to_csv(index=False).encode(),
                           file_name="maeo_movement_report.csv", key="t_csv")

    # ---------------- ② compress
    if go_c:
        with st.spinner("Compressing twice (standard vs. traffic-aware) and measuring quality…"):
            try:
                st.session_state.t_comp_res = smart_compress(src_path, max_frames, width, thr, crf, bg)
            except Exception as e:
                st.error(f"Could not compress this video: {e}")
    cr = st.session_state.get("t_comp_res")
    if not cr:
        if not res:
            st.info("Press **① Analyse movement** to see where the traffic moves, then **② Compress for traffic AI**.")
        return
    st.markdown("### ② Compression: before vs after")
    s_std, s_smt = len(cr["std"]), len(cr["smt"])
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Before: raw frames", fmt_bytes(cr["raw_bytes"]), f"{cr['w']}×{cr['h']} · {cr['n']} frames", delta_color="off")
    k2.metric("Standard compression", fmt_bytes(s_std), f"{cr['raw_bytes']/s_std:.0f}× smaller than raw", delta_color="off")
    k3.metric("After: MAEO-style (traffic-aware)", fmt_bytes(s_smt), f"{cr['raw_bytes']/s_smt:.0f}× smaller than raw")
    k4.metric("Saved vs standard", f"{100*(1-s_smt/s_std):.0f}%", "vehicles at the same quality")
    v1, v2 = st.columns(2)
    with v1:
        st.markdown("**Standard compression** (same quality everywhere)")
        st.video(cr["std"])
        st.caption(fmt_bytes(s_std))
    with v2:
        st.markdown("**MAEO-style** (quality on vehicles, background saved)")
        st.video(cr["smt"])
        st.caption(fmt_bytes(s_smt))
    (vi_std, bg_std), (vi_smt, bg_smt) = cr["q_std"], cr["q_smt"]
    q = pd.DataFrame({"": ["Vehicles / moving areas", "Static background", "File size"],
                      "Standard": [f"{vi_std:.1f} dB", f"{bg_std:.1f} dB", fmt_bytes(s_std)],
                      "MAEO-style": [f"{vi_smt:.1f} dB", f"{bg_smt:.1f} dB", fmt_bytes(s_smt)]})
    st.markdown("**Quality where it matters** (PSNR vs the input, measured on every frame)")
    show_table(q)

    st.markdown("### Where the bits go")
    g1, g2 = st.columns([1.1, 1])
    with g1:
        show_image(importance_overlay(cr["f_orig"], cr["heat"]),
                   caption="Bright tokens = given more bits (where vehicles move). Dark = background, few bits.")
    with g2:
        area = cr["imp_area"] * 100
        fig, ax = plt.subplots(figsize=(5.6, 2.3))
        ax.barh(["Frame area"], [area], color=ORANGE, label="important tokens")
        ax.barh(["Frame area"], [100 - area], left=[area], color=GREY, label="background tokens")
        ax.set_xlim(0, 100); ax.set_xlabel("% of the frame")
        ax.text(area / 2, 0, f"{area:.0f}%", ha="center", va="center", color="white", weight="bold")
        ax.text(area + (100 - area) / 2, 0, f"{100-area:.0f}%", ha="center", va="center", color=NAVY, weight="bold")
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        fig.tight_layout(); st.pyplot(fig)
        st.write(f"Only **{area:.0f}%** of the frame carries the vehicles. MAEO-style coding keeps their quality "
                 f"({vi_smt:.1f} dB vs {vi_std:.1f} dB) and lets the background give up detail, so the file is "
                 f"**{100*(1-s_smt/s_std):.0f}% smaller** than standard compression.")

    st.markdown("### Which frames get more importance")
    df = cr["df"]
    win = max(3, int(round(cr["fps"] / 2)))                         # half-second average hides the encoder's key frames
    sm = df.maeo_smart_bytes.iloc[1:].rolling(win, center=True, min_periods=1).mean() / 1000
    sd = df.standard_bytes.iloc[1:].rolling(win, center=True, min_periods=1).mean() / 1000
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.fill_between(df.time_s.iloc[1:], sm, color=BLUE, alpha=0.25)
    ax.plot(df.time_s.iloc[1:], sm, color=BLUE, lw=2.2, label="MAEO-style bits per frame (KB, ½-s average)")
    ax.plot(df.time_s.iloc[1:], sd, color=GREY, lw=1.6, ls="--", label="standard bits per frame (KB, ½-s average)")
    ax.set_xlabel("time (s)"); ax.set_ylabel("KB per frame"); ax.set_ylim(bottom=0)
    ax2 = ax.twinx()
    ax2.plot(df.time_s, df.moving_tokens_pct, color=ORANGE, lw=2.2, label="traffic activity (% moving tokens)")
    ax2.set_ylabel("% moving tokens", color=ORANGE)
    bt = df.time_s.iloc[cr["busiest"]]
    ax.axvline(bt, color=ORANGE, ls="--", lw=1)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower left", bbox_to_anchor=(0, 1.02), ncol=3, fontsize=8.5, frameon=False)
    for s in ("top",):
        ax.spines[s].set_visible(False); ax2.spines[s].set_visible(False)
    fig.tight_layout(); st.pyplot(fig)
    corr = np.corrcoef(sm.values, df.moving_tokens_pct.iloc[1:].values)[0, 1] if len(sm) > 3 else float("nan")
    link = (f"When traffic is busy, more bits are spent (correlation {corr:.2f}); quiet moments cost less. " if corr == corr and corr > 0.2
            else f"Bits and traffic activity move together only weakly in this clip (correlation {corr:.2f}). ")
    st.caption("Blue = bits MAEO-style coding spends on each moment; orange = how much traffic moves. " + link +
               "Grey = standard coding, which spends bits on the background too. Dashed line = busiest frame. "
               "First (key) frame left out for scale.")

    st.markdown(f"**Busiest frame (t = {bt:.2f} s): zoom on the traffic**")
    box = crop_box(cr["m_busy"], cr["w"], cr["h"])
    z1, z2, z3 = st.columns(3)
    with z1:
        show_image(bgr2pil(cr["f_orig"]).crop(box).resize((288, 288), Image.NEAREST), caption="Input")
    with z2:
        show_image(bgr2pil(cr["f_std"]).crop(box).resize((288, 288), Image.NEAREST), caption=f"Standard · {fmt_bytes(s_std)}")
    with z3:
        show_image(bgr2pil(cr["f_smt"]).crop(box).resize((288, 288), Image.NEAREST), caption=f"MAEO-style · {fmt_bytes(s_smt)}")

    d1, d2, d3 = st.columns(3)
    d1.download_button("Download MAEO-style video", icon=":material/download:", data= cr["smt"], file_name="traffic_maeo_style.mp4", key="t_dl_smt")
    d2.download_button("Download standard video", icon=":material/download:", data= cr["std"], file_name="traffic_standard.mp4", key="t_dl_std")
    d3.download_button("Download per-frame report (CSV)", icon=":material/download:", data= df.to_csv(index=False).encode(),
                       file_name="maeo_traffic_bits_report.csv", key="t_dl_csv")
    st.info("**The MAEO goal in one line:** same bits, better where it matters. A normal codec treats the sky and a car the same; "
            "MAEO spends the bits on what the traffic AI needs.")
    st.caption("How it runs here: importance comes from live motion analysis on MAEO’s 16×16 token grid, and the video is "
               "encoded with standard H.264. In MAEO’s next phase (P5), AI agents for meaning, motion and the "
               "query decide the importance, and the neural codec spends the bits.")


# ======================================================================= page: compressed storage
STORE = os.path.join(HERE, "storage")
PRESETS = {"High quality (CRF 23)": 23, "Balanced (CRF 28)": 28, "Archive (CRF 33)": 33}
SCALES = {"Keep original size": None, "1080 px wide": 1080, "720 px wide": 720, "480 px wide": 480}


def ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        import shutil
        return shutil.which("ffmpeg")


def compress_video(src, dst, crf, width):
    import subprocess
    vf = ["-vf", f"scale={width}:-2"] if width else ["-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2"]
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-i", src, *vf, "-c:v", "libx264", "-preset", "veryfast",
           "-crf", str(crf), "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", dst]
    r = subprocess.run(cmd, capture_output=True, text=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip()[-400:] or "ffmpeg failed")


def measure_quality(orig, comp, samples=12):
    import cv2
    a, b = cv2.VideoCapture(orig), cv2.VideoCapture(comp)
    n = int(b.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = b.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(b.get(cv2.CAP_PROP_FRAME_WIDTH)), int(b.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = max(1, n // samples) if n else 5
    psnrs, i, pair = [], 0, None
    while len(psnrs) < samples:
        oka, fa = a.read(); okb, fb = b.read()
        if not (oka and okb):
            break
        if i % step == 0:
            fa = cv2.resize(fa, (fb.shape[1], fb.shape[0]), interpolation=cv2.INTER_AREA)
            mse = np.mean((fa.astype(np.float32) - fb.astype(np.float32)) ** 2)
            psnrs.append(99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse))
            if pair is None and i >= step:
                pair = (Image.fromarray(cv2.cvtColor(fa, cv2.COLOR_BGR2RGB)), Image.fromarray(cv2.cvtColor(fb, cv2.COLOR_BGR2RGB)))
        i += 1
    a.release(); b.release()
    return (float(np.mean(psnrs)) if psnrs else None), n, fps, (w, h), pair


def library():
    os.makedirs(STORE, exist_ok=True)
    items = []
    for f in sorted(os.listdir(STORE)):
        if f.endswith(".json"):
            try:
                with open(os.path.join(STORE, f), "r", encoding="utf-8") as fh:
                    items.append(json.load(fh))
            except Exception:
                pass
    return sorted(items, key=lambda d: d.get("stored", ""), reverse=True)


def page_storage():
    hero("VIDEO STORAGE · COMPRESS, STORE, PLAY BACK", "Compressed Video Storage",
         "Upload camera footage. It is compressed, quality-checked and kept in a searchable library.", "live", ic="storage")
    if not ffmpeg_exe():
        st.error("The video engine is missing. Close the app and run run.bat again to install it.")
        return
    t1, t2 = st.tabs(["① Compress and store", "② Storage library"])
    with t1:
        c1, c2 = st.columns([1, 1.1])
        with c1:
            up = st.file_uploader("Video to store", type=["mp4", "mov", "avi", "mkv", "m4v", "webm"], key="s_up")
            st.caption("No file? The sample traffic clip is used.")
            preset = st.radio("Compression level", list(PRESETS), index=1, key="s_preset", horizontal=True)
            scale = st.selectbox("Resolution", list(SCALES), index=0, key="s_scale")
            camera = st.text_input("Camera / tag", "Main Blvd – Camera 07", key="s_cam")
            go = st.button("Compress and store", type="primary", key="s_go", icon=":material/compress:")
        with c2:
            if up is None:
                st.video(os.path.join(DATA, "sample_clip.mp4"))
        if go:
            os.makedirs(STORE, exist_ok=True)
            if up is not None:
                src = os.path.join(tempfile.gettempdir(), "maeo_in_" + os.path.basename(up.name))
                with open(src, "wb") as fh:
                    fh.write(up.getvalue())
                name = up.name
            else:
                src, name = os.path.join(DATA, "sample_clip.mp4"), "sample_traffic_clip.mp4"
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base = f"{stamp}_{os.path.splitext(os.path.basename(name))[0][:40]}".replace(" ", "_")
            dst = os.path.join(STORE, base + ".mp4")
            with st.spinner("Compressing and checking quality…"):
                t0 = time.perf_counter()
                try:
                    compress_video(src, dst, PRESETS[preset], SCALES[scale])
                    secs = time.perf_counter() - t0
                    psnr, n, fps, (w, h), pair = measure_quality(src, dst)
                    meta = {"id": base, "file": base + ".mp4", "name": name, "camera": camera, "preset": preset,
                            "stored": datetime.now().isoformat(timespec="seconds"), "orig_bytes": os.path.getsize(src),
                            "comp_bytes": os.path.getsize(dst), "psnr": psnr, "frames": n, "fps": fps, "w": w, "h": h,
                            "secs": secs}
                    with open(os.path.join(STORE, base + ".json"), "w", encoding="utf-8") as fh:
                        json.dump(meta, fh, indent=1)
                    st.session_state.store_last = (meta, pair)
                except Exception as e:
                    st.error(f"Could not compress this video: {e}")
        last = st.session_state.get("store_last")
        if last:
            meta, pair = last
            ratio = meta["orig_bytes"] / max(1, meta["comp_bytes"])
            st.success(f"Stored **{meta['name']}** in {meta['secs']:.1f} s · saved {100*(1-1/ratio):.0f}% of the space.")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Before", fmt_bytes(meta["orig_bytes"]))
            m2.metric("After", fmt_bytes(meta["comp_bytes"]), f"{ratio:.1f}× smaller")
            m3.metric("Quality (PSNR)", f"{meta['psnr']:.1f} dB" if meta["psnr"] else "–", "vs. original")
            m4.metric("Video", f"{meta['w']}×{meta['h']}", f"{meta['frames']} frames")
            if pair:
                a, b = st.columns(2)
                with a:
                    show_image(pair[0], caption="Before: original frame")
                with b:
                    show_image(pair[1], caption="After: frame from the stored, compressed file")
            path = os.path.join(STORE, meta["file"])
            if os.path.exists(path):
                with open(path, "rb") as fh:
                    st.download_button("Download compressed video", icon=":material/download:", data= fh.read(), file_name=meta["name"].rsplit(".", 1)[0] + "_compressed.mp4",
                                       key="s_dl_last")
    with t2:
        items = library()
        if not items:
            st.info("The library is empty. Compress a video in tab ① first.")
        else:
            tot_o = sum(d["orig_bytes"] for d in items); tot_c = sum(d["comp_bytes"] for d in items)
            k1, k2, k3 = st.columns(3)
            k1.metric("Videos stored", f"{len(items)}")
            k2.metric("Space used", fmt_bytes(tot_c), f"was {fmt_bytes(tot_o)}")
            k3.metric("Space saved", fmt_bytes(tot_o - tot_c), f"{100*(1-tot_c/max(1,tot_o)):.0f}%")
            q = st.text_input("Search by name or camera", "", key="s_q").lower().strip()
            show = [d for d in items if not q or q in d["name"].lower() or q in d.get("camera", "").lower()]
            show_table(pd.DataFrame([{"Stored": d["stored"].replace("T", " "), "Camera": d.get("camera", ""), "Video": d["name"],
                                      "Level": d["preset"], "Before": fmt_bytes(d["orig_bytes"]), "After": fmt_bytes(d["comp_bytes"]),
                                      "Saved": f"{100*(1-d['comp_bytes']/max(1,d['orig_bytes'])):.0f}%",
                                      "PSNR (dB)": round(d["psnr"], 1) if d.get("psnr") else None} for d in show]))
            if show:
                pick = st.selectbox("Play a stored video", [f"{d['stored'].replace('T', ' ')} · {d['name']}" for d in show], key="s_pick")
                d = show[[f"{x['stored'].replace('T', ' ')} · {x['name']}" for x in show].index(pick)]
                path = os.path.join(STORE, d["file"])
                if os.path.exists(path):
                    with open(path, "rb") as fh:
                        vid = fh.read()
                    st.video(vid)
                    st.download_button("Download this video", icon=":material/download:", data= vid, file_name=d["name"].rsplit(".", 1)[0] + "_compressed.mp4", key="s_dl_pick")
    st.caption("Storage engine today: standard H.264 compression (runs on any CPU). MAEO’s neural codec needs a GPU and plugs in "
               "as the storage engine on GPU servers; its measured results are on the Research results page.")


# ======================================================================= page: fleet planner
RES = {"720p (1280×720)": (1280, 720), "1080p (1920×1080)": (1920, 1080), "4K (3840×2160)": (3840, 2160)}


def page_fleet():
    hero("CITY CAMERA NETWORKS · PLANNING", "Fleet Planner",
         "Plan bandwidth, storage and cost for a camera fleet, using MAEO’s measured bits per pixel.", "live", ic="fleet")
    c1, c2 = st.columns([1, 1.3])
    with c1:
        name = st.text_input("Project name", "Lahore Smart Traffic – Phase 1")
        cams = st.number_input("Number of cameras", 1, 100000, 500, 50)
        res_name = st.selectbox("Resolution", list(RES), index=1)
        fps = st.select_slider("Frames per second", options=[5, 10, 15, 25, 30], value=15)
        hours = st.slider("Recording hours per day", 1, 24, 24)
        days = st.number_input("Days of footage kept", 1, 365, 30, 1)
        lam = st.select_slider("MAEO quality level", options=LAMS, value=0.045, format_func=lambda x: LAM_LABEL[x], key="f_lam")
        price = st.number_input("Storage price (USD per TB per month)", 0.0, 500.0, 20.0, 1.0)
    w, h = RES[res_name]
    rv = row("P4-V3", lam)
    per_cam = mbps(rv.bpp, w, h, fps)
    raw_cam = mbps(24, w, h, fps)
    secs = hours * 3600 * days
    maeo_tb = per_cam * 1e6 / 8 * secs * cams / 1e12
    raw_tb = raw_cam * 1e6 / 8 * secs * cams / 1e12
    with c2:
        st.markdown(f"### {name}")
        a, b = st.columns(2)
        a.metric("Per-camera uplink", f"{per_cam:.2f} Mbps", f"raw {raw_cam:,.0f} Mbps", delta_color="off")
        b.metric("Total city uplink", f"{per_cam*cams/1000:,.2f} Gbps", f"{cams:,} cameras", delta_color="off")
        a.metric("Storage needed", f"{maeo_tb:,.1f} TB", f"raw {raw_tb:,.0f} TB", delta_color="off")
        b.metric("Storage cost", f"${maeo_tb*price:,.0f} / month", f"{raw_tb/maeo_tb:,.0f}× less than raw", delta_color="off")
        dd = np.arange(1, int(days) + 1)
        fig, ax = plt.subplots(figsize=(6.5, 2.8))
        ax.plot(dd, per_cam * 1e6 / 8 * hours * 3600 * dd * cams / 1e12, color=BLUE, lw=2.5, label="MAEO")
        ax.set_xlabel("days kept"); ax.set_ylabel("storage (TB)"); ax.grid(alpha=0.3); ax.legend()
        ax.set_title("Storage grows with retention")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout(); st.pyplot(fig)
        st.caption(f"Quality at this level (recorded on UVG 1080p): PSNR {rv.psnr_rgb:.1f} dB · MS-SSIM {rv.ms_ssim:.3f} · "
                   f"segmentation agreement {rv.seg_acc*100:.1f}%. Assumes the measured bits per pixel hold at this resolution.")
    st.markdown("#### Drone / IoT link check")
    l1, l2 = st.columns([1, 2])
    with l1:
        link = st.slider("Available link speed (Mbps)", 0.5, 60.0, 8.0, 0.5)
        dfps = st.select_slider("Drone video frame rate", options=[5, 10, 15, 30], value=10)
    tab = MEAN[MEAN.exp == "P4-V3"].copy()
    tab["Quality level"] = tab.lam.map(LAM_LABEL)
    tab["Mbps @1080p"] = tab.bpp.apply(lambda b_: mbps(b_, 1920, 1080, dfps))
    tab["Fits link"] = np.where(tab["Mbps @1080p"] <= link, "Yes", "No")
    fit = tab[tab["Mbps @1080p"] <= link].sort_values("bpp")
    with l2:
        if len(fit):
            bst = fit.iloc[-1]
            st.success(f"Best fit: **{bst['Quality level']}** · {bst['Mbps @1080p']:.1f} Mbps · PSNR {bst.psnr_rgb:.1f} dB · "
                       f"segmentation agreement {bst.seg_acc*100:.1f}%")
        else:
            st.warning("No 1080p level fits this link. Lower the frame rate or the resolution.")
        show_table(tab[["Quality level", "bpp", "Mbps @1080p", "psnr_rgb", "seg_acc", "Fits link"]]
                   .rename(columns={"psnr_rgb": "PSNR (dB)", "seg_acc": "Seg. agreement"}).round(3))
    plan = pd.DataFrame([{"project": name, "cameras": cams, "resolution": res_name, "fps": fps, "hours_per_day": hours,
                          "days_kept": days, "quality_level": LAM_LABEL[lam], "maeo_bpp": round(rv.bpp, 4),
                          "per_camera_mbps": round(per_cam, 3), "total_gbps": round(per_cam * cams / 1000, 3),
                          "storage_tb": round(maeo_tb, 2), "raw_storage_tb": round(raw_tb, 1),
                          "storage_cost_usd_month": round(maeo_tb * price, 0)}])
    st.download_button("Download plan (CSV)", icon=":material/download:", data= plan.to_csv(index=False).encode(), file_name="maeo_fleet_plan.csv", key="f_dl")
    st.caption("For reference: x265 low-delay reaches a similar PSNR at a lower bitrate today. MAEO’s goal is to keep the bits "
               "an AI needs; matching x265 on pixels is ongoing work.")


# ======================================================================= page: step by step
STEPS = ["1 · Before", "2 · Tokens", "3 · Meaning stream", "4 · Pixel stream", "5 · One file", "6 · After decoding"]


def size_bars(before_bytes, after_bytes):
    fig, ax = plt.subplots(figsize=(7.5, 2.0))
    ax.barh([1], [before_bytes / 1e6], color=GREY)
    ax.barh([0], [after_bytes / 1e6], color=BLUE)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["After: MAEO file", "Before: raw frame"])
    ax.set_xlabel("size (MB)")
    ax.text(before_bytes / 1e6, 1, f"  {before_bytes/1e6:.2f} MB", va="center")
    ax.text(after_bytes / 1e6, 0, f"  {after_bytes/1e3:.0f} KB", va="center", color=BLUE, weight="bold")
    ax.set_xlim(0, before_bytes / 1e6 * 1.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def page_steps():
    hero("WALKTHROUGH", "How MAEO compresses", "One traffic frame, step by step: before → after.", "replay", ic="steps")
    if "step" not in st.session_state:
        st.session_state.step = 0
    top1, top2, top3 = st.columns([1.2, 1.2, 1])
    frames_all = sorted(f for f in os.listdir(os.path.join(DATA, "frames")) if f.endswith(".jpg"))
    fsel = top1.selectbox("Traffic frame", frames_all, index=min(2, len(frames_all) - 1), key="sbs_frame")
    lam_s = top2.select_slider("Quality level", options=LAMS, value=0.045, key="sbs_lam", format_func=lambda x: LAM_LABEL[x])
    exp_s = top3.radio("Codec", ["P4-V3", "P4-I3"], key="sbs_exp",
                       format_func=lambda e: "Temporal (video)" if e == "P4-V3" else "Intra (single image)")
    rv = row(exp_s, lam_s)
    frame = Image.open(os.path.join(DATA, "frames", fsel)).convert("RGB").resize((1080, 880))
    W, H = frame.size
    GW, GH = W // TOK, H // TOK
    keep = lattice_mask(GH, GW, 0.45)
    raw_bytes = W * H * 3
    file_bits = rv.bpp * W * H
    s = st.session_state.step
    st.markdown("".join(f"<span class='step-chip {'on' if i == s else ('done' if i < s else '')}'>{t}</span>"
                        for i, t in enumerate(STEPS)), unsafe_allow_html=True)
    try:
        st.progress((s + 1) / len(STEPS))
    except Exception:
        pass
    left, right = st.columns([1.15, 1])
    if s == 0:
        with left:
            show_image(frame, caption="Original camera frame (1080 × 880 crop of a traffic video)")
        with right:
            st.markdown("### Before: the raw frame")
            st.write("A traffic camera sends every pixel of every frame. A traffic AI only needs the cars, lanes and movement.")
            st.metric("Resolution", f"{W} × {H}")
            st.metric("Raw size of this frame", f"{raw_bytes/1e6:.2f} MB", "24 bits per pixel", delta_color="off")
            st.metric("Raw video at 30 fps", f"{raw_bytes*8*30/1e6:,.0f} Mbps", "one camera", delta_color="off")
    elif s == 1:
        with left:
            show_image(draw_grid(frame), caption="Every blue square is one token = a 16 × 16 pixel patch")
        with right:
            st.markdown("### Step 2: cut the frame into tokens")
            st.write("MAEO works on a grid of small patches. Both streams use the same grid at 1/16 of the resolution.")
            st.metric("Tokens in this frame", f"{GW} × {GH} = {GW*GH:,}")
    elif s == 2:
        with left:
            show_image(draw_grid(frame, keep=keep), caption="Bright = semantic tokens kept (45%) · dark = predicted by the decoder")
        with right:
            st.markdown("### Step 3: the meaning stream")
            st.write("A frozen **DINO** vision model gives 64 numbers of *meaning* per token. MAEO keeps **45%** of them on a "
                     "fixed pattern the decoder can rebuild, so the mask is **free**. A Wyner–Ziv prior predicts them from the pixel stream.")
            a, b = st.columns(2)
            a.metric("Semantic tokens kept", f"{int(keep.sum()):,} of {GW*GH:,}")
            b.metric("Mask bits sent", "0", "rebuilt at decoder")
            st.metric("Meaning stream for this frame", f"{rv.sem_bpp*W*H/8/1e3:.1f} KB", f"{rv.sem_bpp:.3f} bits/pixel (recorded)",
                      delta_color="off")
    elif s == 3:
        small = frame.resize((GW, GH), Image.BILINEAR).resize((W, H), Image.NEAREST)
        with left:
            show_image(small, caption="Illustration only: the 1/16 grid the pixel stream works on (not decoded output)")
        with right:
            st.markdown("### Step 4: the pixel stream")
            st.write("A learned neural encoder turns the frame into **128 numbers per token** for textures and colours. "
                     "For video, P-frames send only **motion + what changed**.")
            a, b = st.columns(2)
            a.metric("Pixel stream", f"{rv.rec_bpp*W*H/8/1e3:.1f} KB", f"{rv.rec_bpp:.3f} bits/pixel", delta_color="off")
            if exp_s == "P4-V3":
                b.metric("Motion", f"{rv.motion_bpp*W*H/8/1e3:.1f} KB", f"{rv.motion_bpp:.3f} bits/pixel", delta_color="off")
            else:
                b.metric("Motion", "none", "intra codec", delta_color="off")
    elif s == 4:
        with left:
            st.pyplot(size_bars(raw_bytes, file_bits / 8))
        with right:
            st.markdown("### Step 5: everything goes into one file")
            st.metric("Before", f"{raw_bytes/1e6:.2f} MB", "raw frame", delta_color="off")
            st.metric("After", f"{file_bits/8/1e3:.0f} KB", f"{raw_bytes/(file_bits/8):.0f}× smaller than raw", delta_color="off")
            st.caption(f"From the recorded {rv.bpp:.3f} bits per pixel at {LAM_LABEL[lam_s]}.")
    else:
        with left:
            st.markdown("#### Before vs after")
            show_table(pd.DataFrame({
                "": ["Size of this frame", "Bits per pixel", "Image quality (PSNR)", "Structure (MS-SSIM)",
                     "AI segmentation agrees with original", "Meaning kept (DINO similarity)"],
                "Before (raw)": [f"{raw_bytes/1e6:.2f} MB", "24", "reference", "1.000", "100%", "1.000"],
                "After (MAEO)": [f"{file_bits/8/1e3:.0f} KB", f"{rv.bpp:.3f}", f"{rv.psnr_rgb:.1f} dB", f"{rv.ms_ssim:.3f}",
                                 f"{rv.seg_acc*100:.1f}%", f"{rv.dino_cos:.3f}"]}))
            st.caption("After values are recorded averages on UVG 1080p at this quality level; this frame was not encoded live.")
        with right:
            st.markdown("### Step 6: decode from the file alone")
            a, b = st.columns(2)
            a.metric("Closure test", "0.0 difference", "file decode = direct decode", delta_color="off")
            b.metric("Segmentation agreement", f"{rv.seg_acc*100:.1f}%", "the AI still sees the scene", delta_color="off")
            st.success(f"Before {raw_bytes/1e6:.2f} MB → after {file_bits/8/1e3:.0f} KB ({raw_bytes/(file_bits/8):.0f}× smaller).")
    b1, b2, _ = st.columns([1, 1, 4])
    if b1.button("◀ Back", disabled=s == 0, key="sbs_back"):
        st.session_state.step = max(0, s - 1); rerun()
    if b2.button("Next ▶" if s < len(STEPS) - 1 else "Start again ↺", key="sbs_next", type="primary"):
        st.session_state.step = (s + 1) % len(STEPS); rerun()


# ======================================================================= page: research results
def page_results():
    hero("RESEARCH · UVG 1080p, 7 SEQUENCES × 96 FRAMES", "Research results",
         "Real recorded results from our Kaggle T4 GPU runs. Every frame was decoded from a real file.", "replay", ic="results")
    c1, c2 = st.columns([1, 1])
    with c1:
        exp = st.radio("Codec", ["P4-V3", "P4-I3"], format_func=lambda e: CODEC_NAME[e], key="r_exp")
        lam = st.select_slider("Quality level", options=LAMS, value=0.045, format_func=lambda x: LAM_LABEL[x], key="r_lam")
        rv = row(exp, lam)
        m1, m2, m3 = st.columns(3)
        m1.metric("Bits per pixel", f"{rv.bpp:.3f}")
        m2.metric("PSNR", f"{rv.psnr_rgb:.2f} dB")
        m3.metric("MS-SSIM", f"{rv.ms_ssim:.4f}")
        m1.metric("Seg. agreement", f"{rv.seg_acc*100:.1f}%")
        m2.metric("DINO similarity", f"{rv.dino_cos:.3f}")
        m3.metric("Mask bits", f"{rv.mask_bpp:.3f} bpp")
    with c2:
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        for e, col, mk in (("P4-I3", "#4F7FD6", "o"), ("P4-V3", BLUE, "s")):
            d = MEAN[MEAN.exp == e].sort_values("bpp")
            ax.plot(d.bpp, d.psnr_rgb, marker=mk, color=col, lw=2.5, label=f"MAEO {CODEC_NAME[e].split(' (')[0].lower()}")
        x = R["x265"]
        ax.plot(x["ai_bpp"], x["ai_psnr"], "--", marker="^", color=GREY, label="x265 all-intra (anchor)")
        ax.plot(x["ld_bpp"], x["ld_psnr"], ":", marker="v", color="#5B6675", label="x265 low-delay (anchor)")
        ax.scatter([rv.bpp], [rv.psnr_rgb], s=180, facecolors="none", edgecolors=ORANGE, lw=2.5, zorder=5, label="selected")
        ax.set_xlabel("bits per pixel (from real file sizes)"); ax.set_ylabel("PSNR-RGB (dB)")
        ax.grid(alpha=0.3); ax.legend(fontsize=8.5); ax.set_title("Rate–distortion on UVG")
        fig.tight_layout(); st.pyplot(fig)
    st.markdown("**Per sequence**")
    d = SEQ[(SEQ.exp == exp) & (np.isclose(SEQ.lam, lam))][["seq", "bpp", "psnr_rgb", "ms_ssim", "seg_acc", "dino_cos"]]
    show_table(d.rename(columns={"seq": "Sequence", "psnr_rgb": "PSNR (dB)", "ms_ssim": "MS-SSIM",
                                 "seg_acc": "Seg. agreement", "dino_cos": "DINO cos"}).round(4))
    rep = R["report"]
    st.markdown("**Phase 2 gate and Phase 3 (report, paired BD-rate, negative = fewer bits)**")
    show_table(pd.DataFrame({"Seed": [0, 1, 2], "Uniform mask (0 bits)": rep["p2_bd_uniform"],
                             "Random mask (0 bits)": rep["p2_bd_random"], "P2-E017 (best intra)": rep["p2_bd_e017"],
                             "Temporal vs all-intra (P3)": rep["p3_bd_vs_intra"]}))
    st.caption(f"Speed on one T4 GPU at 1080p: about {R['fps']['video_fp32']['enc_fps']:.1f} fps encode and "
               f"{R['fps']['video_fp32']['dec_fps']:.1f} fps decode (research prototype, not optimised). "
               "x265 is still ahead on pixel quality per bit.")


# ======================================================================= router
{"Overview": page_overview, "Secure Video Vault": page_vault, "Compressed Storage": page_storage,
 "Traffic Analyzer": page_traffic,
 "Fleet Planner": page_fleet, "How MAEO compresses": page_steps, "Research results": page_results}[page]()
