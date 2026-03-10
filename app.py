import os
import time
import copy
import glob
import numpy as np
import torch
import gradio as gr
from PIL import Image
from torchvision.utils import save_image
from diffusers import DDIMScheduler, AutoencoderKL
import shutil

tmp_folder = '/tmp/gradio'

# ---- Gradio/Client JSON-Schema Patch ----
try:
    import gradio_client.utils as _gu
    _orig__json = _gu._json_schema_to_python_type
    def _safe__json_schema_to_python_type(schema, defs=None):
        if isinstance(schema, bool):
            return "object"
        return _orig__json(schema, defs)
    _gu._json_schema_to_python_type = _safe__json_schema_to_python_type
except Exception as _e:
    pass

import utils
from pipeline import StyleGallery

# ---------------------------
# Global Config
# ---------------------------
MODEL_NAME = "pretrained_models/runwayml_stable-diffusion-v1-5"
VAE_PATH = ""
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------
# Model Loader
# ---------------------------
_pipe = None
def _load_models():
    global _pipe
    if _pipe is not None:
        return _pipe
    scheduler = DDIMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")
    pipe = StyleGallery.from_pretrained(MODEL_NAME, scheduler=scheduler, safety_checker=None).to(DEVICE)
    pipe.extractor = pipe.unet
    if VAE_PATH:
        vae = AutoencoderKL.from_pretrained(VAE_PATH)
        pipe.vae = vae
    _pipe = pipe
    return _pipe

# ---------------------------
# Helper Functions
# ---------------------------
def _to_paths(files):
    if not files: return []
    paths = []
    for f in files:
        if isinstance(f, str): paths.append(f); continue
        if isinstance(f, dict) and "path" in f: paths.append(f["path"]); continue
        p = getattr(f, "name", None) or getattr(f, "path", None)
        if p: paths.append(p)
    return paths

def _derive_mask_path(content_path: str):
    try:
        base_dir = os.path.dirname(content_path)
        stem = os.path.splitext(os.path.basename(content_path))[0]
        mask_path = os.path.join(base_dir, f"{stem}_json", "label.png")
        return mask_path if os.path.exists(mask_path) else None
    except Exception: return None

def _mask_signature(mask_path: str | None) -> str:
    if mask_path and os.path.exists(mask_path):
        try: return f"mask={os.path.abspath(mask_path)}@{int(os.path.getmtime(mask_path))}"
        except Exception: return f"mask={os.path.abspath(mask_path)}"
    return "mask=None"

def _make_key(content_path, style_paths, steps, mask_sig: str):
    styles_sig = ";".join(sorted(style_paths))
    return f"{content_path}|{styles_sig}|steps={int(steps)}|{mask_sig}"

def _open_image_512(path):
    img = Image.open(path).convert("RGB")
    return img.resize((512, 512), Image.BICUBIC)

def _mask64_to_overlay_rgba(mask64: np.ndarray, size=512, alpha=0.70):
    up = Image.fromarray(mask64.astype(np.int32)).resize((size, size), Image.NEAREST)
    up_np = np.array(up)
    rng = np.random.default_rng(12345)
    max_id = int(up_np.max())
    palette = np.vstack([[0, 0, 0]] + rng.integers(30, 255, size=(max_id, 3)).tolist()).astype(np.uint8)
    color = palette[up_np]
    overlay = Image.fromarray(color, mode="RGB")
    a = np.where(up_np > 0, int(alpha * 255), 0).astype(np.uint8)
    edges = (
        (up_np != np.roll(up_np, 1, axis=0)) | (up_np != np.roll(up_np, -1, axis=0)) |
        (up_np != np.roll(up_np, 1, axis=1)) | (up_np != np.roll(up_np, -1, axis=1))
    ) & (up_np > 0)
    edge_alpha, edge_color = 255, np.array([255, 255, 255], dtype=np.uint8)
    ov_np = np.array(overlay)
    ov_np[edges] = edge_color
    a[edges] = edge_alpha
    overlay = Image.fromarray(ov_np, mode="RGB")
    overlay.putalpha(Image.fromarray(a))
    return overlay

def _composite_with_mask(image_path, mask64):
    base = _open_image_512(image_path).convert("RGBA")
    ov = _mask64_to_overlay_rgba(mask64)
    out = Image.alpha_composite(base, ov)
    return out.convert("RGB")

def _split_clusters_images(image_path, mask64: np.ndarray, size=512):
    base_rgb = _open_image_512(image_path).convert("RGB")
    base = base_rgb.convert("RGBA")
    up = Image.fromarray(mask64.astype(np.int32)).resize((size, size), Image.NEAREST)
    m = np.array(up)
    ids = [int(x) for x in np.unique(m).tolist() if int(x) != 0]
    out_list = []
    if not ids: return [base_rgb]
    dark_alpha, tint_alpha = 160, 170
    rng = np.random.default_rng(12345)
    max_id = int(m.max())
    palette = np.vstack([[0, 0, 0]] + rng.integers(30, 255, size=(max_id, 3)).tolist()).astype(np.uint8)
    for cid in ids:
        ov = np.zeros((size, size, 4), dtype=np.uint8)
        outside = (m != cid); inside = (m == cid)
        ov[outside, :3] = 0; ov[outside, 3] = dark_alpha
        ov[inside, :3] = palette[cid]; ov[inside, 3] = tint_alpha
        comp = Image.alpha_composite(base, Image.fromarray(ov, mode="RGBA")).convert("RGB")
        out_list.append(comp)
    return out_list

def _pbar_md(frac: float, text: str) -> str:
    frac = float(frac)
    pct = max(0, min(100, int(round(frac * 100))))
    return f"""
<div class="sg-progress pr-progress-container">
  <div class="pr-progress-bar" style="width:{pct}%"></div>
</div>
<div class="pr-progress-text">{text} ({pct}%)</div>
""".strip()

MASK_CACHE = {}
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}

def _resolve_mask_path(content_tmp_path: str, content_root: str | None, mask_root: str | None):
    if not content_tmp_path: return None
    stem = os.path.splitext(os.path.basename(content_tmp_path))[0]
    key = (content_root or "", mask_root or "", stem)
    if key in MASK_CACHE: return MASK_CACHE[key]

    p = _derive_mask_path(content_tmp_path)
    if p: MASK_CACHE[key] = p; return p

    if content_root and os.path.isdir(content_root):
        cand = [c for c in glob.glob(os.path.join(content_root, "**", f"{stem}.*"), recursive=True)
                if os.path.splitext(c)[1].lower() in IMG_EXTS]
        if cand:
            cand.sort(key=len)
            p = _derive_mask_path(cand[0])
            if p: MASK_CACHE[key] = p; return p

    if mask_root and os.path.isdir(mask_root):
        cand = glob.glob(os.path.join(mask_root, "**", f"{stem}_json", "label.png"), recursive=True)
        if cand:
            cand.sort(key=len)
            MASK_CACHE[key] = cand[0]; return cand[0]

    MASK_CACHE[key] = None; return None

def clean_tmp_files(keep_minutes=30):
    logs = []
    for d in ["uploads", "outputs", "outputs/depth_visual"]:
        if os.path.isdir(d):
            for name in os.listdir(d):
                path = os.path.join(d, name)
                try: shutil.rmtree(path, ignore_errors=True) if os.path.isdir(path) else os.remove(path)
                except Exception as e: logs.append(f"[skip] {path}: {e}")
        else: os.makedirs(d, exist_ok=True)
    import tempfile
    tmp = tempfile.gettempdir()
    now = time.time()
    for name in os.listdir(tmp):
        if name.lower().startswith("gradio"):
            p = os.path.join(tmp, name)
            try:
                if now - os.path.getmtime(p) > keep_minutes * 60: shutil.rmtree(p, ignore_errors=True)
            except Exception: pass
    return _pbar_md(1.0, "Cache cleared successfully!")

# ---------------------------
# Background Preprocessing
# ---------------------------
def precompute(content_file, style_files, noise_steps, use_sam, use_depth, state: dict, content_root, mask_root):
    content_paths = _to_paths([content_file]) if content_file else []
    content_tmp = content_paths[0] if content_paths else None
    style_paths = _to_paths(style_files)
    if not content_tmp or not style_paths:
        return state, "Waiting for content and style images..."

    mask_path = _resolve_mask_path(content_tmp, content_root, mask_root)
    key = _make_key(content_tmp, style_paths, noise_steps, _mask_signature(mask_path)) + f"|sam={use_sam}|depth={use_depth}"
    if state and state.get("key") == key:
        return state, "Cache hit: Preprocessing already done ✓"

    pipe = _load_models()
    print(f"[mask] precompute using mask_path => {mask_path}")

    t0 = time.time()
    cluster_matches, content_dict, style_dict = pipe.cluster_match(
        content_tmp, style_paths, steps=int(noise_steps), content_original_mask=mask_path, 
        use_depth=use_depth, use_sam=use_sam
    )
    dt = time.time() - t0

    content_mask64 = content_dict["mask"].detach().cpu().numpy().astype(np.int32)
    content_overlay = _composite_with_mask(content_tmp, content_mask64)

    style_overlays = []
    for sd in style_dict:
        s_mask = sd["mask"].detach().cpu().numpy().astype(np.int32)
        style_overlays.append(_composite_with_mask(sd["image"], s_mask))

    new_state = {
        "key": key, "content_path": content_tmp, "style_paths": style_paths, "steps": int(noise_steps),
        "cluster_matches": cluster_matches, "content_dict": content_dict, "style_dict": style_dict,
        "content_mask_path": mask_path, "content_overlay": content_overlay, "style_overlays": style_overlays,
    }
    return new_state, f"Preprocessing Done ✓ | {len(style_paths)} styles processed"

def precompute_styles_with_progress(content_file, style_files_state, noise_steps, use_sam, use_depth, state: dict, content_root, mask_root):
    style_paths = _to_paths(style_files_state)
    if len(style_paths) == 0:
        msg = _pbar_md(0.00, "Please drop at least one style...")
        yield state, [], msg
        return
    msg1 = _pbar_md(0.10, f"Extracting features for {len(style_paths)} styles...")
    yield state, [], msg1
    new_state, status = precompute(content_file, style_files_state, noise_steps, use_sam, use_depth, state, content_root, mask_root)
    msg2 = _pbar_md(1.00, status)
    yield new_state, style_paths, msg2

def precompute_state_only_with_progress_gen(content_file, style_files_state, noise_steps, use_sam, use_depth, state: dict, content_root, mask_root):
    yield state, _pbar_md(0.10, "Updating cache...")
    new_state, status = precompute(content_file, style_files_state, noise_steps, use_sam, use_depth, state, content_root, mask_root)
    yield new_state, _pbar_md(1.00, status)

# ---------------------------
# Custom Mapping Interactions
# ---------------------------
def pick_content_cluster(evt: gr.SelectData, state):
    if not state: return gr.update(), "**Hint**: Not preprocessed yet"
    x, y = int(evt.index[0]), int(evt.index[1])
    mask = state["content_dict"]["mask"].detach().cpu().numpy().astype(np.int32)
    c, r = np.clip(int(y/512*mask.shape[0]), 0, mask.shape[0]-1), np.clip(int(x/512*mask.shape[1]), 0, mask.shape[1]-1)
    cid = int(mask[c, r])
    return cid, f"**Hint**: Selected Content ID = {cid}"

def pick_style_cluster(evt: gr.SelectData, state, style_idx):
    if not state: return gr.update(), "**Hint**: Not preprocessed yet"
    x, y = int(evt.index[0]), int(evt.index[1])
    mask = state["style_dict"][style_idx]["mask"].detach().cpu().numpy().astype(np.int32)
    c, r = np.clip(int(y/512*mask.shape[0]), 0, mask.shape[0]-1), np.clip(int(x/512*mask.shape[1]), 0, mask.shape[1]-1)
    sid = int(mask[c, r])
    return sid, f"**Hint**: Selected Style ID = {sid}"

def add_mapping(map_state: dict, content_cluster, style_idx, style_cluster):
    if map_state is None: map_state = {}
    pairs = map_state.get("pairs", {})
    if content_cluster is None or style_idx is None or style_cluster is None:
        return map_state, gr.update()
    pairs[int(content_cluster)] = {"style_dict_index": int(style_idx), "style_cluster": int(style_cluster)}
    map_state["pairs"] = pairs
    rows = [[k, v["style_dict_index"], v["style_cluster"]] for k, v in sorted(pairs.items())]
    return map_state, gr.update(value=rows)

def clear_mapping(map_state: dict):
    return {"pairs": {}}, gr.update(value=[])

# ---------------------------
# Core Rendering
# ---------------------------
def run_style_transfer(
    content_file, style_files_state, noise_steps, use_sam, use_depth, start_layer, end_layer, 
    num_optimize_steps, iters, c_ratio, seed,
    state, mode, map_state, content_root, mask_root
):
    content_paths = _to_paths([content_file]) if content_file else []
    content_tmp = content_paths[0] if content_paths else None
    style_paths = _to_paths(style_files_state)

    if not content_tmp or not style_paths:
        raise gr.Error("Error: Please provide a content image and at least one style image.")

    content_mask_path = _resolve_mask_path(content_tmp, content_root, mask_root)
    pipe = _load_models()
    
    controller = utils.Controller(self_layers=(int(start_layer), int(end_layer)))
    torch.manual_seed(int(seed))
    start_time = time.time()

    key = _make_key(content_tmp, style_paths, noise_steps, _mask_signature(content_mask_path)) + f"|sam={use_sam}|depth={use_depth}"
    use_cache = bool(state and state.get("key") == key)
    
    if use_cache:
        cluster_matches, content_dict, style_dict = dict(state["cluster_matches"]), state["content_dict"], state["style_dict"]
        print("[cache] Loading precomputed results")
    else:
        print("[cache] Miss, running cluster_match ...")
        cluster_matches, content_dict, style_dict = pipe.cluster_match(
            content_tmp, style_paths, steps=int(noise_steps), content_original_mask=content_mask_path, 
            use_depth=use_depth, use_sam=use_sam
        )

    auto_matches = copy.deepcopy(cluster_matches)
    if mode == "Custom":
        if map_state and map_state.get("pairs"):
            for c_id, v in map_state["pairs"].items():
                auto_matches[int(c_id)] = {
                    "style_dict_index": int(v["style_dict_index"]),
                    "style_cluster": int(v["style_cluster"]), "similarity": 0.0,
                }
        cluster_matches = auto_matches

    result_tensor = pipe.style_transfer(
        content_dict=content_dict,
        style_dict=style_dict,
        controller=controller,
        cluster_matches=cluster_matches,
        num_optimize_steps=int(num_optimize_steps),
        iters=int(iters),
        c_ratio=float(c_ratio)
    )

    elapsed = time.time() - start_time
    print(f"style_transfer time: {elapsed:.2f}s")
    out_path = os.path.join(OUTPUT_DIR, f"result_{int(time.time())}.png")
    save_image(result_tensor, out_path)
    return Image.open(out_path)

# ==============================================================================
# Professional UI & Layout (English, Fixed Columns, High Contrast Hints)
# ==============================================================================
PR_CSS = """
/* Premiere Pro Dark Theme Base */
body, .gradio-container { background-color: #1e1e1e !important; color: #cccccc !important; font-family: 'Segoe UI', system-ui, sans-serif !important; margin: 0; padding: 0; }
.gradio-container { max-width: 100% !important; }

/* Remove Default Shadows */
.gr-box, .gr-form, .gr-panel { box-shadow: none !important; border: none !important; background: transparent !important; }

/* Left Main Column: STRICTLY stack Program Monitor above Source Monitor */
.main-column {
    display: flex !important;
    flex-direction: column !important;
    gap: 0px !important;
}

/* Right Sidebar: STRICTLY stack panels vertically */
.sidebar-column {
    display: flex !important;
    flex-direction: column !important;
    gap: 12px !important;
    border-left: 1px solid #3e3e42;
    padding-left: 15px !important;
}
.sidebar-column > * {
    width: 100% !important;
    max-width: 100% !important;
    flex-grow: 0 !important;
}

/* Panel Styles */
.pr-panel {
    background-color: #252526 !important;
    border: 1px solid #3e3e42 !important;
    border-radius: 0px !important;
    margin-bottom: 8px !important;
    overflow: hidden;
}
.pr-panel-header {
    background-color: #333333 !important;
    color: #cccccc !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
    padding: 6px 12px !important;
    border-bottom: 1px solid #3e3e42 !important;
    user-select: none;
}
.pr-panel-content { padding: 12px !important; }

/* Inputs & Forms */
input, textarea, .gr-input, .gr-dropdown, select {
    background-color: #1e1e1e !important;
    border: 1px solid #3e3e42 !important;
    border-radius: 0px !important;
    color: #cccccc !important;
    font-size: 13px !important;
}
input:focus, .gr-input:focus { border-color: #0d66d0 !important; outline: none !important; }

/* Components */
.gr-image, .gr-gallery, .gr-file, .gr-dataframe { background-color: #1e1e1e !important; border: 1px solid #3e3e42 !important; border-radius: 0px !important; }
.gr-dataframe th { background-color: #333333 !important; color: #cccccc !important; border-color: #3e3e42 !important;}
.gr-dataframe td { background-color: #1e1e1e !important; color: #cccccc !important; border-color: #3e3e42 !important;}

/* Typography */
h1, h2, h3, h4, h5, h6, p, span, label, .gr-text { color: #cccccc !important; font-family: 'Segoe UI', system-ui, sans-serif !important; }
.gr-text { font-size: 12px !important; }

/* Instruction Hint Text (Black text on Light background) */
.hint-text {
    background-color: #e2e8f0 !important;
    border-left: 4px solid #0d66d0 !important;
    border-radius: 4px !important;
    padding: 6px 10px !important;
    margin-top: 4px !important;
    margin-bottom: 4px !important;
}
.hint-text p, .hint-text span, .hint-text strong {
    color: #000000 !important; 
    font-size: 13px !important;
    margin: 0 !important;
}

/* Buttons */
button { font-weight: 500 !important; border-radius: 0px !important; transition: all 0.1s !important; padding: 6px 16px !important; font-size: 13px !important;}
.btn-primary { background-color: #0d66d0 !important; color: #ffffff !important; border: 1px solid #0d66d0 !important; }
.btn-primary:hover { background-color: #1177bb !important; }
.btn-secondary { background-color: #333333 !important; color: #cccccc !important; border: 1px solid #3e3e42 !important; }
.btn-secondary:hover { background-color: #3e3e42 !important; }
.btn-danger { background-color: #8c1d1d !important; color: white !important; border: 1px solid #8c1d1d !important; }

/* Progress Bar */
.pr-progress-container { width: 100%; height: 6px; background: #3e3e42; border-radius: 0px; overflow: hidden; margin-top: 4px; }
.pr-progress-bar { height: 100%; background: #0d66d0; transition: width 0.3s ease; }
.pr-progress-text { font-size: 11px; color: #999999; margin-top: 4px; }

/* Layout Grid Constraints */
.pr-main { padding-right: 16px; height: 100vh; overflow-y: auto; }
"""

with gr.Blocks(title="StyleGallery Pro", css=PR_CSS, theme=gr.themes.Base()) as demo:
    
    gr.HTML("""
    <script>
        document.body.classList.add('dark');
        document.querySelectorAll('.gr-file p, .gr-files p').forEach(el => {
            if (el.textContent.includes('将文件拖拽到此处')) el.textContent = 'Drop media here';
        });
    </script>
    """)

    style_images_state = gr.State([])

    with gr.Row():
        
        # ==========================================
        # LEFT: Main Workspace (Monitor Fixed Stack)
        # ==========================================
        # Added 'main-column' class to force vertical layout strictly
        with gr.Column(scale=7, elem_classes=["pr-main", "main-column"]):
            
            gr.Markdown("### 🎬 StyleGallery Studio")
            
            # --- TOP: Program Monitor ---
            with gr.Group(elem_classes=["pr-panel"]):
                gr.Markdown('<div class="pr-panel-header">Program Monitor</div>')
                with gr.Column(elem_classes=["pr-panel-content"]):
                    
                    # === AUTO MODE ===
                    with gr.Group(visible=True) as auto_group:
                        with gr.Row():
                            style_gallery = gr.Gallery(label="Analyzed Style Clusters", height=450, columns=5, object_fit="contain")
                            output_auto = gr.Image(label="Program Output", format="png", height=450)

                    # === CUSTOM MODE ===
                    with gr.Group(visible=False) as custom_group:
                        with gr.Row():
                            with gr.Column(scale=1):
                                content_overlay_img = gr.Image(label="Source Monitor (Pick Content)", interactive=True, height=320)
                                picked_content_cluster = gr.Number(label="Active Content ID", precision=0)
                                content_click_info = gr.Markdown("**Hint**: Click on the source monitor to pick a content cluster.", elem_classes=["hint-text"])
                            with gr.Column(scale=1):
                                style_overlay_img = gr.Image(label="Reference Monitor (Pick Style)", interactive=True, height=320)
                                picked_style_cluster = gr.Number(label="Active Style ID", precision=0)
                                style_click_info = gr.Markdown("**Hint**: Click on the reference monitor to pick a style cluster.", elem_classes=["hint-text"])
                        
                        # Mapping Toolbar (Buttons changed to size="sm")
                        with gr.Row(equal_height=True):
                            style_choice = gr.Dropdown(label="Active Reference Track", choices=[], value=None, scale=2)
                            add_pair_btn = gr.Button("🔗 Link Nodes", elem_classes=["btn-primary"], size="sm", scale=1)
                            clear_pair_btn = gr.Button("🗑 Unlink All", elem_classes=["btn-danger"], size="sm", scale=1)
                        
                        with gr.Row():
                            style_clusters_gallery = gr.Gallery(label="Track Visualization", height=200, columns=6, object_fit="contain", scale=2)
                            with gr.Column(scale=1):
                                map_table = gr.Dataframe(headers=["Content ID", "Style Img Idx", "Style ID"], value=[], label="Routing Table", interactive=False)
                                map_state = gr.State({"pairs": {}})
                        
                        output_custom = gr.Image(label="Program Output (Custom Mode)", format="png", height=450)

            # --- BOTTOM: Global Source Monitor ---
            with gr.Group(elem_classes=["pr-panel"]):
                gr.Markdown('<div class="pr-panel-header">Global Source Monitor</div>')
                with gr.Column(elem_classes=["pr-panel-content"]):
                    with gr.Row():
                        with gr.Column(scale=1):
                            content_preview = gr.Image(label="Source Content Preview", interactive=False, height=350)
                        with gr.Column(scale=2):
                            style_files_gallery_raw = gr.Gallery(label="Uploaded Style References", height=350, columns=4, object_fit="contain", interactive=False)

        # ==========================================
        # RIGHT: Sidebar (Forced Vertical Stack)
        # ==========================================
        with gr.Column(scale=3, min_width=320, elem_classes=["sidebar-column"]):
            
            # 1. Parameters
            with gr.Group(elem_classes=["pr-panel"]):
                gr.Markdown('<div class="pr-panel-header">1. Effect Controls</div>')
                with gr.Column(elem_classes=["pr-panel-content"]):
                    mode_radio = gr.Radio(["Auto", "Custom"], value="Auto", label="Mode")
                    num_optimize_steps_slider = gr.Slider(minimum=10, maximum=300, step=1, value=150, label="Optimize Steps")
                    iters_slider = gr.Slider(minimum=1, maximum=10, step=1, value=1, label="Iterations")
                    c_ratio_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.24, label="Content Ratio")
                    noise_steps_slider = gr.Slider(minimum=0, maximum=50, step=1, value=15, label="Noise Floor")
                    start_layer_slider = gr.Slider(minimum=0, maximum=30, step=1, value=10, label="Start Layer")
                    end_layer_slider = gr.Slider(minimum=0, maximum=30, step=1, value=16, label="End Layer")
                    seed_input = gr.Number(label="Random Seed", value=42, precision=0)
                    with gr.Row():
                        use_depth_checkbox = gr.Checkbox(label="Enable Depth Pass", value=True)
                        use_sam_checkbox = gr.Checkbox(label="Enable SAM Pass", value=False)
            
            # 2. Uploads
            with gr.Group(elem_classes=["pr-panel"]):
                gr.Markdown('<div class="pr-panel-header">2. Project Media Bin</div>')
                with gr.Column(elem_classes=["pr-panel-content"]):
                    content_file = gr.File(label="Source Content Track", file_types=["image"], type="filepath", height=100)
                    
                    gr.Markdown("---")
                    style_dropzone = gr.File(label="📥 Drop Style Images Here", file_count="multiple", file_types=["image"], height=100)
                    style_list_preview = gr.Gallery(label="Style Queue", columns=3, height=120, object_fit="contain", interactive=False)
                    clear_style_btn = gr.Button("🗑️ Clear Style Queue", size="sm", elem_classes=["btn-secondary"])

                    with gr.Accordion("Advanced Local Paths", open=False):
                        content_root = gr.Textbox(label="Content Root Dir", placeholder="/data/dataset/content")
                        mask_root = gr.Textbox(label="Mask Root Dir", placeholder="/data/dataset/masks")

            # 3. Execution
            with gr.Group(elem_classes=["pr-panel"]):
                gr.Markdown('<div class="pr-panel-header">3. Execution & Status</div>')
                with gr.Column(elem_classes=["pr-panel-content"]):
                    run_btn_auto = gr.Button("▶ Render Sequence (Auto)", elem_classes=["btn-primary"], visible=True)
                    run_btn_custom = gr.Button("▶ Render Sequence (Custom)", elem_classes=["btn-primary"], visible=False)
                    clean_btn = gr.Button("🧹 Purge Media Cache", elem_classes=["btn-secondary"])
                    
                    global_status_md = gr.Markdown(value="**System Status**: Ready...", elem_classes=["hint-text"])

    # ==========================================================================
    # Logic Bindings
    # ==========================================================================
    pre_state = gr.State({})

    def append_styles(new_files, current_paths):
        if new_files:
            for f in new_files:
                current_paths.append(f.name)
        return current_paths, current_paths, current_paths, None

    style_dropzone.upload(
        fn=append_styles, 
        inputs=[style_dropzone, style_images_state], 
        outputs=[style_images_state, style_list_preview, style_files_gallery_raw, style_dropzone]
    )

    clear_style_btn.click(
        fn=lambda: ([], [], []), 
        inputs=[], 
        outputs=[style_images_state, style_list_preview, style_files_gallery_raw]
    )

    def _preview_content(f):
        p = _to_paths([f]); return p[0] if p else None
    content_file.change(_preview_content, inputs=content_file, outputs=content_preview)

    pre_inputs = [content_file, style_images_state, noise_steps_slider, use_sam_checkbox, use_depth_checkbox, pre_state, content_root, mask_root]

    style_images_state.change(
        precompute_styles_with_progress, inputs=pre_inputs,
        outputs=[pre_state, style_gallery, global_status_md],
    )

    for inp in [content_file, noise_steps_slider, use_sam_checkbox, use_depth_checkbox, content_root, mask_root]:
        inp.change(
            precompute_state_only_with_progress_gen, inputs=pre_inputs,
            outputs=[pre_state, global_status_md],
        )

    def _toggle_mode(mode):
        is_auto = (mode == "Auto")
        return gr.update(visible=is_auto), gr.update(visible=not is_auto), gr.update(visible=is_auto), gr.update(visible=not is_auto)
    mode_radio.change(_toggle_mode, inputs=mode_radio, outputs=[auto_group, custom_group, run_btn_auto, run_btn_custom])

    def _refresh_overlays(state):
        if not state: return None, gr.update(choices=[], value=None), None
        c_img = state.get("content_overlay", None)
        names = [f"{i}: {os.path.basename(p)}" for i, p in enumerate(state["style_paths"])]
        return c_img, gr.update(choices=names, value=names[0] if names else None), None
    pre_state.change(_refresh_overlays, inputs=pre_state, outputs=[content_overlay_img, style_choice, style_overlay_img])

    def _show_style_overlay_and_clusters(choice, state):
        if not state or not choice: return None, []
        idx = int(str(choice).split(":", 1)[0])
        overlay = state["style_overlays"][idx]
        mask64 = state["style_dict"][idx]["mask"].detach().cpu().numpy().astype(np.int32)
        path = state["style_paths"][idx]
        return overlay, _split_clusters_images(path, mask64)
    style_choice.change(_show_style_overlay_and_clusters, inputs=[style_choice, pre_state], outputs=[style_overlay_img, style_clusters_gallery])

    content_overlay_img.select(pick_content_cluster, inputs=[pre_state], outputs=[picked_content_cluster, content_click_info])
    
    def _style_click(evt: gr.SelectData, state, choice):
        if not choice: return gr.update(), "**Error**: Please select a style track from the dropdown first."
        return pick_style_cluster(evt, state, int(str(choice).split(":", 1)[0]))
    style_overlay_img.select(_style_click, inputs=[pre_state, style_choice], outputs=[picked_style_cluster, style_click_info])

    def _add_map(map_state, c, choice, s):
        if not choice: return map_state, gr.update()
        return add_mapping(map_state, c, int(str(choice).split(":", 1)[0]), s)
    add_pair_btn.click(_add_map, inputs=[map_state, picked_content_cluster, style_choice, picked_style_cluster], outputs=[map_state, map_table])
    clear_pair_btn.click(clear_mapping, inputs=[map_state], outputs=[map_state, map_table])

    clean_btn.click(fn=clean_tmp_files, outputs=global_status_md)

    run_inputs = [
        content_file, style_images_state, noise_steps_slider, use_sam_checkbox, use_depth_checkbox, 
        start_layer_slider, end_layer_slider, num_optimize_steps_slider, iters_slider, c_ratio_slider, seed_input,
        pre_state, mode_radio
    ]

    run_btn_auto.click(run_style_transfer, inputs=run_inputs + [gr.State({"pairs": {}}), content_root, mask_root], outputs=[output_auto])
    run_btn_custom.click(run_style_transfer, inputs=run_inputs + [map_state, content_root, mask_root], outputs=[output_custom])

if __name__ == "__main__":
    demo.launch()