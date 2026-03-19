import os
import yaml
import subprocess
import threading
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
from ipywidgets import Layout, HBox, VBox

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════
CONFIG_SAVE_PATH = "config.yaml"
TRAIN_SCRIPT     = "PaddleOCR/tools/train.py"
EVAL_SCRIPT      = "PaddleOCR/tools/eval.py"
EXPORT_SCRIPT    = "PaddleOCR/tools/export_model.py"
MAX_LOG_LINES    = 1000          # FIX: cap buffer so huge outputs don't freeze UI
_UNBUFFERED_ENV  = {**os.environ, "PYTHONUNBUFFERED": "1"}
_processes       = {"train": None, "eval": None, "export": None, "convert": None}

# ═══════════════════════════════════════════════════════════════════════════════
# Inject CSS  — terminal textarea + tab polish
# ═══════════════════════════════════════════════════════════════════════════════
display(HTML("""
<style>
/* FIX: make every log textarea look and scroll like a real terminal */
.log-textarea textarea {
    font-family: 'Courier New', Courier, monospace !important;
    font-size:   12px   !important;
    line-height: 1.5    !important;
    background:  #1e1e2e !important;
    color:       #cdd6f4 !important;
    border:      1px solid #45475a !important;
    border-radius: 6px  !important;
    padding:     10px   !important;
    resize:      vertical !important;
    /* FIX: overflow is handled inside the textarea natively */
    overflow-x:  auto   !important;
    overflow-y:  scroll !important;
    white-space: pre    !important;   /* no forced wrap → horizontal scroll inside box */
    word-break:  normal !important;
}
.section-header {
    font-weight: 700; font-size: 13px; color: #555;
    border-bottom: 2px solid #e0e0e0;
    padding-bottom: 4px; margin: 10px 0 6px 0;
}
</style>
"""))

# ═══════════════════════════════════════════════════════════════════════════════
# Style Helpers
# ═══════════════════════════════════════════════════════════════════════════════
LABEL_STYLE = {"description_width": "180px"}

def make_btn(desc, icon, color, width="160px"):
    btn = widgets.Button(
        description=desc, icon=icon,
        layout=Layout(width=width, height="34px"),
    )
    btn.style.button_color = color
    btn.style.font_weight  = "bold"
    return btn

def section_header(text):
    return widgets.HTML(
        f'<div class="section-header">{text}</div>'
    )

def status_html(msg, color="#2e7d32"):
    return f'<span style="font-size:12px;color:{color};font-weight:600">{msg}</span>'

def download_link_html(zip_path: str) -> str:
    filename = os.path.basename(zip_path)
    href     = f"/kaggle/working/{zip_path}"
    return (
        f'<a href="{href}" download="{filename}" target="_blank" '
        f'style="display:inline-block;margin-top:8px;padding:6px 14px;'
        f'background:#1565c0;color:#fff;border-radius:5px;'
        f'font-weight:600;text-decoration:none;font-size:12px">'
        f'⬇  Download {filename}</a>'
    )

# ─── FIX: Log area uses Textarea instead of Output ───────────────────────────
def make_log(height="360px") -> widgets.Textarea:
    """
    Textarea-based log area.
    - Native browser scrolling: no overflow issues regardless of output size.
    - Horizontal scroll INSIDE the box: long lines stay contained.
    - Vertical scroll: always works, even with thousands of lines.
    """
    ta = widgets.Textarea(
        value="",
        layout=Layout(width="100%", height=height),
    )
    ta.add_class("log-textarea")
    return ta

def log_clear(ta: widgets.Textarea):
    ta.value = ""

def log_append(ta: widgets.Textarea, text: str):
    """
    Append text and enforce MAX_LOG_LINES to prevent UI freeze on huge outputs.
    Drops the oldest lines when the buffer is full.
    """
    new_val = ta.value + text
    lines   = new_val.splitlines(keepends=True)
    if len(lines) > MAX_LOG_LINES:
        trimmed = lines[-MAX_LOG_LINES:]
        new_val = f"[... trimmed to last {MAX_LOG_LINES} lines ...]\n" + "".join(trimmed)
    ta.value = new_val

# ═══════════════════════════════════════════════════════════════════════════════
# Shared subprocess runner
# ═══════════════════════════════════════════════════════════════════════════════
def run_command(
    cmd:          list,
    log:          widgets.Textarea,
    proc_key:     str,
    on_done_msg:  str  = "✅ Done",
    on_finish_cb       = None,
    link_widget:  widgets.HTML = None,   # FIX: for download links after zip
) -> None:
    def _stream():
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=_UNBUFFERED_ENV,
            )
            _processes[proc_key] = proc
            log_append(log, "Command: " + " ".join(cmd) + "\n" + "─" * 60 + "\n")

            for line in iter(proc.stdout.readline, ""):
                log_append(log, line)

            proc.wait()
            code = proc.returncode
            sep  = "─" * 60
            if code == 0:
                log_append(log, f"\n{sep}\n{on_done_msg}\n")
            else:
                log_append(log, f"\n{sep}\n⚠️  Process exited with code {code}\n")

        except Exception as exc:
            log_append(log, f"❌ Error: {exc}\n")
        finally:
            _processes[proc_key] = None
            if on_finish_cb:
                on_finish_cb()

    threading.Thread(target=_stream, daemon=True).start()


def stop_process(proc_key, log, status_label, run_btn, stop_btn):
    proc = _processes.get(proc_key)
    if proc and proc.poll() is None:
        proc.terminate()
        log_append(log, "\n🛑  Stopped by user.\n")
    status_label.value = status_html("🛑 Stopped", "#c62828")
    run_btn.disabled   = False
    stop_btn.disabled  = True


# ═══════════════════════════════════════════════════════════════════════════════
# ── TAB 1 · Training ──────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════
train_config_path_input = widgets.Text(
    placeholder="Enter YAML config path…",
    description="Config Path:",
    style=LABEL_STYLE, layout=Layout(flex="1 1 auto", height="32px"),
)
load_config_btn = make_btn("Load Config", "folder-open", "#E3F2FD", "160px")
save_config_btn = make_btn("Save Config", "save",        "#E8F5E9", "160px")

config_loader_row = HBox(
    [train_config_path_input, load_config_btn, save_config_btn],
    layout=Layout(width="100%", align_items="center", gap="8px"),
)

yaml_editor = widgets.Textarea(
    placeholder="YAML content will appear here after loading…",
    layout=Layout(width="100%", height="300px"),
)

hardware_dropdown = widgets.Dropdown(
    options=[("CPU", "false"), ("GPU", "true")],
    value="true", description="Hardware:",
    style=LABEL_STYLE, layout=Layout(flex="1 1 0%"),
)
distributed_dropdown = widgets.Dropdown(
    options=[("Enabled", True), ("Disabled", False)],
    value=False, description="Distributed:",
    style=LABEL_STYLE, layout=Layout(flex="1 1 0%"),
)
train_btn              = make_btn("▶  Start Training", "play", "#E8F5E9", "180px")
train_stop_btn         = make_btn("■  Stop",           "stop", "#FFEBEE", "120px")
train_stop_btn.disabled = True
train_status_label     = widgets.HTML(value="")

hw_row = HBox(
    [hardware_dropdown, distributed_dropdown,
     widgets.HTML("<div style='flex:1'></div>"),
     train_status_label, train_stop_btn, train_btn],
    layout=Layout(width="100%", align_items="center", gap="8px"),
)

train_log = make_log("380px")

training_tab = VBox(
    [section_header("📂  Config File"),
     config_loader_row,
     section_header("✏️  YAML Editor"),
     yaml_editor,
     section_header("🚀  Training"),
     hw_row,
     train_log],
    layout=Layout(width="100%", padding="12px"),
)

# ── Handlers ──────────────────────────────────────────────────────────────────
def on_load_config(_):
    log_clear(train_log)
    path = train_config_path_input.value.strip()
    if not path:
        log_append(train_log, "⚠️  Please enter a config file path.\n"); return
    if not os.path.isfile(path):
        log_append(train_log, f"❌  File not found: {path}\n"); return
    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        yaml_editor.value = yaml.dump(data, sort_keys=False, allow_unicode=True)
        log_append(train_log, f"✅  Loaded: {path}\n")
    except Exception as exc:
        log_append(train_log, f"❌  Error: {exc}\n")

def on_save_config(_):
    log_clear(train_log)
    if not yaml_editor.value.strip():
        log_append(train_log, "⚠️  YAML editor is empty.\n"); return
    try:
        data = yaml.safe_load(yaml_editor.value)
        os.makedirs(os.path.dirname(os.path.abspath(CONFIG_SAVE_PATH)), exist_ok=True)
        with open(CONFIG_SAVE_PATH, "w") as f:
            yaml.dump(data, f, sort_keys=False, allow_unicode=True)
        size = os.path.getsize(CONFIG_SAVE_PATH)
        log_append(train_log, f"✅  Saved → {CONFIG_SAVE_PATH}  ({size:,} bytes)\n")
    except Exception as exc:
        log_append(train_log, f"❌  Error: {exc}\n")

def _build_train_cmd():
    use_gpu     = hardware_dropdown.value == "true"
    distributed = distributed_dropdown.value is True
    if use_gpu and distributed:
        return ["python3", "-u", "-m", "paddle.distributed.launch",
                "--log_dir=/log/", TRAIN_SCRIPT, CONFIG_SAVE_PATH]
    return ["python3", "-u", TRAIN_SCRIPT, "-c", CONFIG_SAVE_PATH]

def on_start_training(_):
    log_clear(train_log)
    try:
        data = yaml.safe_load(yaml_editor.value) or {}
        data.setdefault("Global", {})
        data["Global"]["use_gpu"]     = hardware_dropdown.value == "true"
        data["Global"]["distributed"] = distributed_dropdown.value is True
        os.makedirs(os.path.dirname(os.path.abspath(CONFIG_SAVE_PATH)), exist_ok=True)
        with open(CONFIG_SAVE_PATH, "w") as f:
            yaml.dump(data, f, sort_keys=False, allow_unicode=True)
        yaml_editor.value = yaml.dump(data, sort_keys=False, allow_unicode=True)
        log_append(train_log, f"💾  Config saved → {CONFIG_SAVE_PATH}\n")
    except Exception as exc:
        log_append(train_log, f"❌  Failed to save config: {exc}\n"); return

    train_btn.disabled        = True
    train_stop_btn.disabled   = False
    train_status_label.value  = status_html("⏳ Training…", "#e65100")
    cmd  = _build_train_cmd()
    mode = "🌐 Distributed (GPU)" if "distributed.launch" in cmd else "💻 Single-device"
    log_append(train_log, f"🚀  Mode: {mode}\n")

    def _on_done():
        train_btn.disabled        = False
        train_stop_btn.disabled   = True
        train_status_label.value  = status_html("✅ Done", "#2e7d32")

    run_command(cmd, train_log, "train", "✅  Training finished", on_finish_cb=_on_done)

def on_stop_training(_):
    stop_process("train", train_log, train_status_label, train_btn, train_stop_btn)

load_config_btn.on_click(on_load_config)
save_config_btn.on_click(on_save_config)
train_btn.on_click(on_start_training)
train_stop_btn.on_click(on_stop_training)


# ═══════════════════════════════════════════════════════════════════════════════
# ── TAB 2 · Evaluation ────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════
eval_model_path_input = widgets.Text(
    placeholder="Path to model checkpoint…",
    description="Model Path:",
    style=LABEL_STYLE, layout=Layout(width="100%", height="32px"),
)
eval_data_dir_input = widgets.Text(
    placeholder="Dataset directory path…",
    description="Dataset Dir:",
    style=LABEL_STYLE, layout=Layout(width="100%", height="32px"),
)
eval_label_files_input = widgets.Textarea(
    placeholder="/dataset_1/test/Label.txt\n/dataset_2/test/Label.txt",
    description="Label Files:",
    style=LABEL_STYLE, layout=Layout(width="100%", height="100px"),
)
eval_btn              = make_btn("▶  Evaluate", "vials", "#E8F5E9", "160px")
eval_stop_btn         = make_btn("■  Stop",     "stop",  "#FFEBEE", "120px")
eval_stop_btn.disabled = True
eval_status_label     = widgets.HTML(value="")
eval_log              = make_log("340px")

evaluation_tab = VBox(
    [section_header("🧪  Evaluation Settings"),
     eval_model_path_input,
     eval_data_dir_input,
     eval_label_files_input,
     HBox(
         [widgets.HTML("<div style='flex:1'></div>"),
          eval_status_label, eval_stop_btn, eval_btn],
         layout=Layout(width="100%", align_items="center", gap="8px"),
     ),
     eval_log],
    layout=Layout(width="100%", padding="12px"),
)

def on_evaluate(_):
    log_clear(eval_log)
    model_path  = eval_model_path_input.value.strip()
    data_dir    = eval_data_dir_input.value.strip()
    label_files = [l.strip() for l in eval_label_files_input.value.splitlines() if l.strip()]

    if not model_path:
        log_append(eval_log, "⚠️  Please enter a model path.\n"); return
    if not os.path.isfile(CONFIG_SAVE_PATH):
        log_append(eval_log, f"⚠️  Config not found at '{CONFIG_SAVE_PATH}'. Save config first.\n"); return

    # Patch eval fields directly into the YAML file so PaddleOCR reads a clean list
    try:
        with open(CONFIG_SAVE_PATH, "r") as f:
            config_data = yaml.safe_load(f) or {}
        config_data.setdefault("Global", {})["checkpoints"] = model_path
        config_data.setdefault("Eval",   {}).setdefault("dataset", {})
        config_data["Eval"]["dataset"]["data_dir"]        = data_dir
        config_data["Eval"]["dataset"]["label_file_list"] = label_files
        with open(CONFIG_SAVE_PATH, "w") as f:
            yaml.dump(config_data, f, sort_keys=False, allow_unicode=True)
        log_append(eval_log,
            f"💾  Eval config patched → {CONFIG_SAVE_PATH}\n"
            f"    checkpoints     : {model_path}\n"
            f"    data_dir        : {data_dir}\n"
            f"    label_file_list : {label_files}\n"
            + "─" * 60 + "\n"
        )
    except Exception as exc:
        log_append(eval_log, f"❌  Failed to patch config: {exc}\n"); return

    eval_btn.disabled        = True
    eval_stop_btn.disabled   = False
    eval_status_label.value  = status_html("⏳ Evaluating…", "#e65100")

    def _on_done():
        eval_btn.disabled       = False
        eval_stop_btn.disabled  = True
        eval_status_label.value = status_html("✅ Done", "#2e7d32")

    run_command(
        ["python3", "-u", EVAL_SCRIPT, "-c", CONFIG_SAVE_PATH],
        eval_log, "eval", "✅ Evaluation finished", on_finish_cb=_on_done,
    )

def on_stop_eval(_):
    stop_process("eval", eval_log, eval_status_label, eval_btn, eval_stop_btn)

eval_btn.on_click(on_evaluate)
eval_stop_btn.on_click(on_stop_eval)


# ═══════════════════════════════════════════════════════════════════════════════
# ── TAB 3 · Export Model ──────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════
export_model_path_input = widgets.Text(
    placeholder="Trained model checkpoint path…",
    description="Model Path:",
    style=LABEL_STYLE, layout=Layout(width="100%", height="32px"),
)
export_save_dir_input = widgets.Text(
    placeholder="Output directory for inference model…",
    description="Output Dir:",
    style=LABEL_STYLE, layout=Layout(width="100%", height="32px"),
)
export_btn              = make_btn("📦  Export",   "file-export", "#E3F2FD", "160px")
export_stop_btn         = make_btn("■  Stop",      "stop",        "#FFEBEE", "120px")
download_btn            = make_btn("⬇  Download",  "download",    "#F3E5F5", "160px")
export_stop_btn.disabled = True
export_status_label     = widgets.HTML(value="")
export_log              = make_log("300px")
export_link             = widgets.HTML(value="")   # FIX: separate widget for download link

export_model_tab = VBox(
    [section_header("📦  Export Paddle Inference Model"),
     export_model_path_input,
     export_save_dir_input,
     HBox(
         [widgets.HTML("<div style='flex:1'></div>"),
          export_status_label, download_btn, export_stop_btn, export_btn],
         layout=Layout(width="100%", align_items="center", gap="8px"),
     ),
     export_log,
     export_link],
    layout=Layout(width="100%", padding="12px"),
)

def on_export(_):
    log_clear(export_log)
    export_link.value = ""
    model_path = export_model_path_input.value.strip()
    save_dir   = export_save_dir_input.value.strip()
    if not model_path or not save_dir:
        log_append(export_log, "⚠️  Please fill in both Model Path and Output Dir.\n"); return
    if not os.path.isfile(CONFIG_SAVE_PATH):
        log_append(export_log, f"⚠️  Config not found at '{CONFIG_SAVE_PATH}'. Save config first.\n"); return

    cmd = [
        "python3", "-u", EXPORT_SCRIPT,
        "-c", CONFIG_SAVE_PATH,
        "-o",
        f"Global.pretrained_model={model_path}",
        f"Global.save_inference_dir={save_dir}",
    ]
    export_btn.disabled        = True
    export_stop_btn.disabled   = False
    export_status_label.value  = status_html("⏳ Exporting…", "#e65100")

    def _on_done():
        export_btn.disabled       = False
        export_stop_btn.disabled  = True
        export_status_label.value = status_html("✅ Done", "#2e7d32")

    run_command(cmd, export_log, "export", "✅ Export finished", on_finish_cb=_on_done)

def on_stop_export(_):
    stop_process("export", export_log, export_status_label, export_btn, export_stop_btn)

def on_download_export(_):
    save_dir = export_save_dir_input.value.strip()
    if not save_dir:
        log_clear(export_log)
        log_append(export_log, "⚠️  Please fill in Output Dir.\n"); return

    log_clear(export_log)
    export_link.value = ""
    zip_path = f"{save_dir}.zip"

    def _show_link():
        export_link.value = download_link_html(zip_path)  # FIX: update HTML widget directly

    run_command(["zip", "-rq", zip_path, save_dir],
                export_log, "export", "✅ Zip finished", on_finish_cb=_show_link)

export_btn.on_click(on_export)
export_stop_btn.on_click(on_stop_export)
download_btn.on_click(on_download_export)


# ═══════════════════════════════════════════════════════════════════════════════
# ── TAB 4 · Convert to ONNX ───────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════
paddle_model_dir_input = widgets.Text(
    placeholder="Paddle inference model directory…",
    description="Paddle Model Dir:",
    style=LABEL_STYLE, layout=Layout(width="100%", height="32px"),
)
onnx_output_dir_input = widgets.Text(
    placeholder="ONNX output directory…",
    description="ONNX Output Dir:",
    style=LABEL_STYLE, layout=Layout(width="100%", height="32px"),
)
convert_btn              = make_btn("🔄  Convert",  "connectdevelop", "#E3F2FD", "160px")
convert_stop_btn         = make_btn("■  Stop",      "stop",           "#FFEBEE", "120px")
download_onnx_btn        = make_btn("⬇  Download",  "download",       "#F3E5F5", "160px")
convert_stop_btn.disabled = True
convert_status_label     = widgets.HTML(value="")
convert_log              = make_log("300px")
convert_link             = widgets.HTML(value="")  # FIX: separate widget for download link

convert_model_tab = VBox(
    [section_header("🔄  Convert Paddle → ONNX"),
     paddle_model_dir_input,
     onnx_output_dir_input,
     HBox(
         [widgets.HTML("<div style='flex:1'></div>"),
          convert_status_label, download_onnx_btn, convert_stop_btn, convert_btn],
         layout=Layout(width="100%", align_items="center", gap="8px"),
     ),
     convert_log,
     convert_link],
    layout=Layout(width="100%", padding="12px"),
)

def on_convert(_):
    log_clear(convert_log)
    convert_link.value = ""
    paddle_dir = paddle_model_dir_input.value.strip()
    onnx_dir   = onnx_output_dir_input.value.strip()
    if not paddle_dir or not onnx_dir:
        log_append(convert_log, "⚠️  Please fill in both directories.\n"); return

    convert_btn.disabled        = True
    convert_stop_btn.disabled   = False
    convert_status_label.value  = status_html("⏳ Converting…", "#e65100")

    def _on_done():
        convert_btn.disabled       = False
        convert_stop_btn.disabled  = True
        convert_status_label.value = status_html("✅ Done", "#2e7d32")

    run_command(
        ["paddlex", "--paddle2onnx",
         "--paddle_model_dir", paddle_dir,
         "--onnx_model_dir",   onnx_dir],
        convert_log, "convert", "✅ Conversion finished", on_finish_cb=_on_done,
    )

def on_stop_convert(_):
    stop_process("convert", convert_log, convert_status_label, convert_btn, convert_stop_btn)

def on_download_onnx(_):
    onnx_dir = onnx_output_dir_input.value.strip()
    if not onnx_dir:
        log_clear(convert_log)
        log_append(convert_log, "⚠️  Please fill in ONNX Output Dir.\n"); return

    log_clear(convert_log)
    convert_link.value = ""
    zip_path = f"{onnx_dir}.zip"

    def _show_link():
        convert_link.value = download_link_html(zip_path)  # FIX: update HTML widget directly

    run_command(["zip", "-rq", zip_path, onnx_dir],
                convert_log, "convert", "✅ Zip finished", on_finish_cb=_show_link)

convert_btn.on_click(on_convert)
convert_stop_btn.on_click(on_stop_convert)
download_onnx_btn.on_click(on_download_onnx)


# ═══════════════════════════════════════════════════════════════════════════════
# ── Root Tab Widget ───────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════
task_tab = widgets.Tab(
    children=[training_tab, evaluation_tab, export_model_tab, convert_model_tab],
    layout=Layout(width="100%"),
)
for i, title in enumerate([
    "🚂  Training", "🧪  Evaluation", "📦  Export Model", "🔄  Convert Model",
]):
    task_tab.set_title(i, title)

display(task_tab)