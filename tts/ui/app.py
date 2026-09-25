"""Web UI: load a checkpoint and synthesize speech, and run/monitor training.

Launch with ``python scripts/webui.py`` (see its --help).
"""

from pathlib import Path

import gradio as gr

from tts.ui.backend import (
    ModelSession,
    SynthesisParams,
    TrainingManager,
    list_checkpoints,
    list_sample_steps,
    read_metrics,
    read_samples,
)

MAX_SAMPLES = 4  # eval sample slots shown in the Train tab

CSS = """
.gradio-container { max-width: 1280px !important; width: 100% !important; margin: 0 auto; }
#title h1 { margin-bottom: 0; }
#title p { margin-top: 4px; opacity: 0.75; }
.stat { border: 1px solid var(--border-color-primary); border-radius: 10px; padding: 10px 14px; }
.stat .label { font-size: 12px; opacity: 0.7; text-transform: uppercase; letter-spacing: .04em; }
.stat .value { font-size: 20px; font-weight: 600; font-variant-numeric: tabular-nums; white-space: nowrap; }
.status-ok { color: var(--color-green-500, #16a34a); }
.status-idle { opacity: 0.7; }
"""

TRAIN_FIELDS = [
    # (config key, label, kind)
    ("max_steps", "Max steps", "int"),
    ("lr", "Learning rate", "float"),
    ("max_tokens_per_batch", "Tokens per micro-batch", "int"),
    ("grad_accum", "Gradient accumulation", "int"),
    ("eval_every", "Evaluate every (steps)", "int"),
    ("save_every", "Checkpoint every (steps)", "int"),
]


def _stat(label: str, value) -> str:
    return f'<div class="stat"><div class="label">{label}</div><div class="value">{value}</div></div>'


def _fmt(value, spec: str = "") -> str:
    if value is None or value == "":
        return "–"
    return format(value, spec) if spec else str(value)


def build_app(runs_root: str = "runs", configs_root: str = "configs", codec: str = "kyutai/mimi") -> gr.Blocks:
    session = ModelSession()
    manager = TrainingManager(runs_root)

    def config_choices() -> list[str]:
        return sorted(str(p) for p in Path(configs_root).glob("*.yaml"))

    with gr.Blocks(title="TTS Studio", fill_width=True) as app:
        gr.Markdown("# TTS Studio\nLoad a checkpoint to generate speech, or start and monitor training.", elem_id="title")

        # ---------------------------------------------------------------- Speak
        with gr.Tab("Speak"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=4):
                    gr.Markdown("### 1 · Model")
                    ckpt = gr.Dropdown(
                        choices=list_checkpoints(runs_root), label="Checkpoint", allow_custom_value=True,
                        info="Pick a training checkpoint or paste a path.",
                    )
                    with gr.Row():
                        refresh_ckpts = gr.Button("↻ Refresh", size="sm")
                        load_btn = gr.Button("Load model", variant="primary", size="sm")
                    with gr.Accordion("Advanced", open=False):
                        device = gr.Dropdown(["auto", "cuda", "cpu"], value="auto", label="Device")
                        codec_path = gr.Textbox(value=codec, label="Codec", info="Hugging Face id or local path")
                    model_status = gr.Markdown("No model loaded.", elem_classes="status-idle")

                    gr.Markdown("### 2 · Voice (optional)")
                    ref_audio = gr.Audio(label="Reference clip (5–20 s)", type="filepath", sources=["upload", "microphone"])
                    ref_text = gr.Textbox(label="Reference transcript", lines=2, placeholder="Exactly what is said in the clip")

                with gr.Column(scale=6):
                    gr.Markdown("### 3 · Text")
                    text = gr.Textbox(
                        label="Text to speak", lines=7, max_lines=20,
                        placeholder="Type what the voice should say…",
                    )
                    with gr.Accordion("Sampling", open=False):
                        with gr.Row():
                            temperature = gr.Slider(0.0, 1.5, value=0.8, step=0.05, label="Temperature", info="0 = deterministic")
                            top_p = gr.Slider(0.1, 1.0, value=0.8, step=0.05, label="Top-p")
                        with gr.Row():
                            top_k = gr.Slider(1, 200, value=30, step=1, label="Top-k")
                            max_seconds = gr.Slider(2, 120, value=30, step=1, label="Max length (s)")
                        seed = gr.Number(value=None, label="Seed (empty = random)", precision=0)
                    generate_btn = gr.Button("Generate speech", variant="primary", size="lg", interactive=False)
                    output_audio = gr.Audio(label="Result", type="numpy", interactive=False)
                    gen_info = gr.Markdown("")

            def on_refresh_ckpts():
                return gr.update(choices=list_checkpoints(runs_root))

            def on_load(path, dev, codec_id):
                if not path:
                    raise gr.Error("Choose a checkpoint first.")
                try:
                    msg = session.load(path, codec_id, dev)
                except Exception as e:  # surface load errors in the UI
                    raise gr.Error(f"Could not load model: {e}") from e
                return gr.update(value="✅ " + msg, elem_classes="status-ok"), gr.update(interactive=True)

            def on_generate(txt, ra, rt, temp, tp, tk, secs, sd):
                params = SynthesisParams(temperature=temp, top_p=tp, top_k=int(tk), max_seconds=secs,
                                         seed=int(sd) if sd not in (None, "") else None)
                try:
                    sr, audio, info = session.synthesize(txt, params, ref_audio=ra, ref_text=rt)
                except (ValueError, RuntimeError) as e:
                    raise gr.Error(str(e)) from e
                return (sr, audio), info

            refresh_ckpts.click(on_refresh_ckpts, outputs=ckpt)
            load_btn.click(on_load, [ckpt, device, codec_path], [model_status, generate_btn])
            generate_btn.click(
                on_generate,
                [text, ref_audio, ref_text, temperature, top_p, top_k, max_seconds, seed],
                [output_audio, gen_info],
            )

        # ---------------------------------------------------------------- Train
        with gr.Tab("Train"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=4):
                    gr.Markdown("### Start a run")
                    cfg_file = gr.Dropdown(config_choices(), value=(config_choices() or [None])[0], label="Base config")
                    run_dir = gr.Textbox(label="Run directory", placeholder="runs/pretrain_s (from config if empty)")
                    data_dirs = gr.Textbox(label="Data shard directories", lines=2, placeholder="One per line (from config if empty)")
                    field_inputs = []
                    for i in range(0, len(TRAIN_FIELDS), 2):
                        with gr.Row():
                            for key, label, kind in TRAIN_FIELDS[i : i + 2]:
                                field_inputs.append(gr.Textbox(label=label, placeholder="from config"))
                    extra = gr.Textbox(label="Extra overrides", lines=2, placeholder="key=value, one per line (e.g. ref_prob=0.3)")
                    with gr.Row():
                        start_btn = gr.Button("▶ Start / resume", variant="primary")
                        stop_btn = gr.Button("■ Stop & save", variant="stop")
                    train_msg = gr.Markdown("")
                    gr.Markdown(
                        "Empty fields use the base config. Training runs in the background and keeps going "
                        "if you close this page; **Stop & save** finishes the current step and writes a checkpoint.",
                        elem_classes="status-idle",
                    )

                with gr.Column(scale=8):
                    gr.Markdown("### Monitor")
                    with gr.Row():
                        run_pick = gr.Dropdown(manager.list_runs(), label="Run", allow_custom_value=True, scale=4)
                        refresh_runs = gr.Button("↻", size="sm", scale=0, min_width=48)
                        auto = gr.Checkbox(value=True, label="Auto-refresh (5 s)", scale=1)
                    run_state = gr.Markdown("")
                    with gr.Row():
                        stat_step = gr.HTML(_stat("Step", "–"))
                        stat_loss = gr.HTML(_stat("Train loss", "–"))
                        stat_val = gr.HTML(_stat("Val loss", "–"))
                        stat_lr = gr.HTML(_stat("LR", "–"))
                        stat_speed = gr.HTML(_stat("Audio h / h", "–"))
                    # Fixed color maps: the plot must know its series before the
                    # first data arrives, or a run that starts empty never draws lines.
                    train_plot = gr.LinePlot(
                        x="step", y="value", color="series", title="Training loss", height=260,
                        color_map={"loss": "#4f46e5", "slow_loss": "#0ea5e9", "fast_loss": "#f59e0b"},
                    )
                    val_plot = gr.LinePlot(
                        x="step", y="value", color="series", title="Validation loss", height=220,
                        color_map={"val_loss": "#4f46e5", "val_slow_loss": "#0ea5e9", "val_fast_loss": "#f59e0b"},
                    )
                    with gr.Accordion("Log", open=False):
                        log_box = gr.Code(language=None, lines=14, interactive=False)

                    gr.Markdown("### Eval samples")
                    sample_step = gr.Dropdown([], label="Step")
                    sample_rows = []
                    for k in range(MAX_SAMPLES):
                        with gr.Group(visible=False) as grp:
                            s_text = gr.Markdown()
                            with gr.Row():
                                s_gen = gr.Audio(label="Generated", type="filepath", interactive=False)
                                s_ref = gr.Audio(label="Ground truth (codec)", type="filepath", interactive=False)
                        sample_rows.append((grp, s_text, s_gen, s_ref))

            def collect_overrides(rd, dd, *vals):
                *fields, extra_txt = vals
                overrides = []
                if rd.strip():
                    overrides.append(f"run_dir={rd.strip()}")
                dirs = [d.strip() for d in dd.splitlines() if d.strip()]
                if dirs:
                    overrides.append("data=[" + ", ".join(dirs) + "]")
                for (key, _, _), v in zip(TRAIN_FIELDS, fields):
                    if str(v).strip():
                        overrides.append(f"{key}={str(v).strip()}")
                overrides += [l.strip() for l in extra_txt.splitlines() if l.strip()]
                return overrides

            def on_start(cfg_path, rd, dd, *vals):
                if not cfg_path:
                    raise gr.Error("Choose a base config.")
                overrides = collect_overrides(rd, dd, *vals)
                try:
                    started = manager.start(cfg_path, overrides)
                except Exception as e:
                    raise gr.Error(str(e)) from e
                return f"Started training in {started}", gr.update(choices=manager.list_runs(), value=started)

            def on_stop(run):
                if not run:
                    raise gr.Error("Choose a run.")
                if manager.stop(run):
                    return "Stop requested: finishing the current step and saving a checkpoint…"
                return "That run is not running."

            def on_refresh(run, current_step):
                if not run:
                    return (
                        "Pick a run to monitor.",
                        *(_stat(l, "–") for l in ("Step", "Train loss", "Val loss", "LR", "Audio h / h")),
                        None, None, "", gr.update(choices=[], value=None), *sample_updates(None, None),
                    )
                train_df, val_df, latest = read_metrics(run)
                state = "🟢 **Running**" if manager.is_running(run) else "⚪ **Not running**"
                steps = list_sample_steps(run)
                step = current_step if current_step in steps else (steps[0] if steps else None)
                return (
                    state,
                    _stat("Step", _fmt(latest.get("step"))),
                    _stat("Train loss", _fmt(latest.get("loss"), ".3f")),
                    _stat("Val loss", _fmt(latest.get("val_loss"), ".3f")),
                    _stat("LR", _fmt(latest.get("lr"), ".1e")),
                    _stat("Audio h / h", _fmt(latest.get("audio_hours_per_h"))),
                    # None (not an empty frame) until data exists: a plot first
                    # given an empty frame never draws lines for later data.
                    train_df if not train_df.empty else None,
                    val_df if not val_df.empty else None,
                    manager.log_tail(run),
                    gr.update(choices=steps, value=step),
                    *sample_updates(run, step),
                )

            def sample_updates(run, step):
                samples = read_samples(run, step) if run and step else []
                updates = []
                for k in range(MAX_SAMPLES):
                    if k < len(samples):
                        s = samples[k]
                        updates += [gr.update(visible=True), f"**{k + 1}.** {s['text']}", s["generated"], s["reference"]]
                    else:
                        updates += [gr.update(visible=False), "", None, None]
                return updates

            sample_outputs = [c for row in sample_rows for c in row]
            monitor_outputs = [run_state, stat_step, stat_loss, stat_val, stat_lr, stat_speed,
                               train_plot, val_plot, log_box, sample_step, *sample_outputs]

            start_btn.click(on_start, [cfg_file, run_dir, data_dirs, *field_inputs, extra], [train_msg, run_pick])
            stop_btn.click(on_stop, run_pick, train_msg)
            refresh_runs.click(lambda: gr.update(choices=manager.list_runs()), outputs=run_pick)
            run_pick.change(on_refresh, [run_pick, sample_step], monitor_outputs)
            sample_step.input(lambda r, s: sample_updates(r, s), [run_pick, sample_step], sample_outputs)

            timer = gr.Timer(5.0)
            timer.tick(on_refresh, [run_pick, sample_step], monitor_outputs)
            auto.change(lambda on: gr.Timer(active=on), auto, timer)

    return app


def launch(host: str = "127.0.0.1", port: int = 7860, auth: tuple[str, str] | None = None, **kwargs) -> None:
    app = build_app(**kwargs)
    app.launch(
        server_name=host, server_port=port, auth=auth,
        theme=gr.themes.Soft(primary_hue="indigo", neutral_hue="slate"), css=CSS,
        allowed_paths=[str(Path(kwargs.get("runs_root", "runs")).resolve())],
    )
