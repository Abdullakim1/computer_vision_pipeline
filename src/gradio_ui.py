"""CineForge Web UI - Professional Gradio Interface.

Run:  PYTHONPATH=. python3 -m src.gradio_ui

Features:
- Text-to-video with multiple backends
- Image-to-video with Ken Burns effect
- Storyboard film generation
- Video grading
- Quality analysis
- Showcase portfolio generation
"""

try:
    import gradio as gr
    _HAS_GRADIO = True
except Exception:  # pragma: no cover
    gr = None
    _HAS_GRADIO = False


def _brief(clip):
    m = clip.metadata.get("metrics", {})
    return (f"frames={clip.T}  {m.get('duration_s')}s  "
            f"flow={m.get('mean_optical_flow_px')}  sharp={m.get('avg_sharpness')}")


def _build_studio():
    from src.forge import CineForgeStudio
    return CineForgeStudio()


def text_to_video(prompt, backend, width, height, duration, fps, seed, look, motion, style):
    """Generate video from text prompt."""
    studio = _build_studio()
    clip = studio.text_to_video(
        prompt=prompt,
        backend=backend,
        width=width,
        height=height,
        fps=fps,
        duration=duration,
        seed=seed,
        look=look,
        motion=motion,
        style=style,
    )
    clip.write_video("outputs/ui.mp4")
    gif = clip.to_gif("outputs/ui.gif", fps=min(12, clip.T))
    metrics = clip.metadata.get("metrics", {})
    result = _brief(clip)
    return "outputs/ui.gif", "outputs/ui.mp4", f"1  Generated!\n{result}"


def image_to_video(image, look, width, height, duration, fps, interp, motion):
    """Generate video from static image (Ken Burns)."""
    studio = _build_studio()
    clip = studio.image_to_video(
        image_path=image,
        width=width,
        height=height,
        look=look,
        duration=duration,
        fps=fps,
        interp=interp,
    )
    clip.write_video("outputs/ui_i2v.mp4")
    gif = clip.to_gif("outputs/ui_i2v.gif", fps=24)
    metrics = clip.metadata.get("metrics", {})
    return "outputs/ui_i2v.gif", "outputs/ui_i2v.mp4", f"1  Generated!\n{metrics.get('perceptual_quality', 'N/A')}"


def story_to_video(story, beats, width, height, fps, seed, look):
    """Generate storyboard film."""
    studio = _build_studio()
    clip = studio.direct(
        prompt=story,
        beats=beats,
        width=width,
        height=height,
        fps=fps,
        seed=seed,
        look=look,
    )
    clip.write_video("outputs/ui_story.mp4")
    gif = clip.to_gif("outputs/ui_story.gif", fps=24)
    return "outputs/ui_story.gif", "outputs/ui_story.mp4"


def grade_video(image, look):
    """Apply color grading to image."""
    studio = _build_studio()
    clip = studio.text_to_video(image, look=look)
    clip = studio.regrade(clip, look)
    clip.write_video("outputs/ui_graded.mp4")
    return f"outputs/ui_graded.mp4"


def analyze_video(video, fps):
    """Analyze video quality."""
    import cv2
    cap = cv2.VideoCapture(video)
    frames = []
    while len(frames) < 300 and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()
    
    studio = _build_studio()
    if frames:
        clip = studio.text_to_video("analysis", backend="cinematic", frames=frames, fps=fps)
        metrics = clip.metadata.get("metrics", {})
        return (
            f"Sharpness: {metrics.get('avg_sharpness', 'N/A')}\n"
            f"Flow: {metrics.get('mean_optical_flow_px', 'N/A')}\n"
            f"Quality: {metrics.get('perceptual_quality', 'N/A')}\n"
            f"Color: {metrics.get('avg_colorfulness', 'N/A')}"
        )
    return "No frames detected"


def create_showcase(look):
    """Generate showcase portfolio."""
    studio = _build_studio()
    prompts = [
        "cinematic aurora over snowy mountain valley, teal and violet light, orbit camera",
        "golden hour over peaceful meadow, slow pan",
        "cyberpunk city night scene, neon signs, pan right",
        "moody ocean waves at sunset, slow zoom in",
        "dark stormy sky with lightning, dramatic dutch tilt",
    ]
    themes = ["aurora", "golden", "cyber", "moody", "storm"]
    
    outputs = []
    for prompt, theme in zip(prompts, themes):
        clip = studio.text_to_video(
            prompt, backend="cinematic", width=960, height=540,
            fps=24, duration=4, seed=sum(ord(c) for c in theme)
        )
        clip.write_video(f"outputs/showcase_{theme}.mp4")
        gif = clip.to_gif(f"outputs/showcase_{theme}.gif", fps=10, scale=0.5)
        outputs.append(f"✅ {theme}: {gif}")
    
    return "\n".join(outputs)


def build():
    """Build and return a professional Gradio interface."""
    if not _HAS_GRADIO:
        print("Gradio is not installed. Run 'pip install gradio'.")
        return None
    
    studio = _build_studio()
    
    # Get available options
    looks = studio.get_available_looks()
    themes = studio.get_available_themes()
    motions = studio.get_available_motions()
    backends = [b["name"] for b in studio.backends() if b["ready"]]
    if not backends:
        backends = ["cinematic"]
    
    with gr.Blocks(
        title="CineForge Studio",
    ) as demo:
        gr.HTML("""
        <div class='header'>
            <h1>🎬 CineForge Video Generation Studio</h1>
            <p>Professional Generative Video Synthesis Platform</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                with gr.Tabs():
                    # Text to Video Tab
                    with gr.Tab("Text → Video"):
                        gr.Markdown("**Describe a cinematic shot and generate it.**")
                        
                        with gr.Row():
                            prompt = gr.Textbox(
                                label="Prompt",
                                value="cinematic aurora over snowy mountain valley, teal and violet light, orbit camera",
                                lines=2,
                            )
                        
                        with gr.Row():
                            backend = gr.Dropdown(
                                choices=["cinematic", "seedance", "kling"],
                                value="cinematic",
                                label="Backend",
                            )
                        
                        with gr.Row():
                            w = gr.Slider(128, 1920, value=960, step=8, label="Width")
                            h = gr.Slider(128, 1080, value=540, step=8, label="Height")
                        
                        with gr.Row():
                            dur = gr.Slider(2, 16, value=4, step=0.5, label="Duration (s)")
                            fps = gr.Slider(8, 60, value=24, step=1, label="FPS")
                        
                        with gr.Row():
                            seed = gr.Number(value=None, label="Seed (optional)")
                            look = gr.Dropdown(choices=looks, value="argo", label="Color Grade")
                        
                        with gr.Row():
                            motion = gr.Dropdown(choices=motions[:6], value="orbit", label="Camera Motion")
                            style = gr.Dropdown(choices=["cinematic", "anime", "3d", "realistic"], value="cinematic", label="Style")
                        
                        with gr.Row():
                            btn = gr.Button("🎬 Generate Video", variant="primary", size="lg")
                        
                        with gr.Row():
                            gif = gr.Image(label="Preview (GIF)")
                            mp4 = gr.Video(label="Video (MP4)")
                        
                        with gr.Accordion("Metrics", open=False):
                            metrics_text = gr.Textbox(label="Quality Metrics", lines=4)
                        
                        btn.click(
                            text_to_video,
                            [prompt, backend, w, h, dur, fps, seed, look, motion, style],
                            [gif, mp4, metrics_text],
                        )
                    
                    # Image to Video Tab
                    with gr.Tab("Image → Video"):
                        gr.Markdown("**Animate a still image with Ken Burns effect.**")
                        
                        with gr.Row():
                            img_input = gr.Image(label="Upload Image", type="filepath")
                        
                        with gr.Row():
                            i_look = gr.Dropdown(choices=looks, value="argo", label="Color Grade")
                            i_interp = gr.Slider(1, 6, value=1, step=1, label="Interpolation")
                        
                        with gr.Row():
                            i_dur = gr.Slider(2, 16, value=4, step=0.5, label="Duration (s)")
                            i_fps = gr.Slider(8, 60, value=24, step=1, label="FPS")
                        
                        with gr.Row():
                            i_motion = gr.Dropdown(choices=motions[:4], value="orbit", label="Motion")
                        
                        with gr.Row():
                            btn2 = gr.Button("🎭 Animate Image", variant="primary")
                        
                        with gr.Row():
                            gif2 = gr.Image()
                            mp4_2 = gr.Video()
                        
                        btn2.click(
                            image_to_video,
                            [img_input, i_look, w, h, i_dur, i_fps, i_interp, i_motion],
                            [gif2, mp4_2],
                        )
                    
                    # Storyboard Tab
                    with gr.Tab("Story → Film"):
                        gr.Markdown("**Turn a story into a short film with storyboard.**")
                        
                        with gr.Row():
                            story = gr.Textbox(
                                label="Story",
                                value="A lone traveler crosses a silent dune under a violet sky."
                            )
                        
                        with gr.Row():
                            beats = gr.Slider(1, 8, value=3, step=1, label="Number of Shots")
                            b_look = gr.Dropdown(choices=looks, value="argo", label="Style")
                        
                        with gr.Row():
                            btn3 = gr.Button("🎞️ Direct Story", variant="primary")
                        
                        with gr.Row():
                            gif3 = gr.Image()
                            mp4_3 = gr.Video()
                        
                        btn3.click(
                            story_to_video,
                            [story, beats, w, h, fps, seed, b_look],
                            [gif3, mp4_3],
                        )
            
            with gr.Column(scale=1):
                gr.HTML("""
                <div class='stats' style='padding: 1rem; border-radius: 0.5rem; color: white; margin-bottom: 1rem;'>
                    <h3>Studio Info</h3>
                    <p>Generations: <strong>{}</strong></p>
                    <p>Backends: <strong>{}</strong></p>
                    <p>Themes: <strong>{}</strong></p>
                    <p>Looks: <strong>{}</strong></p>
                </div>
                <div style='padding: 1rem;'>
                    <h3>Quick Start</h3>
                    <ul>
                        <li>Use <strong>Text → Video</strong> for generative content</li>
                        <li>Use <strong>Image → Video</strong> for animations</li>
                        <li>Use <strong>Story → Film</strong> for movies</li>
                    </ul>
                </div>
                """.format(
                    studio._generation_count,
                    len(backends),
                    len(themes),
                    len(looks),
                ))
    
    return demo


if __name__ == "__main__":
    demo = build()
    if demo is not None:
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            theme=gr.themes.Soft(primary_hue="slate", secondary_hue="blue"),
            css="""
            .gradio-container { font-family: 'Inter', sans-serif; }
            .header { text-align: center; padding: 2rem; }
            .stats { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
            """,
        )
