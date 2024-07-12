import gradio as gr
from ultralytics import YOLO
from ultralytics_tracking import workplace_tracking
from triton_run import triton_run_server

iface = gr.Interface(
    fn=workplace_tracking,
    inputs=[
        gr.Video(sources='upload', format='mp4', label='Input video'),
        gr.Slider(minimum=0, maximum=1, value=0.2, label="Confidence threshold"),
        gr.Slider(minimum=0, maximum=1, value=0.3, label="IoU threshold"),
    ],
    outputs=[
        gr.Video(format='mp4', autoplay=True),
    ],
    title="Workplace Tracking",
    description="Upload video for inference."
)

if __name__ == "__main__":
    triton_run_server()
    iface.launch()