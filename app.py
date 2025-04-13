# MIT License

# Copyright (c) Microsoft

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Copyright (c) [2025] [Microsoft]
# Copyright (c) [2025] Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT
# This file has been modified by Bytedance Ltd. and/or its affiliates on 2025/04/10
# Original file was released under MIT, with the full license text # available at https://github.com/atong01/conditional-flow-matching/blob/1.0.7/LICENSE.
# This modified file is released under the same license.

import gradio as gr
import os
os.environ['SPCONV_ALGO'] = 'native'
from typing import *
import torch
import numpy as np
from hi3dgen.pipelines import Hi3DGenPipeline
import trimesh
import tempfile

MAX_SEED = np.iinfo(np.int32).max
TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')
WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights')
os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)

def cache_weights(weights_dir: str) -> dict:
    import os
    from huggingface_hub import snapshot_download

    os.makedirs(weights_dir, exist_ok=True)
    model_ids = [
        "Stable-X/trellis-normal-v0-1",
        "Stable-X/yoso-normal-v1-8-1",
        "ZhengPeng7/BiRefNet",
    ]
    cached_paths = {}
    for model_id in model_ids:
        print(f"Caching weights for: {model_id}")
        # Check if the model is already cached
        local_path = os.path.join(weights_dir, model_id.split("/")[-1])
        if os.path.exists(local_path):
            print(f"Already cached at: {local_path}")
            cached_paths[model_id] = local_path
            continue
        # Download the model and cache it
        print(f"Downloading and caching model: {model_id}")
        # Use snapshot_download to download the model
        local_path = snapshot_download(repo_id=model_id, local_dir=os.path.join(weights_dir, model_id.split("/")[-1]), force_download=False)
        cached_paths[model_id] = local_path
        print(f"Cached at: {local_path}")

    return cached_paths

def preprocess_mesh(mesh_prompt):
    print("Processing mesh")
    trimesh_mesh = trimesh.load_mesh(mesh_prompt)
    trimesh_mesh.export(mesh_prompt+'.glb')
    return mesh_prompt+'.glb'

def preprocess_image(image):
    if image is None:
        return None
    image = hi3dgen_pipeline.preprocess_image(image, resolution=1024)
    return image

def generate_3d(image, seed=-1,  
                ss_guidance_strength=3, ss_sampling_steps=50,
                slat_guidance_strength=3, slat_sampling_steps=6,):
    if image is None:
        return None, None, None

    if seed == -1:
        seed = np.random.randint(0, MAX_SEED)
    
    image = hi3dgen_pipeline.preprocess_image(image, resolution=1024)
    normal_image = normal_predictor(image, resolution=768, match_input_resolution=True, data_type='object')

    outputs = hi3dgen_pipeline.run(
        normal_image,
        seed=seed,
        formats=["mesh",],
        preprocess_image=False,
        sparse_structure_sampler_params={
            "steps": ss_sampling_steps,
            "cfg_strength": ss_guidance_strength,
        },
        slat_sampler_params={
            "steps": slat_sampling_steps,
            "cfg_strength": slat_guidance_strength,
        },
    )
    generated_mesh = outputs['mesh'][0]
    
    # Save outputs
    import datetime
    output_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    os.makedirs(os.path.join(TMP_DIR, output_id), exist_ok=True)
    mesh_path = f"{TMP_DIR}/{output_id}/mesh.glb"
    
    # Export mesh
    trimesh_mesh = generated_mesh.to_trimesh(transform_pose=True)

    trimesh_mesh.export(mesh_path)

    return normal_image, mesh_path, mesh_path

def convert_mesh(mesh_path, export_format):
    """Download the mesh in the selected format."""
    if not mesh_path:
        return None
    
    # Create a temporary file to store the mesh data
    temp_file = tempfile.NamedTemporaryFile(suffix=f".{export_format}", delete=False)
    temp_file_path = temp_file.name
    
    new_mesh_path = mesh_path.replace(".glb", f".{export_format}")
    mesh = trimesh.load_mesh(mesh_path)
    mesh.export(temp_file_path)  # Export to the temporary file
    
    return temp_file_path # Return the path to the temporary file

# Create the Gradio interface with improved layout
with gr.Blocks(css="footer {visibility: hidden}") as demo:
    gr.Markdown(
        """
        <h1 style='text-align: center;'>Hi3DGen: High-fidelity 3D Geometry Generation from Images via Normal Bridging</h1>
        <p style='text-align: center;'>
            <strong>V0.1, Introduced By 
            <a href="https://gaplab.cuhk.edu.cn/" target="_blank">GAP Lab</a> from CUHKSZ and 
            <a href="https://www.nvsgames.cn/" target="_blank">Game-AIGC Team</a> from ByteDance</strong>
        </p>
        """
    )
    
    with gr.Row():
        gr.Markdown("""
                    <p align="center">
                    <a title="Website" href="https://stable-x.github.io/Hi3DGen/" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
                        <img src="https://www.obukhov.ai/img/badges/badge-website.svg">
                    </a>
                    <a title="arXiv" href="https://stable-x.github.io/Hi3DGen/hi3dgen_paper.pdf" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
                        <img src="https://www.obukhov.ai/img/badges/badge-pdf.svg">
                    </a>
                    <a title="Github" href="https://github.com/Stable-X/Hi3DGen" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
                        <img src="https://img.shields.io/github/stars/Stable-X/Hi3DGen?label=GitHub%20%E2%98%85&logo=github&color=C8C" alt="badge-github-stars">
                    </a>
                    <a title="Social" href="https://x.com/ychngji6" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
                        <img src="https://www.obukhov.ai/img/badges/badge-social.svg" alt="social">
                    </a>
                    </p>
                    """)

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Tabs():
                
                with gr.Tab("Single Image"):
                    with gr.Row():
                        image_prompt = gr.Image(label="Image Prompt", image_mode="RGBA", type="pil")
                        normal_output = gr.Image(label="Normal Bridge", image_mode="RGBA", type="pil")
                        
                with gr.Tab("Multiple Images"):
                    gr.Markdown("<div style='text-align: center; padding: 40px; font-size: 24px;'>Multiple Images functionality is coming soon!</div>")
                        
            with gr.Accordion("Advanced Settings", open=False):
                seed = gr.Slider(-1, MAX_SEED, label="Seed", value=0, step=1)
                gr.Markdown("#### Stage 1: Sparse Structure Generation")
                with gr.Row():
                    ss_guidance_strength = gr.Slider(0.0, 10.0, label="Guidance Strength", value=3, step=0.1)
                    ss_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=50, step=1)
                gr.Markdown("#### Stage 2: Structured Latent Generation")
                with gr.Row():
                    slat_guidance_strength = gr.Slider(0.0, 10.0, label="Guidance Strength", value=3.0, step=0.1)
                    slat_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=6, step=1)
                    
            with gr.Group():
                with gr.Row():
                    gen_shape_btn = gr.Button("Generate Shape", size="lg", variant="primary")
                        
        # Right column - Output
        with gr.Column(scale=1):
            with gr.Column():
                model_output = gr.Model3D(label="3D Model Preview (Each model is approximately 40MB, may take around 1 minute to load)")
            with gr.Column():
                export_format = gr.Dropdown(
                    choices=["obj", "glb", "ply", "stl"],
                    value="glb",
                    label="File Format"
                )
                download_btn = gr.DownloadButton(label="Export Mesh", interactive=False)

    image_prompt.upload(
        preprocess_image,
        inputs=[image_prompt],
        outputs=[image_prompt]
    )
    
    gen_shape_btn.click(
        generate_3d,
        inputs=[
            image_prompt, seed,  
            ss_guidance_strength, ss_sampling_steps,
            slat_guidance_strength, slat_sampling_steps
        ],
        outputs=[normal_output, model_output, download_btn]
    ).then(
        lambda: gr.Button(interactive=True),
        outputs=[download_btn],
    )
    
    
    def update_download_button(mesh_path, export_format):
        if not mesh_path:
            return gr.File.update(value=None, interactive=False)
        
        download_path = convert_mesh(mesh_path, export_format)
        return download_path
    
    export_format.change(
        update_download_button,
        inputs=[model_output, export_format],
        outputs=[download_btn]
    ).then(
        lambda: gr.Button(interactive=True),
        outputs=[download_btn],
    )
    
    examples = gr.Examples(
        examples=[
            f'assets/example_image/{image}'
            for image in os.listdir("assets/example_image")
        ],
        inputs=image_prompt,
    )

    gr.Markdown(
        """
        **Acknowledgments**: Hi3DGen is built on the shoulders of giants. We would like to express our gratitude to the open-source research community and the developers of these pioneering projects:
        - **3D Modeling:** Our 3D Model is finetuned from the SOTA open-source 3D foundation model [Trellis](https://github.com/microsoft/TRELLIS) and we draw inspiration from the teams behind [Rodin](https://hyperhuman.deemos.com/rodin), [Tripo](https://www.tripo3d.ai/app/home), and [Dora](https://github.com/Seed3D/Dora).
        - **Normal Estimation:** Our Normal Estimation Model builds on the leading normal estimation research such as [StableNormal](https://github.com/hugoycj/StableNormal) and [GenPercept](https://github.com/aim-uofa/GenPercept).
        
        **Your contributions and collaboration push the boundaries of 3D modeling!**
        """
    )

if __name__ == "__main__":
    # Download and cache the weights
    cache_weights(WEIGHTS_DIR)

    hi3dgen_pipeline = Hi3DGenPipeline.from_pretrained("weights/trellis-normal-v0-1")
    hi3dgen_pipeline.cuda()

    # Initialize normal predictor
    try:
        normal_predictor = torch.hub.load(os.path.join(torch.hub.get_dir(), 'hugoycj_StableNormal_main'), "StableNormal_turbo", yoso_version='yoso-normal-v1-8-1', source='local', local_cache_dir='./weights', pretrained=True)
    except:
        normal_predictor = torch.hub.load("hugoycj/StableNormal", "StableNormal_turbo", trust_repo=True, yoso_version='yoso-normal-v1-8-1', local_cache_dir='./weights')    

    # Launch the app
    demo.launch(share=False, server_name="0.0.0.0", server_port=6073, auth=("aabb", "aabb"))

