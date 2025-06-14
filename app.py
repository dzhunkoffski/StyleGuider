import os
from PIL import Image
from omegaconf import OmegaConf
import streamlit as st

from diffusion_core.utils import load_512
from diffusion_core.guiders import GuidanceEditing

from streamlit_application.utils import (
    get_scheduler, get_model, upload_image, click_generate, click_init_editor
)
from streamlit_application.sidebars import (
    vgg_config_input, attn_v_config_input
)

# Load application config
if "config" not in st.session_state:
    try:
        st.session_state["config"] = OmegaConf.load(".streamlit/app_config.yaml")
    except Exception as e:
        st.error(f"Config load error: {str(e)}")
        st.stop()
if "vgg_guidance_cfg" not in st.session_state:
    st.session_state["vgg_guidance_cfg"] = OmegaConf.load('.streamlit/vgg_style.yaml')
if "attnv_guidance_cfg" not in st.session_state:
    st.session_state["attnv_guidance_cfg"] = OmegaConf.load('.streamlit/attnmap_v_style.yaml')

# Load diffusers backend
# if 'diffusion_scheduler' not in st.session_state:
#     st.session_state['diffusion_scheduler'] = get_scheduler(config['scheduler_name'])
# if 'dissuion_model' not in st.session_state:
#     st.session_state["diffusion_model"] = get_model(
#         st.session_state['diffusion_scheduler'], config['model_name'],
#         config['device']
#     )

# Media
if 'content_image_path' not in st.session_state:
    st.session_state["content_image_path"] = None
if 'style_image_path' not in st.session_state:
    st.session_state["style_image_path"] = None
if not os.path.exists(st.session_state["config"]['tmp']):
    os.makedirs(st.session_state["config"]['tmp'])
if 'res_img' not in st.session_state:
    st.session_state["res_img"] = None

# Other states
if 'exp_configs' not in st.session_state:
    st.session_state["exp_configs"] = {}
if 'guidance_editor' not in st.session_state:
    st.session_state["guidance_editor"] = None

with st.sidebar:
    st.write("# Select Guidance's configuration")
    guider_type = st.selectbox(
        "Configuration", options=['VGG', 'Self-Attention-V']
    )
    if guider_type == 'VGG':
        vgg_config_input()
    elif guider_type == 'Self-Attention-V':
        attn_v_config_input()
    else:
        raise NotImplementedError
    st.button("Init guidance editor", on_click=click_init_editor)

cnt_img_panel, sty_img_panel = st.columns(2)
print(st.session_state["content_image_path"])
print(st.session_state["style_image_path"])
with cnt_img_panel:
    st.text('Content image')
    st.session_state["content_image_path"] = upload_image(
        label='Choose a content image file', save_to=os.path.join(st.session_state["config"]["tmp"], 'content.png')
    )
    if st.session_state["content_image_path"] is not None:
        st.image(Image.fromarray(load_512(st.session_state["content_image_path"])))
with sty_img_panel:
    st.text('Style image')
    st.session_state["style_image_path"] = upload_image(
        label='Choose a stlye image file', save_to=os.path.join(st.session_state["config"]["tmp"], 'style.png')
    )
    if st.session_state["style_image_path"] is not None:
        st.image(Image.fromarray(load_512(st.session_state["style_image_path"])))
if st.session_state["style_image_path"] is not None and st.session_state["content_image_path"] is not None:
    st.button("Apply stylization", on_click=click_generate)
if st.session_state["res_img"] is not None:
    st.image(Image.fromarray(st.session_state["res_img"]))
