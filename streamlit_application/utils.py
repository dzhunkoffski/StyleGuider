import os
from PIL import Image

from diffusion_core.guiders import GuidanceEditing

import streamlit as st

from diffusion_core import diffusion_models_registry, diffusion_schedulers_registry
from diffusion_core.utils import load_512

def upload_image(label: str, save_to: str) -> str:
    try:
        uploaded_file = st.file_uploader(label, type=["jpg", "jpeg", "png"])
        img = Image.open(uploaded_file)
        img.save(save_to)
        return save_to
    except Exception as e:
        return None

def get_scheduler(scheduler_name: str):
    if scheduler_name not in diffusion_schedulers_registry:
        raise ValueError(f"Incorrect scheduler type: {scheduler_name}, possible are {diffusion_schedulers_registry}")
    scheduler = diffusion_schedulers_registry[scheduler_name]()
    return scheduler

def get_model(scheduler, model_name, device):
    model = diffusion_models_registry[model_name](scheduler)
    model.to(device)
    return model

def click_reset_content():
    os.remove(st.session_state['content_image_path'])
    st.session_state['content_image_path'] = None

def click_reset_style():
    os.remove(st.session_state['style_image_path'])
    st.session_state['style_image_path'] = None

def click_generate():
    if st.session_state["guidance_editor"] is None:
        st.error("Initialize Guidance Editor first !!!")
    else:
        cnt_img = Image.fromarray(load_512(st.session_state["content_image_path"]))
        sty_img = Image.fromarray(load_512(st.session_state["style_image_path"]))
        with st.spinner(text="Generating stylized version, please wait..."):
            res = st.session_state["guidance_editor"].call_stylisation(
                image_gt=cnt_img, inv_prompt="", trg_prompt="",
                control_image=sty_img, inv_control_prompt="", verbose=True
            )
        st.session_state["res_img"] = res

def click_init_editor():
    st.session_state['diffusion_scheduler'] = get_scheduler(st.session_state["config"]['scheduler_name'])
    st.session_state["diffusion_model"] = get_model(
        st.session_state['diffusion_scheduler'], st.session_state["config"]['model_name'],
        st.session_state["config"]['device']
    )
    st.session_state["guidance_editor"] = GuidanceEditing(
        st.session_state["diffusion_model"], config=st.session_state["guidance_cfg"], root_path='.streamlit/tmp',
        do_others_rescaling=st.session_state["exp_configs"]["do_others_rescaling"],
        others_rescaling_iter_start=st.session_state["exp_configs"]["others_rescaling_iter_start"],
        others_rescaling_iter_end=st.session_state["exp_configs"]["others_rescaling_iter_end"]
    )