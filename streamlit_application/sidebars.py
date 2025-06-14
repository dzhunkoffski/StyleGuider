import copy

import streamlit as st
from omegaconf import OmegaConf

def vgg_config_input():
    st.session_state["guidance_cfg"] = copy.deepcopy(st.session_state["vgg_guidance_cfg"])
    st.write("## Content Guidance")
    cnt_g_start, cnt_g_end, cnt_g_scale = st.columns(3)
    with cnt_g_start:
        st.session_state["exp_configs"]["content_guider_start"] = st.number_input(
            "content guidance start",
            min_value=0, max_value=50, value=0, step=1,
            help="Number of iteration from which guidance on the content starts"
        )
    with cnt_g_end:
        st.session_state["exp_configs"]["content_guider_end"] = st.number_input(
            "content guidance end",
            min_value=0, max_value=50, value=50, step=1,
            help="Number of iteration where guidance on the content ends"
        )
        if st.session_state["exp_configs"]["content_guider_end"] < st.session_state["exp_configs"]["content_guider_start"]:
            st.warning("You choose end iteration for content guidance less then start iteration. Type propper value.")
    with cnt_g_scale:
        st.session_state["exp_configs"]["content_guider_scale"] = st.number_input(
            "content guidance scale",
            min_value=0.0, value=100000.0,
            help="Scale for guidance opn the content"
        )

    st.write("## Style Guidance")
    sty_g_start, sty_g_end, sty_g_style = st.columns(3)
    with sty_g_start:
        st.session_state["exp_configs"]["style_guider_start"] = st.number_input(
            'style guidance start',
            min_value=0, max_value=50, value=30, step=1,
            help="Number of iteration where guidance on the style starts"
        )
    with sty_g_end:
        st.session_state["exp_configs"]["style_guider_end"] = st.number_input(
            "style guidance end",
            min_value=0, max_value=50, value=50, step=1,
            help="Number of iteration where guidance on the style ends"
        )
        if st.session_state["exp_configs"]["style_guider_start"] > st.session_state["exp_configs"]["style_guider_end"]:
            st.warning("You choose end iteration for style guidance less then start iteration. Type propper value.")
    with sty_g_style:
        st.session_state["exp_configs"]["style_guider_scale"] = st.number_input(
            "style guidance scale",
            min_value=0.0, value=500000.0
        )
    st.session_state["guidance_cfg"]["guiders"][2]["kwargs"]["style_layers"] = st.multiselect(
        "Layers which outputs will be used for style guidance",
        [f'conv_{i}' for i in range(10)], default=["conv_1", "conv_2", "conv_3"],
        help="Select convolution layers  of the VGG that will be used for style guidance"
    )

    st.write("## Others configurations")
    st.session_state["exp_configs"]["do_others_rescaling"] = st.checkbox(
        "rescale guidance noise",
        help="Whether to rescale guidance noise respectfuly to unet predicted noise"
    )
    # if st.session_state["exp_configs"]["do_others_rescaling"]:
    noise_res_start, noise_res_end, noise_res_scale = st.columns(3)
    with noise_res_start:
        st.session_state["exp_configs"]["others_rescaling_iter_start"] = st.number_input(
            "rescale guidance noise start",
            min_value=0, max_value=50, value=30, step=1,
            help="Number of iteration where guidance noise rescale starts"
        )
    with noise_res_end:
        st.session_state["exp_configs"]["others_rescaling_iter_end"] = st.number_input(
            "rescale guidance noise end",
            min_value=0, max_value=50, value=50, step=1,
            help="Number of iteration where guidance noise rescale ends"
        )
    with noise_res_scale:
        st.session_state["exp_configs"]["others_rescaling_factor"] = st.number_input(
            "rescale guidance noise coeff",
            min_value=0.0, value=2.0,
            help="Multiplier - how many times is the norm of guidance noise is greater then norm of the unet noise. Use this to stabilize guidance"
        )
    
    st.write("## AdaIN")
    adain_start, adain_end = st.columns(2)
    with adain_start:
        st.session_state["guidance_cfg"]["adain_start_ix"] = st.number_input(
            "AdaIN start index", min_value=0, max_value=50, value=0, step=1,
            help="Number of iteration where generated latent start to being normalized with AdaIN and style latent"
        )
    with adain_end:
        st.session_state["guidance_cfg"]["adain_end_ix"] = st.number_input(
            "AdaIN end index", min_value=0, max_value=50, value=0, step=1,
            help="Number of iteration where generated latent end to being normalized with AdaIN and style latent"
        )
        if st.session_state["guidance_cfg"]["adain_start_ix"] > st.session_state["guidance_cfg"]["adain_end_ix"]:
            st.warning("You choose end iteration for AdaIN less then start iteration. Type propper value.")

    for guiding_ix in range(st.session_state["exp_configs"]["content_guider_start"], st.session_state["exp_configs"]["content_guider_end"]):
        st.session_state["guidance_cfg"]["guiders"][1]["g_scale"][guiding_ix] = st.session_state["exp_configs"]["content_guider_scale"]
    for guiding_ix in range(st.session_state["exp_configs"]["style_guider_start"], st.session_state["exp_configs"]["style_guider_end"]):
        st.session_state["guidance_cfg"]["guiders"][2]["g_scale"][guiding_ix] = st.session_state["exp_configs"]["style_guider_scale"]

def attn_v_config_input():
    st.session_state["guidance_cfg"] = copy.deepcopy(st.session_state["attnv_guidance_cfg"])
    st.session_state["exp_configs"]["qkv_guider_start"] = 0
    st.session_state["exp_configs"]["qkv_guider_end"] = 50
    st.session_state["exp_configs"]["qkv_guider_scale"] = 1.0

    st.write("## Content Guidance")
    cnt_iter_start, cnt_iter_end, cnt_scale = st.columns(3)
    print(st.session_state["guidance_cfg"]["guiders"][1]['kwargs'])
    with cnt_iter_start:
        st.session_state["guidance_cfg"]["guiders"][1]["kwargs"]["attn_map_iter_start"] = st.number_input(
            "content guidance start",
            min_value=0, max_value=50, value=0, step=1,
            help="Number of iteration from which guidance on the content starts"
        )
    with cnt_iter_end:
        st.session_state["guidance_cfg"]["guiders"][1]["kwargs"]["attn_map_iter_end"] = st.number_input(
            "content guidance end",
            min_value=0, max_value=50, value=50, step=1,
            help="Number of iteration where guidance on the content ends"
        )
        if st.session_state["guidance_cfg"]["guiders"][1]["kwargs"]["attn_map_iter_start"] > st.session_state["guidance_cfg"]["guiders"][1]["kwargs"]["attn_map_iter_end"]:
            st.warning("You choose end iteration for content guidance less then start iteration. Type propper value.")
    with cnt_scale:
        st.session_state["guidance_cfg"]["guiders"][1]["kwargs"]["attn_map_scale"] = st.number_input(
            "content guidance scale",
            min_value=0.0, value=3000000.0,
            help="Scale for guidance opn the content"
        )
    ########################################
    st.write("## Style Guidance")
    sty_iter_start, sty_iter_end, sty_scale = st.columns(3)
    with sty_iter_start:
        st.session_state["guidance_cfg"]["guiders"][1]["kwargs"]["v_iter_start"] = st.number_input(
            'style guidance start',
            min_value=0, max_value=50, value=30, step=1,
            help="Number of iteration where guidance on the style starts"
        )
    with sty_iter_end:
        st.session_state["guidance_cfg"]["guiders"][1]["kwargs"]["v_iter_end"] = st.number_input(
            "style guidance end",
            min_value=0, max_value=50, value=50, step=1,
            help="Number of iteration where guidance on the style ends"
        )
        if st.session_state["guidance_cfg"]["guiders"][1]["kwargs"]["v_iter_start"] > st.session_state["guidance_cfg"]["guiders"][1]["kwargs"]["v_iter_end"]:
            st.warning("You choose end iteration for style guidance less then start iteration. Type propper value.")
    with sty_scale:
        st.session_state["guidance_cfg"]["guiders"][1]["kwargs"]["v_scale"] = st.number_input(
            "style guidance scale",
            min_value=0.0, value=10000.0
        )
    ########################################
    st.write("## Others configurations")
    st.session_state["exp_configs"]["do_others_rescaling"] = st.checkbox(
        "rescale guidance noise",
        help="Whether to rescale guidance noise respectfuly to unet predicted noise"
    )
    # if st.session_state["exp_configs"]["do_others_rescaling"]:
    noise_res_start, noise_res_end, noise_res_scale = st.columns(3)
    with noise_res_start:
        st.session_state["exp_configs"]["others_rescaling_iter_start"] = st.number_input(
            "rescale guidance noise start",
            min_value=0, max_value=50, value=30, step=1,
            help="Number of iteration where guidance noise rescale starts"
        )
    with noise_res_end:
        st.session_state["exp_configs"]["others_rescaling_iter_end"] = st.number_input(
            "rescale guidance noise end",
            min_value=0, max_value=50, value=50, step=1,
            help="Number of iteration where guidance noise rescale ends"
        )
    with noise_res_scale:
        st.session_state["exp_configs"]["others_rescaling_factor"] = st.number_input(
            "rescale guidance noise coeff",
            min_value=0.0, value=2.0,
            help="Multiplier - how many times is the norm of guidance noise is greater then norm of the unet noise. Use this to stabilize guidance"
        )
    ########################################
    st.write("## AdaIN")
    adain_start, adain_end = st.columns(2)
    with adain_start:
        st.session_state["guidance_cfg"]["adain_start_ix"] = st.number_input(
            "AdaIN start index", min_value=0, max_value=50, value=30, step=1,
            help="Number of iteration where generated latent start to being normalized with AdaIN and style latent"
        )
    with adain_end:
        st.session_state["guidance_cfg"]["adain_end_ix"] = st.number_input(
            "AdaIN end index", min_value=0, max_value=50, value=50, step=1,
            help="Number of iteration where generated latent end to being normalized with AdaIN and style latent"
        )
        if st.session_state["guidance_cfg"]["adain_start_ix"] > st.session_state["guidance_cfg"]["adain_end_ix"]:
            st.warning("You choose end iteration for AdaIN less then start iteration. Type propper value.")

    for guiding_ix in range(st.session_state["exp_configs"]["qkv_guider_start"], st.session_state["exp_configs"]["qkv_guider_end"]):
        st.session_state["guidance_cfg"]["guiders"][1]['g_scale'][guiding_ix] = st.session_state["exp_configs"]["qkv_guider_scale"]