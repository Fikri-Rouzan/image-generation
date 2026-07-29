import streamlit as st
import torch
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import logic
from PIL import Image, ImageDraw, ImageOps, ImageFilter

# Config
st.set_page_config(page_title="StudioAI", layout="wide", page_icon="🎨")

# Load Models
@st.cache_resource
def get_models():
    return logic.load_models_cached()

try:
    pipe_txt2img, pipe_inpaint = get_models()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

st.title("🎨 StudioAI: Craeating Amazing Paint with Stable Diffusion")

with st.sidebar:
    st.header("⚙️ Parameters")
    # Basic
    steps = st.slider("Quality Steps", 15, 50, 30)
    cfg = st.slider("Creativity (CFG)", 1.0, 20.0, 7.5)
    seed = st.number_input("Seed Control", value=42)

    st.divider()

    # Skilled
    st.subheader("🚀 Advanced")
    scheduler_name = st.selectbox("Scheduler", ["Euler A", "DPM++", "DDIM"])
    num_images = st.slider("Batch Size (Jumlah Gambar)", 1, 4, 1)

    st.divider()
    if st.button("🧹 Flush RAM"):
        logic.flush_memory()
        st.toast("Memory Cleared!")

# Tab (fitur) yang tersedia
tab_gen, tab_edit = st.tabs(["✨ GENERATE", "🛠️ EDIT"])

# Tab Generate
with tab_gen:
    c1, c2 = st.columns([1, 1], gap="large")

    # Input
    with c1:
        st.subheader("Input Blueprint")
        with st.form(key="gen_form"):
            prompt = st.text_area("Prompt", "a cute robot in a futuristic city, 8k, masterpiece", height=150)
            neg_prompt = st.text_input("Negative Prompt", "blurry, bad anatomy, worst quality")

            submit_gen = st.form_submit_button("🚀 Initialize Generation", type="primary")

        if submit_gen:
            with st.spinner("Processing Image"):
                logic.flush_memory()

                generated_list = logic.generate_image(
                    pipe_txt2img, prompt, neg_prompt, seed, steps, cfg, num_images, scheduler_name
                )

                st.session_state['generated_images'] = generated_list

                # Ini mencegah error "List has no attribute .size" di Tab Edit.
                if generated_list:
                    st.session_state['current_image'] = generated_list[0]

            # Refresh halaman untuk update tampilan
            st.rerun()

    # Output
    with c2:
        st.subheader("Visual Output")

        if 'generated_images' in st.session_state:
            imgs = st.session_state['generated_images']

            # Batch image: ada lebih dari 1 gambar
            if len(imgs) > 1:
                cols = st.columns(2) # Grid 2 Kolom
                for idx, img in enumerate(imgs):
                    with cols[idx % 2]:
                        st.image(img, caption=f"Img {idx+1}", use_container_width=True)

                        # Tombol Pilih Gambar untuk diedit
                        if st.button(f"Select Img {idx+1}", key=f"sel_{idx}"):
                             st.session_state['current_image'] = img
                             st.toast(f"✅ Image {idx+1} Selected for Editing!")

            # Single Image
            elif len(imgs) == 1:
                st.image(imgs[0], caption="Result", use_container_width=True)

        else:
            st.info("👈 Masukkan prompt di panel kiri dan tekan Generate.")


# Tab Edit (Inpainting dan Outpainting)
with tab_edit:
    if 'current_image' in st.session_state:
        source_img = st.session_state['current_image']

        # Validasi tipe data
        if isinstance(source_img, list):
            st.warning("⚠️ Terdeteksi List Gambar. Mengambil gambar pertama secara otomatis.")
            source_img = source_img[0]
            st.session_state['current_image'] = source_img

        W, H = source_img.size

        # Pilihan Mode
        mode = st.radio("Select Mode:", ["Inpainting (Edit Objek)", "Outpainting (Zoom Out)"], horizontal=True)
        st.divider()

        # Mode Inpainting
        if mode == "Inpainting (Edit Objek)":
            col_tools, col_result = st.columns([1, 1], gap="large")

            # Logic Reset Canvas: Agar coretan hilang setelah generate
            if 'canvas_key' not in st.session_state:
                st.session_state['canvas_key'] = 0

            img_id = str(id(source_img))

            dynamic_key = f"canvas_{st.session_state['canvas_key']}_{img_id}"

            with col_tools:
                st.subheader("✍️ Draw Mask")
                st.caption("Warnai area yang ingin diubah.")

                # Widget Canvas
                canvas_result = st_canvas(
                    fill_color="rgba(255, 255, 255, 1.0)", # Warna Kuas Putih
                    stroke_width=20,
                    stroke_color="#FFFFFF",
                    background_image=source_img,
                    update_streamlit=True,
                    height=H, width=W,
                    drawing_mode="freedraw",
                    key=dynamic_key
                )

            with col_result:
                st.subheader("Settings")

                # Form Input
                with st.form("inpaint_input"):
                    edit_prompt = st.text_input("Prompt Baru (Ganti jadi apa?)", "a pair of sunglasses")
                    strength = st.slider("Strength (Seberapa kuat perubahannya?)", 0.1, 1.0, 0.85)
                    submit_inpaint = st.form_submit_button("⚡ Execute Inpainting", type="primary")

                # Logic Eksekusi
                if submit_inpaint:
                    if canvas_result.image_data is not None and np.max(canvas_result.image_data) > 0:

                        with st.spinner("Processing Inpainting..."):
                            logic.flush_memory()

                            # Proses Masker
                            # Ambil Alpha Channel
                            mask_data = canvas_result.image_data[:, :, 3]

                            # Ubah abu-abu jadi putih mutlak
                            mask_data[mask_data > 0] = 255
                            mask_image = Image.fromarray(mask_data.astype('uint8'), mode='L')

                            # Samakan ukuran mask dengan gambar asli (PENTING!)
                            if mask_image.size != source_img.size:
                                mask_image = mask_image.resize(source_img.size, resample=Image.NEAREST)

                            # Menebalkan atau mempertegas Masker
                            mask_image = mask_image.filter(ImageFilter.MaxFilter(15))

                            # Tampilkan Masker yang akan dilihat model
                            with st.expander("🕵️ Lihat Masker Final (Debug)"):
                                st.image(mask_image, caption="Masker Tajam (Tanpa Blur)", width=200)

                            try:
                                result_img = logic.run_inpainting(
                                    pipe_inpaint, source_img, mask_image, edit_prompt, strength
                                )

                                st.session_state['current_image'] = result_img
                                st.session_state['canvas_key'] = str(int(st.session_state.get('canvas_key', 0)) + 1)
                                st.success("Inpainting Selesai!")
                                st.rerun()

                            except Exception as e:
                                st.error(f"Error pada logic: {e}")
                    else:
                        st.warning("⚠️ Silakan coret gambar terlebih dahulu!")

        # Mode Outpainting
        elif mode == "Outpainting (Zoom Out)":
            c_out_1, c_out_2 = st.columns([1, 1], gap="large")

            with c_out_1:
                st.subheader("Original")
                st.image(source_img, caption="Gambar saat ini", use_container_width=True)

            with c_out_2:
                st.subheader("Expansion")
                with st.form("outpaint_input"):
                    st.info("Gambar akan diperluas 128px ke segala arah.")
                    out_prompt = st.text_input(
                        "Prompt Deskriptif (Jelaskan gambar UTUH)",
                        "wide angle view of [masukkan prompt awal], detailed background, 8k"
                    )
                    submit_outpaint = st.form_submit_button("🔍 Zoom Out (Expand)", type="primary")

                if submit_outpaint:
                    with st.spinner("Expanding Canvas..."):
                        logic.flush_memory()
                        try:
                            # Siapkan Canvas & Mask (Logic dari kode siswa di atas)
                            canvas_ready, mask_ready = logic.prepare_outpainting(source_img)

                            # Jalankan Inpainting pada area kosong
                            final_result = logic.run_inpainting(
                                pipe_inpaint, canvas_ready, mask_ready, out_prompt, 1.0
                            )
                            st.session_state['current_image'] = final_result
                            st.rerun()

                        except Exception as e:
                            st.error(f"Error pada logic Outpainting: {e}")
                            st.caption("Pastikan fungsi prepare_outpainting di logic.py sudah benar.")

    else:
        # Tampilan jika belum ada gambar sama sekali
        st.info("👈 Belum ada gambar. Silakan ke Tab 'GENERATE' dan buat gambar dulu.")
