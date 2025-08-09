import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import cv2
from sklearn.cluster import KMeans
from scipy.ndimage import distance_transform_edt, gaussian_filter
import io
import imageio
import random


# Streamlit page config / Branding

st.set_page_config(page_title="Sand-Which?", page_icon="🥪", layout="centered")

st.markdown(
    "<h1 style='text-align:center; color:#d2691e; margin-bottom:0'>🥪 Sand-Which?</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center; color:#0096FF; margin-top:4px'>Why waste time thinking about the calories when you can waste time thinking about the next bite...</p><br>",
    unsafe_allow_html=True
)


# Helper functions (vision + scoring)

def load_image(uploaded_file):
    """Load uploaded image in OpenCV BGR format with 3 channels."""
    image = Image.open(uploaded_file).convert("RGB")
    bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return bgr

def pil_from_bgr(bgr):
    """Convert BGR OpenCV image to PIL RGB."""
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

def show_image_pil(img_pil, caption=None):
    """Display a PIL image in Streamlit."""
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    st.image(buf, caption=caption, use_column_width=True)

def get_largest_contour_mask(bgr, min_area=5000):
    """Find the largest contour mask from the sandwich image."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.ones(gray.shape, dtype=np.uint8) * 255 
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area:
        h, w = gray.shape
        mask = np.zeros_like(gray)
        cx, cy = w // 2, h // 2
        rw, rh = w // 3, h // 3
        mask[cy-rh:cy+rh, cx-rw:cx+rw] = 255
        return mask
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [c], -1, 255, -1)
    mask = cv2.dilate(mask, np.ones((15, 15), np.uint8), iterations=1)
    return mask

def refine_with_grabcut(bgr, init_mask):
    """Refine segmentation mask using GrabCut."""
    mask_gc = np.where(init_mask == 255, cv2.GC_PR_FGD, cv2.GC_BGD).astype('uint8')
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(bgr, mask_gc, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
        final_mask = np.where(
            (mask_gc == cv2.GC_FGD) | (mask_gc == cv2.GC_PR_FGD),
            255, 0
        ).astype('uint8')
    except Exception:
        final_mask = init_mask
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return final_mask

def color_kmeans_clusters(bgr, mask, k=3):
    """Segment filling vs bread by color clustering."""
    h, w, _ = bgr.shape
    mask_bool = mask > 0
    pixels = bgr[mask_bool].reshape(-1, 3).astype(np.float32)
    if len(pixels) < k:
        k = max(1, len(pixels))
    if len(pixels) == 0:
        labels = -np.ones((h, w), dtype=np.int32)
        centers = np.array([[0, 0, 0]], dtype=np.uint8)
        return labels, centers
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=4).fit(pixels)
    labels = -np.ones((h, w), dtype=np.int32)
    labels[mask_bool] = kmeans.labels_
    centers = kmeans.cluster_centers_.astype(np.uint8)
    return labels, centers

def compute_fill_and_texture(bgr, mask):
    """Compute filling density map and texture map."""
    labels, centers = color_kmeans_clusters(bgr, mask, k=3)
    centers = np.uint8(centers)
    if centers.ndim == 1:
        centers = np.tile(centers[None, :], (1, 1))
    if centers.shape[0] == 0:
        centers = np.array([[0,0,0]], dtype=np.uint8)
    try:
        centers_hsv = cv2.cvtColor(centers.reshape(1, -1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
        Vs = centers_hsv[:, 2].astype(float)
        bread_cluster = int(np.argmax(Vs))
    except Exception:
        bread_cluster = 0
    fill_map = np.zeros(mask.shape, dtype=float)
    fill_map[(labels != -1) & (labels != bread_cluster)] = 1.0
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)
    mag *= (mask > 0).astype(float)
    fill_map = gaussian_filter(fill_map, sigma=8)
    mag = gaussian_filter(mag, sigma=4)
    return fill_map, mag

def compute_stability(mask):
    """Compute stability map (distance from edge)."""
    dist = distance_transform_edt(mask > 0)
    if dist.max() > 0:
        dist = dist / dist.max()
    return dist

def make_heatmap(fill_map, texture_map, stability_map, weights=(0.5, 0.2, 0.3)):
    """Combine feature maps into a bite-worthiness heatmap."""
    heat = (weights[0] * fill_map +
            weights[1] * texture_map +
            weights[2] * stability_map)
    heat = np.clip(heat, 0, None)
    heat = gaussian_filter(heat, sigma=6)
    if heat.max() > 0:
        heat = heat / heat.max()
    return heat

def penalize_hand_region(bgr, heat, skin_penalty_strength=0.75, blur_kernel=51):
    """Detect likely skin regions and reduce heat there."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 60], dtype=np.uint8)
    upper_skin = np.array([25, 200, 255], dtype=np.uint8)
    mask_skin = cv2.inRange(hsv, lower_skin, upper_skin)
    lower_skin2 = np.array([160, 20, 60], dtype=np.uint8)
    upper_skin2 = np.array([179, 200, 255], dtype=np.uint8)
    mask_skin2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
    mask_skin = cv2.bitwise_or(mask_skin, mask_skin2)
    mask_skin = cv2.GaussianBlur(mask_skin.astype(np.float32), (blur_kernel, blur_kernel), 0) / 255.0
    if heat.shape != mask_skin.shape:
        mask_skin = cv2.resize(mask_skin, (heat.shape[1], heat.shape[0]))
    adjusted = heat * (1.0 - skin_penalty_strength * mask_skin)
    return adjusted

def boost_protrusions(mask, heat, erosion_radius=25, boost_strength=0.6):
    """Boost bite-worthiness on protruding parts of the sandwich contour."""
    if mask.dtype != np.uint8:
        mask_u8 = (mask > 0).astype(np.uint8) * 255
    else:
        mask_u8 = mask.copy()
    k = max(3, int(erosion_radius))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    eroded = cv2.erode(mask_u8, kernel, iterations=1)
    rim = ((mask_u8 > 0) & (eroded == 0)).astype(np.uint8)
    if rim.sum() == 0:
        return heat
    dist = distance_transform_edt(mask_u8 > 0).astype(float)
    dist_norm = dist / (dist.max() + 1e-8)
    protrusion_strength = rim.astype(float) * dist_norm
    protrusion_boost = gaussian_filter(protrusion_strength, sigma=8)
    if protrusion_boost.max() > 0:
        protrusion_boost = protrusion_boost / protrusion_boost.max()
    if heat.shape != protrusion_boost.shape:
        protrusion_boost = cv2.resize(protrusion_boost, (heat.shape[1], heat.shape[0]))
    adjusted = heat + boost_strength * protrusion_boost
    return adjusted

def overlay_heat_on_image(bgr, heat, best_pt=None, alpha=0.6):
    """Overlay heatmap and optional bite point on image."""
    heat_norm = heat.copy()
    if heat_norm.max() > 0:
        heat_norm = heat_norm / heat_norm.max()
    heat_uint8 = np.uint8(255 * np.clip(heat_norm, 0, 1))
    heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGBA)
    base = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA)
    if (base.shape[0], base.shape[1]) != (heat_color.shape[0], heat_color.shape[1]):
        heat_color = cv2.resize(heat_color, (base.shape[1], base.shape[0]))
    overlay = (base.astype(float) * (1 - alpha) + heat_color.astype(float) * alpha).astype(np.uint8)
    pil = Image.fromarray(overlay)
    if best_pt is not None:
        draw = ImageDraw.Draw(pil)
        x, y = int(best_pt[0]), int(best_pt[1])
        r = max(12, min(pil.size) // 30)
        draw.ellipse((x-r, y-r, x+r, y+r), outline=(255, 255, 255, 255), width=3)
        draw.line((x-r, y, x+r, y), fill=(255, 255, 255, 255), width=2)
        draw.line((x, y-r, x, y+r), fill=(255, 255, 255, 255), width=2)
    return pil

def find_best_point_from_heat(heat, mask, min_distance_from_eaten=6):
    """Find the best bite point from the heatmap with a safety margin from eaten edges."""
    heat_masked = heat * (mask > 0)
    if heat_masked.max() == 0:
        h, w = heat.shape
        return (w // 2, h // 2)
    dist_to_bg = distance_transform_edt(mask > 0)
    safe_mask = (dist_to_bg >= min_distance_from_eaten).astype(float)
    combined = heat_masked * safe_mask
    if combined.max() == 0:
        combined = heat_masked
    y, x = np.unravel_index(np.argmax(combined), combined.shape)
    return (x, y)

def make_pulse_frames(pil_image, center, radius=20, frames=8):
    """Create small pulsing frames highlighting center. Returns list of PIL images."""
    frames_list = []
    w, h = pil_image.size
    for i in range(frames):
        factor = 1.0 + 0.15 * np.sin(2 * np.pi * i / frames)  # pulsing
        r = int(radius * (1.0 + 0.25 * np.sin(2 * np.pi * i / frames)))
        frame = pil_image.copy()
        draw = ImageDraw.Draw(frame)
        # outer pulsing circle
        draw.ellipse([center[0]-r-6, center[1]-r-6, center[0]+r+6, center[1]+r+6], outline=(255, 50, 50, 150), width=6)
        # inner crosshair
        draw.ellipse([center[0]-8, center[1]-8, center[0]+8, center[1]+8], outline=(255,255,255,255), width=3)
        frames_list.append(frame)
    return frames_list

def make_zoom_crop(pil_img, center, crop_size=160, scale=2):
    """Return a zoomed crop centered on 'center' as a PIL image."""
    w, h = pil_img.size
    cx, cy = int(center[0]), int(center[1])
    half = int(crop_size // (2 * scale))
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(w, cx + half)
    y2 = min(h, cy + half)
    crop = pil_img.crop((x1, y1, x2, y2))
    zoomed = crop.resize((crop_size, crop_size), resample=Image.Resampling.LANCZOS)
    # draw a small crosshair
    draw = ImageDraw.Draw(zoomed)
    draw.line((crop_size//2 - 12, crop_size//2, crop_size//2 + 12, crop_size//2), fill=(255,255,255,255), width=2)
    draw.line((crop_size//2, crop_size//2 - 12, crop_size//2, crop_size//2 + 12), fill=(255,255,255,255), width=2)
    return zoomed


# Streamlit UI

uploaded = st.file_uploader("Upload sandwich photo (jpg, png) 🍞", type=["jpg", "jpeg", "png"])
risk_mode = st.checkbox("Risky Bite Mode (less safety)", value=False)

if uploaded:
    bgr = load_image(uploaded)
    h, w, _ = bgr.shape
    original_pil = pil_from_bgr(bgr)
    st.write("")

    # Show original on top
    st.markdown("**Original**")
    st.image(original_pil, use_container_width=True)

    if st.button("Find next bite 🍽️"):
        with st.spinner("Analyzing sandwich..."):
            init_mask = get_largest_contour_mask(bgr)
            mask = refine_with_grabcut(bgr, init_mask)
            fill_map, texture_map = compute_fill_and_texture(bgr, mask)
            stability_map = compute_stability(mask)

            weights = (0.6, 0.25, 0.15) if risk_mode else (0.5, 0.2, 0.3)
            heat = make_heatmap(fill_map, texture_map, stability_map, weights=weights)

            # Penalize hand
            heat = penalize_hand_region(bgr, heat, skin_penalty_strength=0.75)

            # Boost protrusions
            heat = boost_protrusions(mask, heat, erosion_radius=max(7, min(h, w)//20), boost_strength=0.7)

           
            if heat.max() > 0:
                heat = heat / heat.max()

            # Safety margin
            min_dist = 4 if risk_mode else 8
            best_pt = find_best_point_from_heat(heat, mask, min_distance_from_eaten=min_dist)

            # Create overlay and zoom crop
            overlay_pil = overlay_heat_on_image(bgr, heat, best_pt)
            zoom_pil = make_zoom_crop(overlay_pil, best_pt, crop_size=180, scale=2)

            
            pulse_bytes = None
            try:
                frames = make_pulse_frames(overlay_pil, best_pt, radius=max(18, min(w,h)//30), frames=8)
                buf = io.BytesIO()
                frames_rgb = [np.array(f.convert("RGBA")) for f in frames]
                imageio.mimsave(buf, frames_rgb, format='GIF', fps=8)
                buf.seek(0)
                pulse_bytes = buf.read()
            except Exception:
                pulse_bytes = None

            # Layout: before/after, and zoom / GIF
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown("**Before**")
                st.image(original_pil, caption="Original Sandwich", use_container_width=True)
            with col2:
                st.markdown("**After — Suggested Bite**")
                if pulse_bytes:
                    st.image(pulse_bytes)

                else:
                    st.image(overlay_pil, caption="Next Bite Suggestion", use_container_width=True)

           

            # Bite efficiency score (fun)
            score = int(np.clip((heat[int(best_pt[1]), int(best_pt[0])] * 100) + random.randint(0, 8), 0, 100))
            st.markdown(f"**Bite Efficiency:** {score} / 100 🍽️")
            st.progress(score / 100)

            # Fun messages
            messages = [
                "Mmmm... that’s the spot! 😋",
                "Go for the gold! 🏆",
                "Perfect bite incoming... 🚀",
                "Crunch time! 🥪",
                "Steady hand — take it slow! ✋" if not risk_mode else "Bold move — I like it! 🔥"
            ]
            st.markdown(f"<p style='font-size:18px; color:#ff4500;'>{random.choice(messages)}</p>", unsafe_allow_html=True)

            # Explain reasoning
            st.markdown("**Why this bite?**")
            reasons = [
                "- High filling density in this region.",
                "- Located away from the sandwich edge (stable)." if not risk_mode else "- Risky: more filling, but closer to edge.",
                "- Texture indicates interesting mix of ingredients.",
                "- Protrusion detected (biteable bump) and hand area penalized so finger wasn't chosen."
            ]
            st.write("\n".join(reasons))

            # Optional debug info
            with st.expander("Debug: show intermediate masks and heatmap"):
                try:
                    mask_vis = Image.fromarray(cv2.cvtColor(np.stack([mask]*3, axis=-1), cv2.COLOR_BGR2RGB))
                    st.write("Segmentation mask")
                    st.image(mask_vis, use_container_width=True)
                except Exception:
                    st.write("Mask not available")
                try:
                    heat_vis = (255 * (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)).astype(np.uint8)
                    heat_color = cv2.applyColorMap(heat_vis, cv2.COLORMAP_JET)
                    heat_pil = Image.fromarray(cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB))
                    st.write("Heatmap")
                    st.image(heat_pil, use_container_width=True)
                except Exception:
                    st.write("Heatmap not available")

    if st.button("I feel adventurous — random bite"):
        rx, ry = np.random.randint(0, w), np.random.randint(0, h)
        blank_heat = np.zeros((h, w), dtype=float)
        rand_img = overlay_heat_on_image(bgr, blank_heat, (rx, ry))
        show_image_pil(rand_img, caption=f"Random bite at {(rx, ry)}")
else:
    st.info("Upload an image to begin. Tip: take the photo from above at ~45° showing the eaten part clearly.")
