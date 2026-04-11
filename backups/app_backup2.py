import streamlit as st
import os
import datetime
import json
import pandas as pd
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from deepface import DeepFace

# --- 1. SETUP & CONFIGURATION ---
LOGO_PATH = "images/NautilusLogoDesign.png"
USER_DB_FILE = "users.json"
SCANS_DB_FILE = "scans.json" 
SCANS_DIR = "scanned_images" 
FACE_DB_DIR = "face_db"

os.makedirs(SCANS_DIR, exist_ok=True)
os.makedirs(FACE_DB_DIR, exist_ok=True)

st.set_page_config(
    page_title="Nautilus AI",
    page_icon=LOGO_PATH if os.path.exists(LOGO_PATH) else "⚓",
    layout="wide"
)

# --- 2. DATA MANAGEMENT (PERSISTENCE) ---
def load_json_db(filepath, default_data):
    if not os.path.exists(filepath):
        with open(filepath, "w") as f:
            json.dump(default_data, f)
        return default_data
    else:
        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except:
            return default_data

def save_json_db(filepath, data):
    with open(filepath, "w") as f:
        json.dump(data, f)

# --- USER MANAGEMENT ---
def load_users():
    return load_json_db(USER_DB_FILE, {"admin": "1234"})

def save_new_user(username, password):
    db = load_users()
    db[username] = password
    save_json_db(USER_DB_FILE, db)
    st.session_state.user_db = db

# --- SCAN MANAGEMENT ---
def load_scans():
    return load_json_db(SCANS_DB_FILE, [])

def save_scan_record(record):
    history = load_scans()
    history.append(record)
    save_json_db(SCANS_DB_FILE, history)
    st.session_state.scan_history = history

def save_image_locally(uploaded_file, username):
    """Saves image to a user-specific subfolder."""
    user_folder = os.path.join(SCANS_DIR, username)
    os.makedirs(user_folder, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ext = os.path.splitext(uploaded_file.name)[1] if hasattr(uploaded_file, 'name') else ".jpg"
    filename = f"scan_{timestamp}{ext}"
    filepath = os.path.join(user_folder, filename)
    
    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    return filename, filepath

def delete_scan_by_filename(filename_to_delete):
    full_history = st.session_state.scan_history
    record_to_delete = next((item for item in full_history if item["File Name"] == filename_to_delete), None)
    
    if record_to_delete:
        index = full_history.index(record_to_delete)
        file_path = record_to_delete.get("File Path")
        
        if file_path and os.path.exists(file_path):
            try: os.remove(file_path)
            except Exception as e: st.error(f"Error deleting file: {e}")
        
        full_history.pop(index)
        save_json_db(SCANS_DB_FILE, full_history)
        st.session_state.scan_history = full_history
        st.success("Record and image deleted.")
        st.rerun()

# --- DEEPFACE HELPERS ---
def pil_to_cv2(pil_image):
    """Convert PIL Image to OpenCV BGR numpy array."""
    rgb = np.array(pil_image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr

def cv2_to_pil(cv2_image):
    """Convert OpenCV BGR numpy array to PIL Image."""
    rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def annotate_faces(image_bgr):
    """
    Run DeepFace.analyze() for emotion, age, gender.
    Draw bounding boxes and emotion labels on the image.
    Returns (annotated_bgr, results_list).
    """
    try:
        results = DeepFace.analyze(
            img_path=image_bgr,
            actions=['emotion', 'age', 'gender'],
            enforce_detection=False,
            detector_backend='opencv'
        )
    except Exception as e:
        return image_bgr, [], str(e)

    if not isinstance(results, list):
        results = [results]

    annotated = image_bgr.copy()

    for face in results:
        region = face.get('region', {})
        x = region.get('x', 0)
        y = region.get('y', 0)
        w = region.get('w', 0)
        h = region.get('h', 0)

        emotion = face.get('dominant_emotion', 'N/A')
        age = face.get('age', '?')
        gender = face.get('dominant_gender', '?')

        # Green bounding box
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Label background for readability
        label = f"{emotion}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        
        # Draw filled rectangle behind text
        cv2.rectangle(annotated, 
                      (x, y - text_size[1] - 10), 
                      (x + text_size[0] + 4, y), 
                      (0, 255, 0), -1)
        cv2.putText(annotated, label, (x + 2, y - 6),
                    font, font_scale, (0, 0, 0), thickness)

        # Secondary info below box
        info_label = f"Age:{age} | {gender}"
        cv2.putText(annotated, info_label, (x, y + h + 20),
                    font, 0.55, (0, 255, 0), 1)

    return annotated, results, None

def try_find_face(image_bgr, db_path):
    """
    Try to identify a face against the registered face database.
    Returns a list of matched identity names, or empty list.
    """
    if not os.path.exists(db_path) or not os.listdir(db_path):
        return []

    try:
        dfs = DeepFace.find(
            img_path=image_bgr,
            db_path=db_path,
            enforce_detection=False,
            detector_backend='opencv',
            silent=True
        )
        matches = []
        for df in dfs:
            if not df.empty:
                # Extract person name from path (folder name)
                for identity_path in df['identity'].tolist():
                    parts = identity_path.replace("\\", "/").split("/")
                    # Structure: face_db/<user>/<person_name>/image.jpg
                    for i, part in enumerate(parts):
                        if part == FACE_DB_DIR and i + 2 < len(parts):
                            person_name = parts[i + 2]
                            if person_name not in matches:
                                matches.append(person_name)
                            break
        return matches
    except Exception:
        return []


# --- 3. SESSION STATE ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "current_user" not in st.session_state:
    st.session_state.current_user = None
if "theme" not in st.session_state:
    st.session_state.theme = "light"
if "user_db" not in st.session_state:
    st.session_state.user_db = load_users()
if "scan_history" not in st.session_state:
    st.session_state.scan_history = load_scans()
if "last_image" not in st.session_state:
    st.session_state.last_image = None
if "live_running" not in st.session_state:
    st.session_state.live_running = False


# --- 4. THEME & STYLING ---
BUTTON_COLOR = "#005EB8" 
BUTTON_TEXT_COLOR = "#FFFFFF"

if st.session_state.theme == "dark":
    bg_color = "#121212"
    text_color = "#ffffff"
    input_bg = "#2b2b2b"
    input_text = "#ffffff"
    placeholder_color = "#cccccc"
    dropdown_bg = "#2b2b2b"
    dropdown_text = "#ffffff"
else:
    bg_color = "#F5F7F9"     
    text_color = "#000000"   
    input_bg = "#ffffff"
    input_text = "#000000"
    placeholder_color = "#555555"
    dropdown_bg = "#ffffff"
    dropdown_text = "#000000"

st.markdown(f"""
    <style>
    .stApp {{ background-color: {bg_color}; color: {text_color}; }}
    h1, h2, h3, h4, h5, h6, p, li, .stMarkdown, .stText, label {{ color: {text_color} !important; }}

    input::placeholder {{ color: {placeholder_color} !important; opacity: 1 !important; font-weight: 500; }}
    .stTextInput > div > div > input {{ background-color: {input_bg} !important; color: {input_text} !important; border: 1px solid #ccc; }}
    
    li[role="option"] {{ background-color: {dropdown_bg} !important; color: {dropdown_text} !important; }}
    div[data-baseweb="popover"] > div {{ background-color: {dropdown_bg} !important; }}
    li[role="option"]:hover, li[role="option"][aria-selected="true"] {{ background-color: {BUTTON_COLOR} !important; color: white !important; }}

    .stFormSubmitButton > div > div:last-child {{ display: none !important; }}

    div.stButton > button, 
    button[kind="secondaryFormSubmit"], 
    button[data-testid="baseButton-secondary"],
    [data-testid="stFileUploader"] button {{
        background-color: {BUTTON_COLOR} !important;
        color: {BUTTON_TEXT_COLOR} !important;
        border: 1px solid {BUTTON_COLOR} !important;
        font-weight: bold !important;
    }}

    div.stButton > button:hover, 
    button[kind="secondaryFormSubmit"]:hover,
    button[data-testid="baseButton-secondary"]:hover,
    [data-testid="stFileUploader"] button:hover {{
        background-color: #004a94 !important; 
        color: white !important;
        border-color: #004a94 !important;
    }}

    [data-testid="stCameraInput"] {{ background-color: transparent !important; border: none !important; }}
    [data-testid="stCameraInput"] * {{ color: #FFFFFF !important; fill: #FFFFFF !important; }}

    button[data-baseweb="tab"] {{ color: {text_color} !important; }}
    </style>
    """, unsafe_allow_html=True)

# --- 5. PAGE FUNCTIONS ---

def login_page():
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        if os.path.exists(LOGO_PATH):
            col_a, col_b, col_c = st.columns([1, 2, 1])
            with col_b:
                st.image(LOGO_PATH, use_container_width=True)
        else:
            st.markdown(f"<h1 style='text-align: center;'>⚓ Nautilus</h1>", unsafe_allow_html=True)

        st.markdown(f"<h3 style='text-align: center;'>Facial Recognition System</h3>", unsafe_allow_html=True)
        st.write("") 

        tab1, tab2 = st.tabs(["🔒 LOGIN", "📝 SIGN UP"])

        with tab1:
            with st.form("login_form"):
                username = st.text_input("Login User", placeholder="Enter Username", label_visibility="collapsed")
                password = st.text_input("Login Pass", type="password", placeholder="Enter Password", label_visibility="collapsed")
                st.write("")
                if st.form_submit_button("Login"):
                    st.session_state.user_db = load_users()
                    if username in st.session_state.user_db and st.session_state.user_db[username] == password:
                        st.session_state.logged_in = True
                        st.session_state.current_user = username
                        st.session_state.last_image = None
                        st.session_state.theme = "light" 
                        st.rerun()
                    else:
                        st.error("Incorrect username or password.")

        with tab2:
            with st.form("signup_form"):
                new_user = st.text_input("New User", placeholder="Create Username", label_visibility="collapsed")
                new_pass = st.text_input("New Pass", type="password", placeholder="Create Password", label_visibility="collapsed")
                if st.form_submit_button("Create Account"):
                    if new_user and new_pass:
                        save_new_user(new_user, new_pass)
                        st.success("Account created! Go to the Login tab.")
                    else:
                        st.warning("Please fill in all fields.")

def recognition_page():
    st.title("👤 Face Scan")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Webcam Scan")
        st.write("Capture a face for analysis.")
        st.caption("Please don't forget to give permission to continue.")
        img_file = st.camera_input(label="", label_visibility="collapsed")
        
    with col2:
        st.subheader("Upload File")
        uploaded_file = st.file_uploader("Drag & drop image", type=['jpg', 'png', 'jpeg'])
    
    final_image = img_file if img_file else uploaded_file
    
    if final_image:
        st.success("Image captured successfully.")
        if st.button("Process Image ➡️"):
            st.session_state.last_image = final_image
            st.switch_page(detection_screen)

def detection_page():
    st.title("🔍 Detection & Emotion Analysis")
    
    if st.session_state.last_image:
        # Convert to PIL then to OpenCV
        pil_img = Image.open(st.session_state.last_image)
        cv2_img = pil_to_cv2(pil_img)

        with st.spinner("🧠 Analyzing face(s) with DeepFace..."):
            annotated_img, results, error = annotate_faces(cv2_img)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📸 Annotated Image")
            annotated_pil = cv2_to_pil(annotated_img)
            st.image(annotated_pil, caption="Detected faces with emotions", use_container_width=True)

        with col2:
            if error:
                st.error(f"Analysis error: {error}")
            elif results:
                st.subheader("📊 Analysis Results")
                for i, face in enumerate(results):
                    with st.expander(f"Face #{i+1} — {face.get('dominant_emotion', 'N/A')}", expanded=True):
                        rcol1, rcol2 = st.columns(2)
                        with rcol1:
                            st.metric("Dominant Emotion", face.get('dominant_emotion', 'N/A'))
                            st.metric("Age", face.get('age', '?'))
                        with rcol2:
                            st.metric("Gender", face.get('dominant_gender', '?'))
                            gender_probs = face.get('gender', {})
                            if gender_probs:
                                dominant_g = face.get('dominant_gender', '')
                                conf = gender_probs.get(dominant_g, 0)
                                st.metric("Gender Confidence", f"{conf:.1f}%")

                        st.write("**Emotion Breakdown:**")
                        emotions = face.get('emotion', {})
                        if emotions:
                            sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
                            for emo_name, emo_val in sorted_emotions:
                                st.progress(emo_val / 100, text=f"{emo_name}: {emo_val:.1f}%")

                # --- Face Recognition ---
                st.divider()
                user_db_path = os.path.join(FACE_DB_DIR, st.session_state.current_user)
                if os.path.exists(user_db_path) and os.listdir(user_db_path):
                    with st.spinner("🔎 Searching face database..."):
                        matches = try_find_face(cv2_img, user_db_path)
                    if matches:
                        st.success(f"🎯 Recognised: **{', '.join(matches)}**")
                    else:
                        st.info("No match found in your face database.")
                else:
                    st.info("No faces registered yet. Go to **Training** to add faces.")
            else:
                st.warning("No faces detected in this image.")

        st.divider()
        if st.button("💾 Save Scan & Image to Storage"):
            saved_filename, saved_path = save_image_locally(
                st.session_state.last_image, 
                st.session_state.current_user
            )
            emotion_str = ""
            if results:
                emotions_list = [f.get('dominant_emotion', 'N/A') for f in results]
                emotion_str = ", ".join(emotions_list)
            
            new_record = {
                "Date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                "User": st.session_state.current_user, 
                "Emotion": emotion_str if emotion_str else "No face detected",
                "Status": "Analysed", 
                "File Name": saved_filename,
                "File Path": saved_path 
            }
            save_scan_record(new_record)
            st.success(f"✅ Saved: {saved_filename}")
                
    else:
        st.warning("No active scan found for this session.")
        st.info("Please go to the **Recognition** page to start a new scan.")

def training_page():
    st.title("🧠 Face Training Database")
    st.write("Register faces of known people so the system can recognise them.")
    
    current_user = st.session_state.current_user
    user_db_path = os.path.join(FACE_DB_DIR, current_user)
    os.makedirs(user_db_path, exist_ok=True)

    # --- Register New Person ---
    st.subheader("➕ Register a New Person")
    
    person_name = st.text_input("Person's Name", placeholder="e.g. John, Jane, Ali...")
    
    col_cam, col_upload = st.columns(2)
    
    with col_cam:
        st.write("**Capture from Webcam:**")
        cam_img = st.camera_input("Take a photo", label_visibility="collapsed", key="training_cam")
    
    with col_upload:
        st.write("**Or Upload Images:**")
        uploaded_files = st.file_uploader(
            "Upload face images", 
            type=['jpg', 'png', 'jpeg'], 
            accept_multiple_files=True,
            key="training_upload"
        )

    if st.button("💾 Save Face(s) to Database"):
        if not person_name.strip():
            st.warning("⚠️ Please enter a person's name first.")
        else:
            person_folder = os.path.join(user_db_path, person_name.strip())
            os.makedirs(person_folder, exist_ok=True)
            saved_count = 0

            # Save camera capture
            if cam_img:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = os.path.join(person_folder, f"cam_{timestamp}.jpg")
                with open(filepath, "wb") as f:
                    f.write(cam_img.getbuffer())
                saved_count += 1

            # Save uploaded files
            if uploaded_files:
                for idx, ufile in enumerate(uploaded_files):
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    ext = os.path.splitext(ufile.name)[1] if hasattr(ufile, 'name') else ".jpg"
                    filepath = os.path.join(person_folder, f"upload_{timestamp}_{idx}{ext}")
                    with open(filepath, "wb") as f:
                        f.write(ufile.getbuffer())
                    saved_count += 1

            if saved_count > 0:
                # Clear any cached representations so DeepFace re-indexes
                pkl_files = [f for f in os.listdir(user_db_path) if f.endswith('.pkl')]
                for pkl in pkl_files:
                    try: os.remove(os.path.join(user_db_path, pkl))
                    except: pass
                st.success(f"✅ Saved {saved_count} image(s) for **{person_name.strip()}**!")
                st.rerun()
            else:
                st.warning("No images provided. Please capture or upload at least one image.")

    # --- View Registered People ---
    st.divider()
    st.subheader("📋 Registered People")
    
    if os.path.exists(user_db_path):
        people = [d for d in os.listdir(user_db_path) 
                  if os.path.isdir(os.path.join(user_db_path, d))]
        
        if people:
            for person in people:
                person_path = os.path.join(user_db_path, person)
                images = [f for f in os.listdir(person_path) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                with st.expander(f"👤 {person} — {len(images)} image(s)"):
                    if images:
                        cols = st.columns(min(len(images), 4))
                        for idx, img_name in enumerate(images[:4]):
                            with cols[idx % 4]:
                                img_path = os.path.join(person_path, img_name)
                                st.image(img_path, caption=img_name, use_container_width=True)
                        
                        if len(images) > 4:
                            st.caption(f"... and {len(images) - 4} more image(s)")
                    
                    if st.button(f"🗑️ Delete {person}", key=f"del_{person}"):
                        import shutil
                        shutil.rmtree(person_path)
                        # Clear pkl cache
                        pkl_files = [f for f in os.listdir(user_db_path) if f.endswith('.pkl')]
                        for pkl in pkl_files:
                            try: os.remove(os.path.join(user_db_path, pkl))
                            except: pass
                        st.success(f"Deleted **{person}** from database.")
                        st.rerun()
        else:
            st.info("No people registered yet. Use the form above to add someone.")
    else:
        st.info("No face database found. Start by registering a person above.")

def live_emotion_page():
    st.title("🎭 Live Emotion Detection")
    st.write("Capture frames from your webcam to detect emotions in real time.")
    st.caption("Each captured frame is analysed for faces and emotions. "
               "A green bounding box is drawn around detected faces with the emotion label.")

    st.divider()

    # Camera input — each time the user takes a photo, it's analysed
    img_file = st.camera_input("📷 Capture a frame for emotion analysis", key="live_emotion_cam")

    if img_file is not None:
        # Convert to OpenCV
        pil_img = Image.open(img_file)
        cv2_img = pil_to_cv2(pil_img)

        with st.spinner("Analysing emotions..."):
            annotated_img, results, error = annotate_faces(cv2_img)

        col1, col2 = st.columns([1.2, 0.8])

        with col1:
            annotated_pil = cv2_to_pil(annotated_img)
            st.image(annotated_pil, caption="Live Analysis", use_container_width=True)

        with col2:
            if error:
                st.error(f"Error: {error}")
            elif results:
                for i, face in enumerate(results):
                    emotion = face.get('dominant_emotion', 'N/A')
                    age = face.get('age', '?')
                    gender = face.get('dominant_gender', '?')
                    
                    st.markdown(f"""
                    ### Face #{i+1}
                    | Attribute | Value |
                    |-----------|-------|
                    | **Emotion** | {emotion} |
                    | **Age** | {age} |
                    | **Gender** | {gender} |
                    """)
                    
                    # Mini emotion bars
                    emotions = face.get('emotion', {})
                    if emotions:
                        top_3 = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]
                        for emo_name, emo_val in top_3:
                            st.progress(emo_val / 100, text=f"{emo_name}: {emo_val:.1f}%")
            else:
                st.info("No faces detected. Try adjusting your position.")
        
        st.caption("💡 **Tip:** Click the camera button again to capture another frame and see updated emotions!")
    else:
        st.info("👆 Click the camera button above to start capturing frames.")

def storage_page():
    st.title("📂 Data Storage")
    
    full_history = st.session_state.scan_history
    current_user = st.session_state.current_user
    user_scans = [scan for scan in full_history if scan.get('User') == current_user]
    
    if not user_scans:
        st.info(f"No scans found for user: {current_user}")
    else:
        df = pd.DataFrame(user_scans)
        display_cols = [c for c in df.columns if c != "File Path"]
        st.dataframe(df[display_cols], use_container_width=True)
        st.divider()
        
        st.subheader("Manage Records")
        
        options = [r['File Name'] for r in user_scans]
        
        col_list, col_preview, col_btn = st.columns([2, 2, 1])
        
        with col_list:
            selected_filename = st.selectbox("Select Record to View/Delete", options)
        
        with col_preview:
            if selected_filename:
                selected_record = next((item for item in user_scans if item["File Name"] == selected_filename), None)
                if selected_record and os.path.exists(selected_record["File Path"]):
                    st.image(selected_record["File Path"], caption="Preview", use_container_width=True)
                else:
                    st.warning("Image file missing")

        with col_btn:
            st.write("") 
            st.write("")
            if st.button("Delete Selected 🗑️"):
                if selected_filename:
                    delete_scan_by_filename(selected_filename)

def settings_page():
    st.title("⚙️ Settings")
    st.subheader("Appearance")
    mode_options = ["Light Mode (White)", "Dark Mode (Dark Grey)"]
    current_index = 0 if st.session_state.theme == "light" else 1
    choice = st.radio("Theme Selection", mode_options, index=current_index, label_visibility="collapsed")
    selected_theme = "light" if "Light" in choice else "dark"
    
    if st.button("Apply Theme Change"):
        st.session_state.theme = selected_theme
        st.rerun()

    st.divider()
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.current_user = None 
        st.session_state.last_image = None
        st.session_state.theme = "light" 
        st.rerun()

# --- 6. NAVIGATION SETUP ---
login_screen = st.Page(login_page, title="Login", icon="🔒")
recognition_screen = st.Page(recognition_page, title="Recognition", icon="📸")
detection_screen = st.Page(detection_page, title="Detection", icon="📊")
training_screen = st.Page(training_page, title="Training", icon="🧠")
live_emotion_screen = st.Page(live_emotion_page, title="Live Emotion", icon="🎭")
storage_screen = st.Page(storage_page, title="Storage", icon="💾")
settings_screen = st.Page(settings_page, title="Settings", icon="🛠️")

if st.session_state.logged_in:
    pg = st.navigation({
        "Nautilus App": [recognition_screen, detection_screen, training_screen, live_emotion_screen, storage_screen, settings_screen]
    })
else:
    pg = st.navigation([login_screen])

pg.run()

############################################### IMPORTANT INSTRUCTION ############################################
# To run >>> python -m streamlit run app.py
# To stop >>> ctrl + C 
# username = admin password = 1234 < This needs to be deleted for security reasons at the end
#
# pip install streamlit pandas pillow deepface opencv-python tf-keras
# 