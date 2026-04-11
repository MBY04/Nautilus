import streamlit as st
import os
import datetime
import json
import pandas as pd
from PIL import Image

# --- 1. SETUP & CONFIGURATION ---
LOGO_PATH = "images/NautilusLogoDesign.png"
USER_DB_FILE = "users.json"
SCANS_DB_FILE = "scans.json" 
SCANS_DIR = "scanned_images" 

# We don't create the directory here anymore, we create it per-user later
os.makedirs(SCANS_DIR, exist_ok=True)

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
    # 1. Create a folder for THIS user: scanned_images/admin/
    user_folder = os.path.join(SCANS_DIR, username)
    os.makedirs(user_folder, exist_ok=True)
    
    # 2. Save file inside that folder
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
        
        # Delete the actual file
        if file_path and os.path.exists(file_path):
            try: os.remove(file_path)
            except Exception as e: st.error(f"Error deleting file: {e}")
        
        # Remove from data
        full_history.pop(index)
        save_json_db(SCANS_DB_FILE, full_history)
        st.session_state.scan_history = full_history
        st.success("Record and image deleted.")
        st.rerun()

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
    # UPDATED: Using a soft off-white for better contrast than pure white
    bg_color = "#F5F7F9"     
    text_color = "#000000"   
    input_bg = "#ffffff"     # Inputs can stay bright white for contrast against bg
    input_text = "#000000"
    placeholder_color = "#555555" # Slightly darker placeholder for visibility
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
    st.title("🔍 Detection")
    
    if st.session_state.last_image:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(st.session_state.last_image, caption="Source Image", use_container_width=True)
        with col2:
            st.info("Image Loaded.")
            st.subheader("Analysis")
            st.write("Waiting for DeepFace integration...") 
            st.divider()
            
            if st.button("Save Scan & Image to Storage"):
                saved_filename, saved_path = save_image_locally(
                    st.session_state.last_image, 
                    st.session_state.current_user
                )
                new_record = {
                    "Date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "User": st.session_state.current_user, 
                    "Status": "Pending", 
                    "File Name": saved_filename,
                    "File Path": saved_path 
                }
                save_scan_record(new_record)
                st.success(f"Saved: {saved_filename}")
                
    else:
        st.warning("No active scan found for this session.")
        st.info("Please go to the **Recognition** page to start a new scan.")

def storage_page():
    st.title("📂 Data Storage")
    
    full_history = st.session_state.scan_history
    current_user = st.session_state.current_user
    user_scans = [scan for scan in full_history if scan.get('User') == current_user]
    
    if not user_scans:
        st.info(f"No scans found for user: {current_user}")
    else:
        df = pd.DataFrame(user_scans)
        st.dataframe(df.drop(columns=["File Path"]), use_container_width=True)
        st.divider()
        
        st.subheader("Manage Records")
        
        options = [r['File Name'] for r in user_scans]
        
        # Adjusted column widths for better layout
        col_list, col_preview, col_btn = st.columns([2, 2, 1])
        
        with col_list:
            selected_filename = st.selectbox("Select Record to View/Delete", options)
        
        with col_preview:
            if selected_filename:
                selected_record = next((item for item in user_scans if item["File Name"] == selected_filename), None)
                if selected_record and os.path.exists(selected_record["File Path"]):
                    # UPDATED: use_container_width=True makes it fill the column
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
storage_screen = st.Page(storage_page, title="Storage", icon="💾")
settings_screen = st.Page(settings_page, title="Settings", icon="🛠️")

if st.session_state.logged_in:
    pg = st.navigation({"Nautilus App": [recognition_screen, detection_screen, storage_screen, settings_screen]})
else:
    pg = st.navigation([login_screen])

pg.run()

############################################### IMPORTANT INSTRUCTION ############################################
# To run >>> python -m streamlit run app.py
# To stop >>> ctrl + C 
# username = admin password = 1234 < This needs to be deleted for security reasons at the end

# Please put the command below as it is important for it working!
#
# pip install streamlit pandas pillow
# 