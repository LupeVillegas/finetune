

import streamlit as st
import subprocess
import os
import time
from pathlib import Path


st.title('Fine-tune with microSAM')
st.subheader("📂 File Search and Picker", divider = True)

#Get the base folder
base_folder = Path.cwd()  
st.write(f"Root folder: `{base_folder}`")

#List all fOLDERS recursively
all_folders = [p for p in base_folder.rglob("*") if p.is_dir()]

#Filter files based on the search 
user_path = st.text_input("🔍 Search for a new root folder")
if 'root_path' not in st.session_state:
    st.session_state.root_path = None

if user_path:
    if os.path.isdir(user_path):
        st.success(f"Folder found: {user_path}")
        
        # Save as root path
        st.session_state.root_path = user_path
        
        # List folders inside
        filtered_folders = [f for f in os.listdir(user_path) if os.path.isdir(os.path.join(user_path, f))]
        base_folder = user_path 
        st.write(f"New root folder: `{base_folder}`")
        
    else:
        st.error("Invalid path. Please enter a valid directory.")
else:
    filtered_folders = all_folders



#Let user select one folder 
if filtered_folders:
    selected_folder = st.selectbox("Select a folder", filtered_folders)
    st.success(f"✅ You selected: `{selected_folder}`")
else:
    st.warning("⚠️ No folders found matching your search.")


if ('size', 'model') not in st.session_state:
    st.session_state.size = 30
    st.session_state.model = 'vit_t_lm'

# Add a selectbox to the sidebar:
st.subheader("Choose a Model", divider = True)
model = st.selectbox(
    'MicroSAM Model:',
    ('vit_l_lm', 'vit_b_lm', 'vit_t_lm', 'vit_b', 'vit_t')
)
st.session_state.model = model
st.write('Model = ', st.session_state.model)

minimal_size = st.sidebar.number_input('Minimal size', value = 30, step = 1)
st.session_state.size = minimal_size 
st.sidebar.write('Size = ', st.session_state.size)

if 'train' not in st.session_state:
    st.session_state.train = 80

train_percentage = st.sidebar.number_input('Train images %', value = 80, min_value=0, max_value=100, step = 5)
st.session_state.train = train_percentage
st.sidebar.write('Train = ', st.session_state.train, '%')
st.sidebar.write('Validation = ', 100 - st.session_state.train, '%')


if ('batch', 'epoch') not in st.session_state:
    st.session_state.batch = 1
    st.session_state.epoch = 5

batch = st.sidebar.number_input('Batch', value = 1, step = 1)
st.session_state.batch = batch
st.sidebar.write('Batch = ', st.session_state.batch)

epoch = st.sidebar.number_input('Epoch', value = 5, step = 1)
st.session_state.epoch = epoch
st.sidebar.write('Epoch = ', st.session_state.epoch)

#================
st.subheader("🚀 Train ", divider = True)

if 'name' not in st.session_state:
    st.session_state.name = "my_model"

name = st.text_input("Enter name for you model: ")
st.session_state.name = name
st.write('Name model  = ', st.session_state.name)


# Button to run

# if st.button("Run Script"):
#     #if base_folder:
#     result = subprocess.run(["python", "finetune_dummy.py", base_folder, model, str(minimal_size), str(train_percentage), str(batch), str(epoch), name], 
#                                 capture_output = True, text=True)
#     output = result.stdout.strip()
#     st.text(output)

# else:
#     st.warning("⚠️ Please enter all variables first.") 

if st.button("Run Script"):
    if name:
        subprocess.run(
            ["python", "finetune_dummy.py", base_folder, model, str(minimal_size), str(train_percentage), str(batch), str(epoch), name],
            capture_output=True,
            text=True, 
        )
        #output = result.stdout.strip()
        #st.text(output)

    else:
        st.warning("⚠️ Please enter all variables first.") 


# import json
# import os
# name = st.text_input("Enter your name")
# age = st.number_input("Enter your age", min_value=0, max_value=120, step=1)

# if st.button("Run Script and Load Data"):
#     if name:
#         subprocess.run(
#             ["python", "external_script_save.py",  base_folder, model, str(minimal_size), str(train_percentage), str(batch), str(epoch), name],
#             capture_output=True,
#             text=True
#         )
        
#     else:
#         st.warning("⚠️ Please enter your name.")

# result = subprocess.run(
#     ["python", "external_script_save.py", name, str(age)],
#     capture_output=True,
#     text=True
# )

# st.code(result.stdout)  # Show all output
# if result.stderr:
#     st.error(result.stderr)  # Show any errors



# Progress BAR
'Starting a long computation...'
latest_iteration = st.empty()
bar = st.progress(0)

for i in range(100):
  # Update the progress bar with each iteration.
  latest_iteration.text(f'Iteration {i+1}')
  bar.progress(i + 1)
  

'...and now we\'re done!'




#link = st.text_input(label="Insert link", value="", max_chars= 100, placeholder="Example: bla", help="Be sure to insert a valid URL", on_change=adogniletterainserita, args=[st.session_state.link_input] 
#st.link_button("Go to gallery", "https://streamlit.io/gallery")  ##Click a botton and goo link

#st.page_link("pague", r"C:\Users\malieva_lab\Documents\How_To\how_Streamlit")

#st.page_link("http://www.google.com", label="Google", icon="🌎")
#st.page_link("finetune_mSAM.py", label="Home", icon="🏠")
#st.page_link("data/segmentation", label="Page 1", icon="1️⃣")
#st.page_link("data/image", label="Page 2", icon="2️⃣", disabled=True)

#Chose multiple files to upload a the save time  --- 
#uploaded_files = st.file_uploader(
#    "Choose a CSV file", accept_multiple_files=True
#)
#for uploaded_file in uploaded_files:
#    bytes_data = uploaded_file.read()
#    st.write("filename:", uploaded_file.name)
#    st.write(bytes_data)

#url = "https://www.streamlit.io"
#st.write("check out this [link](%s)" % url)
#st.markdown("check out this [link](%s)" % url)



if st.button("Get Root Directory"):
    root_path = Path.cwd()  # Get current working directory
    st.success(f"Root Path: {root_path}")

    import streamlit as st

