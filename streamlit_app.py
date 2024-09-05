import streamlit as st
from streamlit_option_menu import option_menu
import os




with st.sidebar:
    selected = option_menu(
        menu_title= 'SuperPoint Transformer',
        options= ['🏠 Home', '🛠️ Segmentation Tool'],
        default_index= 0, 
    )
if selected == '🏠 Home':
    
    # Apply custom CSS styles for the dark theme and improved design
    st.markdown(
        """
        <style>
        /* Set the dark mode theme for the application */
        body {
            background-color: #121212;
            color: #ffffff;
            font-family: 'Segoe UI', 'Helvetica', 'Arial', sans-serif;
        }
        /* Header styles */
        .header {
            padding: 20px;
            background-color: #1a535c;
            color: #ffffff;
            text-align: center;
            margin-bottom: 20px;
        }
        /* Logo styles */
        .logo {
            display: block;
            margin: 0 auto;
            width: 150px;
            border-radius: 50%;
        }
        /* Content styles */
        .content {

            background-color: #333;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
            margin-bottom: 20px;
        }
        /* Styles for headings and text */
        h1, h2 {
            color: #4ecdc4;
        }
        /* Button styles */
        .button {
            display: inline-block;
            padding: 10px 20px;
            background-color: #4ecdc4;
            color: #121212;
            font-size: 16px;
            font-weight: bold;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            transition: background-color 0.3s, color 0.3s;
        }
        .button:hover {
            background-color: #2b7a78;
            color: #ffffff;
        }
        /* Image styles */
        .image {
            width: 100%;
            border-radius: 10px;
            margin-top: 20px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.title("Superpoint Transformer")

    st.write("The work present in this web-app is an implementation of the model presented in the paper \"Scalable 3D Panoptic Segmentation As Superpoint Graph Clustering (3DV 2024 Oral)\".")
    st.write("Make sure to visit their github and star the repo if you ❤️ the work.")
    st.link_button("SPT Github Repository.","https://github.com/drprojects/superpoint_transformer")
    st.write("If you like this project, don't forget to visit my github by clicking the github icon in the top right corner and drop a ⭐🤩. It would be greatly appreciated! Thanks in advance!")


    st.image("./new_superpoint_transformer/media/teaser_spt.png",output_format="auto")

    st.markdown("""**Superpoint Transformer (SPT)** is a superpoint-based transformer 🤖 architecture that efficiently ⚡ 
    performs **semantic segmentation** on large-scale 3D scenes. This method includes a 
    fast algorithm that partitions 🧩 point clouds into a hierarchical superpoint 
    structure, as well as a self-attention mechanism to exploit the relationships 
    between superpoints at multiple scales. """)
    st.markdown("""
                | ✨ SPT in numbers ✨ |
                | :---: |
                | 📊 **SOTA on S3DIS 6-Fold** (76.0 mIoU) |
                | 📊 **SOTA on KITTI-360 Val** (63.5 mIoU) |
                | 📊 **Near SOTA on DALES** (79.6 mIoU) | 
                | 🦋 **212k parameters** ([PointNeXt](https://github.com/guochengqian/PointNeXt) ÷ 200, [Stratified Transformer](https://github.com/dvlab-research/Stratified-Transformer) ÷ 40) | 
                | ⚡ S3DIS training in **3h on 1 GPU** ([PointNeXt](https://github.com/guochengqian/PointNeXt) ÷ 7, [Stratified Transformer](https://github.com/dvlab-research/Stratified-Transformer) ÷ 70) | 
                | ⚡ **Preprocessing x7 faster than [SPG](https://github.com/loicland/superpoint_graph)** |
    """)
    st.write("This implementation of the model has only been trained on Area 5 of S3DIS, on a T4 GPU machine. Resulting in metrics such as:")
    st.write("Mean IoU: 67%")
    st.write("Overall Accuracy: 89%")


    st.write("Have fun!")



elif selected == '🛠️ Segmentation Tool':
     # Apply custom CSS styles for the dark theme and improved design
    st.markdown(
        """
        <style>
        /* Styles for headings and text */
        h1, h2 {
            color: #4ecdc4;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    input_path =st.file_uploader("Upload a 3d point cloud file.")
    data_path = ""



    
