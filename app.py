import streamlit as st
from cbir import search, distance_functions
from PIL import Image
import os

st.set_page_config(page_title="CBIR Intelligent", layout="wide")

st.title("Recherche d'Images par Contenu (CBIR)")
st.markdown("---")

with st.sidebar:
    st.header("Parametres")
    
    uploaded_file = st.file_uploader("Televerser une image", type=["jpg", "png", "jpeg"])
    
    k = st.slider("Nombre de resultats", min_value=1, max_value=20, value=5)
    
    distance_name = st.selectbox(
        "Mesure de distance",
        options=list(distance_functions.keys()),
        index=0
    )
    
    descriptor_name = st.selectbox(
        "Descripteur",
        options=["GLCM", "Haralick", "BitDesc", "Concaténation"],
        index=3
    )
    
    st.markdown("---")
    st.info("Le systeme predit d'abord la classe de l'image, puis recherche uniquement dans cette categorie.")

if uploaded_file is not None:
    temp_path = "temp_query.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Image requete")
        st.image(uploaded_file, use_container_width=True)
    
    with col2:
        st.subheader("Recherche en cours")
        
        with st.spinner("Analyse de l'image et recherche des similaires..."):
            results = search(temp_path, k, distance_name, descriptor_name)
        
        if results:
            st.success(f"{len(results)} images similaires trouvees")
            
            st.subheader("Resultats de la recherche")
            
            cols = st.columns(min(4, k))
            for idx, (img_path, distance) in enumerate(results):
                col_idx = idx % len(cols)
                with cols[col_idx]:
                    try:
                        img = Image.open(img_path)
                        st.image(img, use_container_width=True)
                        st.caption(f"Distance {distance_name}: {distance:.4f}")
                        st.caption(f"Fichier: {os.path.basename(img_path)}")
                    except:
                        st.error("Erreur chargement image")
        else:
            st.error("Aucun resultat trouve")
    
    if os.path.exists(temp_path):
        os.remove(temp_path)

else:
    st.info("Televersez une image dans la barre laterale pour commencer la recherche")
    
    with st.expander("Comment utiliser l'application"):
        st.markdown("""
        1. Televersez une image dans la barre laterale
        2. Ajustez les parametres
        3. Analysez les resultats

        Mesures de distance: Euclidienne, Canberra, Cosinus
        Descripteurs: GLCM, Haralick, BitDesc, Concatenation
        """)