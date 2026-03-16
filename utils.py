import zipfile
import os
import shutil
from config import COLLECTIONS_DIR

def export_collection(collection_name: str, output_zip_path: str):
    coll_path = os.path.join(COLLECTIONS_DIR, collection_name)
    if not os.path.exists(coll_path):
        raise FileNotFoundError(f"Collection {collection_name} not found")
    
    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(coll_path):
            for file in files:
                full = os.path.join(root, file)
                arc = os.path.relpath(full, COLLECTIONS_DIR)
                zipf.write(full, arc)

def import_collection(zip_path: str, new_name: str = None):
    target_name = new_name or os.path.splitext(os.path.basename(zip_path))[0]
    target_path = os.path.join(COLLECTIONS_DIR, target_name)
    
    if os.path.exists(target_path):
        raise FileExistsError(f"'{target_name}' already exists")
    
    os.makedirs(target_path, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        zipf.extractall(target_path)
    return target_name