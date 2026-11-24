import os
import glob

workspace = r"c:\Users\varsh\OneDrive\Desktop\software"
for p in glob.glob(os.path.join(workspace, "**/*.py"), recursive=True):
    with open(p, "r", encoding="utf-8") as f:
        content = f.read()
    
    if "from software" in content:
        new_content = content.replace("from ", "from ")
        new_content = new_content.replace("from ", "from ")
        with open(p, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"Updated {p}")
