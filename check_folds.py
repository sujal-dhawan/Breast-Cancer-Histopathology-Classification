import os, glob

print("📂 Current working dir:", os.getcwd())

# List all files in data/
if os.path.exists("data"):
    print("📁 Files inside data/:", os.listdir("data"))
else:
    print("❌ 'data' folder not found")

# Search for folds file
folds = glob.glob("data/Folds.*") + glob.glob("data/BreaKHis_v1/Folds.*")
print("🔍 Found folds files:", folds)
