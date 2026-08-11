python - <<'PY'
import sys, os
print("python:", sys.version)
print("cwd:", os.getcwd())

try:
    import numpy as np
    print("numpy:", np.__version__)
except Exception as e:
    print("numpy import failed:", repr(e))

try:
    import pandas as pd
    print("pandas:", pd.__version__)
    for f in ["Data/E288_200.csv", "Data/E288_300.csv", "Data/E288_400.csv", "Data/E605.csv", "Data/E772.csv"]:
        if os.path.exists(f):
            df = pd.read_csv(f, nrows=2)
            print("\\n", f)
            print("columns:", list(df.columns))
            print(df.head(2).to_string(index=False))
        else:
            print("missing:", f)
except Exception as e:
    print("pandas/data check failed:", repr(e))

try:
    import torch
    print("\\ntorch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("cuda device:", torch.cuda.get_device_name(0))
except Exception as e:
    print("torch import failed:", repr(e))

try:
    import lhapdf
    print("\\nlhapdf module:", lhapdf)
    try:
        print("lhapdf paths:", lhapdf.paths())
    except Exception as e:
        print("lhapdf paths failed:", repr(e))

    for name in [
        "NNPDF40_nnlo_as_01180",
        "NNPDF40_nlo_as_01180",
        "MSHT20nnlo_as118",
        "CT18NNLO",
    ]:
        try:
            s = lhapdf.getPDFSet(name)
            print("PDF set available:", name, "members:", s.size)
        except Exception as e:
            print("PDF set unavailable:", name)
except Exception as e:
    print("lhapdf import failed:", repr(e))
PY