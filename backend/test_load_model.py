import gzip, pickle, joblib, sys, traceback
from pathlib import Path
P=Path('backend/models/stock_model_compressed.pkl')
print('file',P.exists(),P.stat().st_size)
# try gzip
try:
    with gzip.open(P,'rb') as f:
        obj=pickle.load(f)
    print('loaded via gzip, type',type(obj))
except Exception as e:
    print('gzip failed:',str(e))
    traceback.print_exc()
# try joblib
try:
    obj=joblib.load(P)
    print('loaded via joblib, type',type(obj))
except Exception as e:
    print('joblib failed:',str(e))
    traceback.print_exc()
# try pickle
try:
    with open(P,'rb') as f:
        obj=pickle.load(f)
    print('loaded via pickle, type',type(obj))
except Exception as e:
    print('pickle failed:',str(e))
    traceback.print_exc()
