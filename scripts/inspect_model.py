import joblib, sklearn, traceback, re, sys
print('INSTALLED_SKLEARN::', sklearn.__version__)
try:
    joblib.load('src/models/credit_risk_model.pkl')
    print('MODEL_LOADED::SUCCESS')
except Exception as e:
    print('LOAD_ERROR::', type(e).__name__, e)
    traceback.print_exc()
# search for version-like bytes inside the pickle
try:
    b = open('src/models/credit_risk_model.pkl','rb').read()
    m = re.search(rb'\d+\.\d+(?:\.\d+)?', b)
    if m:
        print('FOUND_VERSION_LIKE_IN_PICKLE::', m.group().decode())
    else:
        print('FOUND_VERSION_LIKE_IN_PICKLE::NONE')
except Exception as e:
    print('PICKLE_READ_ERROR::', e)
    sys.exit(2)
