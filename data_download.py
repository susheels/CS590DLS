import struct
import pickle
import os
import re
import wget
import zipfile
import rgx_utils as rgx
import task_utils as utils

utils.download_and_unzip('https://samate.nist.gov/SRD/testsuites/juliet/Juliet_Test_Suite_v1.3_for_C_Cpp.zip','Juliet Test Suite Dataset','data')
try:
    os.makedirs('pretrained')
except FileExistsError:
    pass

