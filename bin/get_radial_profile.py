import json
import numpy as np
from cee_profile import module

file_path ='/Users/vinayavalsan/ILOT_work/codes/cee-profiles/test' 
fileid = 500 
FileID = "{0:06d}".format(fileid)
full_profile = module.read_data(file_path).get_full_radial_profile(FileID)

with open(f'{FileID}_profile.json','w') as f:
    json.dump(full_profile, f)


