import json
import os
import pickle
import random as r
import scipy.io.wavfile
import numpy as np
# import bob.kaldi
# import bob.io.audio
import sys
from ast import literal_eval
from functools import reduce

import pandas as pd
from p2fa import align
from pydub import AudioSegment
from pydub import AudioSegment
from pydub.playback import play

directory_seperator = '/' if "linux" in sys.platform.lower() else '\\'
system_type = 'linux' if "linux" in sys.platform.lower() else 'windows'
