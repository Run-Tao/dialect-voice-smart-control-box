from canto_tts import CantoTTS
from pathlib import Path
import time
out = Path(r'D:\Run_Tao\DialectVoice_Research\data\tts_validation\canto')
out.mkdir(parents=True, exist_ok=True)
t0=time.perf_counter()
tts=CantoTTS()
print('init_sec', round(time.perf_counter()-t0,2))
t1=time.perf_counter()
tts.synthesize('開燈', str(out/'test_open_light.wav'))
print('gen_sec', round(time.perf_counter()-t1,2))
print('exists', (out/'test_open_light.wav').exists(), (out/'test_open_light.wav').stat().st_size)
