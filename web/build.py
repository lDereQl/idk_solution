import base64, io, json, os, sys
from PIL import Image

S = os.path.dirname(os.path.abspath(__file__))
IN, OUT = S + '/in', S + '/out'
os.makedirs(OUT, exist_ok=True)

tpl = open(S + '/template.html', encoding='utf-8').read()
head, body = tpl.split('<!--BODY-->')
head = head.replace('<!--HEAD-->', '').strip()

data = json.load(open(IN + '/kaunas_demo_data.json', encoding='utf-8'))
gj = json.load(open(IN + '/osm_pipelines_all_kaunas.geojson', encoding='utf-8'))
kte = [[[round(x, 6), round(y, 6)] for x, y, *_ in f['geometry']['coordinates']]
       for f in gj['features'] if f['properties'].get('operator') == 'Kauno termofikacinė elektrinė']
assert len(kte) == 26

net = 'data:image/png;base64,' + base64.b64encode(open(IN + '/szns_heat_AZ_kaunas_LKS94.png', 'rb').read()).decode()
buf = io.BytesIO()
# Static backdrop: keep the imagery, take out the baked-in legend, plant labels, markers and orange lines
# (plants are out of scope, and the network is drawn live on top).
import numpy as np
from PIL import ImageFilter
src = Image.open(IN + '/kaunas_dh_map_preview.png').convert('RGB')
a = np.array(src).astype(int); r, g, b = a[..., 0], a[..., 1], a[..., 2]
m = (a.max(axis=2) > 135) | ((r > 120) & (r - g > 40) & (g - b > 5))
mask = Image.fromarray((m * 255).astype('uint8')).filter(ImageFilter.MaxFilter(11))
for box in [(0, 0, 330, 145), (1130, 10, 1195, 95), (10, 705, 290, 760), (420, 742, 1200, 800)]:
    mask.paste(255, box)
soft = src.copy()
arr = np.array(soft); arr[np.array(mask) > 0] = np.median(arr[np.array(mask) == 0], axis=0); soft = Image.fromarray(arr)
soft = soft.filter(ImageFilter.MedianFilter(9)).filter(ImageFilter.GaussianBlur(14))
clean = Image.composite(soft, src, mask.filter(ImageFilter.GaussianBlur(3)))
clean.save(OUT + '/backdrop_clean.png')
clean.save(buf, 'JPEG', quality=84)
backdrop = 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()

js = lambda o: json.dumps(o, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/')
body = (body.replace('/*__DATA__*/null', js(data)).replace('/*__KTE__*/[]', js(kte))
            .replace('__NET_URI__', net).replace('__BACKDROP_URI__', backdrop))

full = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n' + head + '\n</head>\n<body>\n' + body.strip() + '\n</body>\n</html>\n')
open(OUT + '/kaunas-heat-forecast-demo.html', 'w', encoding='utf-8').write(full)
open(OUT + '/kaunas-heat-forecast.html', 'w', encoding='utf-8').write(head + '\n' + body.strip() + '\n')
print(len(full) / 1e6, 'MB')
