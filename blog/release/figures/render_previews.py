"""Raster previews of our deliberately simple SVG subset using existing Pillow.

Supports rectangles, M/L paths with arrow markers, and text used by
build_figures.py. No browser, downloaded fonts, or external SVG renderer needed.
The SVG files remain the publication masters.
"""
from pathlib import Path
import math
import os
import re
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.mplconfig'))
from PIL import Image, ImageDraw, ImageFont
from matplotlib.font_manager import FontProperties, findfont

SCALE = 2
FONTS = {w: findfont(FontProperties(family='DejaVu Sans', weight=w)) for w in ('normal', 'bold')}

def render(path):
    root = ET.parse(path).getroot()
    w, h = int(root.get('width')), int(root.get('height'))
    img = Image.new('RGB', (w*SCALE, h*SCALE), 'white')
    draw = ImageDraw.Draw(img)
    def point(x,y): return (round(x*SCALE),round(y*SCALE))
    def recurse(el):
        tag = el.tag.rsplit('}',1)[-1]
        if tag in ('defs','title','desc'): return
        if tag in ('svg','g'):
            for child in el: recurse(child)
        elif tag == 'rect':
            x,y=float(el.get('x',0)),float(el.get('y',0))
            rw,rh=float(el.get('width')),float(el.get('height'))
            bbox=(*point(x,y),*point(x+rw,y+rh))
            fill=el.get('fill'); stroke=el.get('stroke'); sw=round(float(el.get('stroke-width',1))*SCALE)
            if el.get('stroke-dasharray'):
                # A dashed frame with square corners in this preview; SVG retains rounding.
                draw.rectangle(bbox,fill=fill)
                dash,gap=[float(v)*SCALE for v in el.get('stroke-dasharray').split()]
                for start,end in [((bbox[0],bbox[1]),(bbox[2],bbox[1])),((bbox[2],bbox[1]),(bbox[2],bbox[3])),((bbox[2],bbox[3]),(bbox[0],bbox[3])),((bbox[0],bbox[3]),(bbox[0],bbox[1]))]:
                    dx,dy=end[0]-start[0],end[1]-start[1];length=math.hypot(dx,dy);pos=0
                    while pos<length:
                        a=(start[0]+dx*pos/length,start[1]+dy*pos/length);stop=min(pos+dash,length)
                        b=(start[0]+dx*stop/length,start[1]+dy*stop/length)
                        draw.line((a,b),fill=stroke,width=sw);pos+=dash+gap
            else:
                draw.rounded_rectangle(bbox,radius=round(float(el.get('rx',0))*SCALE),fill=fill,outline=stroke,width=sw)
        elif tag == 'text':
            font=ImageFont.truetype(FONTS[el.get('font-weight','normal')],round(float(el.get('font-size',18))*SCALE))
            anchor={'start':'ls','middle':'ms','end':'rs'}[el.get('text-anchor','start')]
            draw.text(point(float(el.get('x')),float(el.get('y'))),el.text or '',font=font,fill=el.get('fill','#172b42'),anchor=anchor)
        elif tag == 'path':
            tokens=re.findall(r'[ML]|-?\d+(?:\.\d+)?',el.get('d'))
            paths=[];current=[];i=0
            while i<len(tokens):
                cmd=tokens[i];i+=1
                if cmd not in ('M','L'):raise ValueError(el.get('d'))
                if cmd=='M' and current:paths.append(current);current=[]
                current.append(point(float(tokens[i]),float(tokens[i+1])));i+=2
            if current:paths.append(current)
            width=round(float(el.get('stroke-width',1))*SCALE);color=el.get('stroke','#62798d')
            for pts in paths:
                if len(pts)>1:draw.line(pts,fill=color,width=width)
            if el.get('marker-end') and len(paths[-1])>1:
                a,b=paths[-1][-2:];dx,dy=b[0]-a[0],b[1]-a[1];length=math.hypot(dx,dy);ux,uy=dx/length,dy/length
                tip=(b[0]+2*SCALE*ux,b[1]+2*SCALE*uy)
                left=(b[0]-12*SCALE*ux-7*SCALE*uy,b[1]-12*SCALE*uy+7*SCALE*ux)
                right=(b[0]-12*SCALE*ux+7*SCALE*uy,b[1]-12*SCALE*uy-7*SCALE*ux)
                draw.polygon((tip,left,right),fill=color)
        else:raise ValueError(f'Unsupported SVG element: {tag}')
    recurse(root)
    img.resize((w,h),Image.Resampling.LANCZOS).save(path.with_suffix('.png'))
    print(path.with_suffix('.png').name)

if __name__ == '__main__':
    for name in ('pipeline-overview','context-budget','evaluation-design','comparison-placeholder'):
        render(ROOT / f'{name}.svg')
