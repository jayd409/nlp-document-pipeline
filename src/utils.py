import base64, os
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style='whitegrid', palette='muted')

def to_b64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b64

def save_html(charts, title, kpis=None, path='outputs/dashboard.html'):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    kpi_html = ''
    if kpis:
        kpi_html = '<div class="kpis">' + ''.join(
            f'<div class="kpi"><b>{v}</b><span>{k}</span></div>' for k,v in kpis) + '</div>'
    cards = ''.join(
        f'<div class="card"><p>{n}</p><img src="data:image/png;base64,{to_b64(f)}"/></div>'
        for n,f in charts)
    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>{title}</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f5f7fa;margin:0;padding:0;color:#2d3748}}
.header{{background:linear-gradient(135deg,#1a202c,#2d3748);color:white;padding:32px 40px}}
.header h1{{margin:0;font-size:1.6rem}}
.header p{{margin:4px 0 0;opacity:.7;font-size:.95rem}}
.kpis{{display:flex;gap:16px;padding:24px 40px;flex-wrap:wrap}}
.kpi{{background:white;border-radius:10px;padding:18px 24px;flex:1;min-width:140px;box-shadow:0 1px 4px rgba(0,0,0,.08)}}
.kpi b{{display:block;font-size:1.6rem;color:#2b6cb0}}
.kpi span{{font-size:.8rem;color:#718096}}
.grid{{display:flex;flex-wrap:wrap;gap:20px;padding:0 40px 40px}}
.card{{background:white;border-radius:10px;padding:16px;box-shadow:0 1px 4px rgba(0,0,0,.08);flex:1;min-width:340px}}
.card p{{font-size:.85rem;color:#718096;margin:0 0 8px}}
.card img{{width:100%;height:auto}}
</style></head>
<body>
<div class="header"><h1>{title}</h1><p>NLP Pipeline — Summarization · Q&A · Text Generation</p></div>
{kpi_html}
<div class="grid">{cards}</div>
</body></html>"""
    with open(path, 'w') as f:
        f.write(html)
