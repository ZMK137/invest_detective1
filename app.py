import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, render_template, request, session
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from scipy.io import arff
import os
import datetime

app = Flask(__name__)
app.secret_key = 'TAJNY_KLUCZ_SESJI_INVEST_DETECTIVE'

# ======================================================
# 1. MODEL ML
# ======================================================
def train_model():
    arff_file = 'data.arff'
    if not os.path.exists(arff_file): return train_fake()
    try:
        data, meta = arff.loadarff(arff_file)
        df = pd.DataFrame(data)
        if df['class'].dtype == object:
            df['class'] = df['class'].astype(str).apply(lambda x: 1 if '1' in x else 0)
        features = ['Attr4', 'Attr2', 'Attr1', 'Attr36']
        valid = [f for f in features if f in df.columns]
        if len(valid) < 4: return train_fake()
        df = df[valid + ['class']].dropna()
        clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        clf.fit(df[valid].values, df['class'].values)
        return clf
    except: return train_fake()

def train_fake():
    np.random.seed(42)
    X = np.random.rand(500, 4)
    X[:, 0] *= 3.0
    y = [1 if (x[0]<1 or x[1]>0.8) else 0 for x in X]
    clf = RandomForestClassifier()
    clf.fit(X, y)
    return clf

ml_model = train_model()

# ======================================================
# 2. MODEL DCF
# ======================================================
def calculate_dcf(stock_obj, cashflow, df_history):
    try:
        if 'Operating Cash Flow' not in cashflow.index: return None
        ocf = cashflow.loc['Operating Cash Flow'].iloc[0]
        capex = cashflow.loc['Capital Expenditure'].iloc[0] if 'Capital Expenditure' in cashflow.index else 0
        fcf = ocf + capex 
        
        shares = stock_obj.info.get('sharesOutstanding', None)
        if not shares: return None

        cagr = 0.03
        if len(df_history) >= 2:
            rev_now = df_history['Przychody'].iloc[0]
            rev_past = df_history['Przychody'].iloc[-1]
            years = len(df_history) - 1
            if years > 0 and rev_past > 0:
                try: cagr = (rev_now / rev_past) ** (1/years) - 1
                except: pass

        growth_rate = max(0.02, min(cagr, 0.15))
        wacc = 0.10
        terminal_growth = 0.025

        future_fcfs = []
        for i in range(1, 6):
            fcf_proj = fcf * ((1 + growth_rate) ** i)
            disc = (1 + wacc) ** i
            future_fcfs.append(fcf_proj / disc)

        last_fcf = fcf * ((1 + growth_rate) ** 5)
        terminal_value = (last_fcf * (1 + terminal_growth)) / (wacc - terminal_growth)
        discounted_tv = terminal_value / ((1 + wacc) ** 5)

        total_value = sum(future_fcfs) + discounted_tv
        fair_value = total_value / shares
        
        return {
            'fair_value': round(fair_value, 2),
            'fcf': fcf,
            'wacc': wacc * 100,
            'growth': round(growth_rate * 100, 2),
            'cagr_raw': round(cagr * 100, 2),
            'sum_pv_5y': sum(future_fcfs),
            'terminal_value_pv': discounted_tv,
            'shares': shares
        }
    except Exception as e:
        return None

# ======================================================
# 3. WYKRESY (POPRAWIONY BENFORD)
# ======================================================
def create_static_charts(df):
    charts = {}
    plt.style.use('dark_background')
    bg_color = '#131722'
    grid_color = '#2a2e39'
    text_color = '#b2b5be'
    line_blue = '#2962ff'
    years = df['Rok'].astype(int).tolist()

    def save_fig(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', facecolor=bg_color, dpi=100)
        buf.seek(0)
        img = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)
        return img
    
    def setup_ax(ax, title):
        ax.set_facecolor(bg_color)
        ax.grid(True, color=grid_color, linestyle='-', linewidth=0.5)
        ax.set_title(title, color='white', fontsize=10, fontweight='bold', loc='left')
        ax.tick_params(colors=text_color, labelsize=8)
        for spine in ax.spines.values(): spine.set_edgecolor(grid_color)

    # 1. Z-Score
    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.patch.set_facecolor(bg_color)
    setup_ax(ax, 'ALTMAN Z-SCORE')
    z = df['Z_Score_Val'].tolist()
    ax.axhline(1.8, color='#f23645', linestyle='--', linewidth=1, alpha=0.7)
    ax.axhline(2.99, color='#089981', linestyle='--', linewidth=1, alpha=0.7)
    ax.plot(years, z, marker='o', color=line_blue, linewidth=1.5, markersize=4)
    charts['z_score'] = save_fig(fig)

    # 2. Rentowność
    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.patch.set_facecolor(bg_color)
    setup_ax(ax, 'RENTOWNOŚĆ')
    ax.plot(years, df['ROE'], label='ROE', color=line_blue, linewidth=1.5)
    ax.plot(years, df['ROA'], label='ROA', color='#d1d4dc', linewidth=1.5, linestyle=':')
    ax.legend(frameon=False, fontsize=8, labelcolor=text_color)
    charts['prof'] = save_fig(fig)

    # 3. Płynność
    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.patch.set_facecolor(bg_color)
    setup_ax(ax, 'PŁYNNOŚĆ')
    ax.fill_between(years, df['Current_Ratio'], color=line_blue, alpha=0.1)
    ax.plot(years, df['Current_Ratio'], color=line_blue, linewidth=1.5)
    ax.axhline(1.0, color='#f23645', linestyle='--', linewidth=1)
    charts['liq'] = save_fig(fig)

    # 4. BENFORD (POPRAWIONY - LINIA VS SŁUPKI)
    all_nums = []
    # Wykluczamy kolumny techniczne i wskaźnikowe, bierzemy tylko "duże" liczby finansowe
    exclude = ['Rok', 'F_Score', 'Z_Score_Val', 'ml_prob', 'ROE', 'ROA', 'ROS', 'Current_Ratio', 'Quick_Ratio', 'Debt_Ratio']
    
    for col in df.select_dtypes(include=[np.number]).columns:
        if col not in exclude:
            # Bierzemy tylko liczby większe niż 10 (żeby uniknąć małych wskaźników)
            vals = df[col].dropna().abs()
            vals = vals[vals > 10] 
            all_nums.extend(vals.astype(str).tolist())
            
    first_digits = [int(str(n)[0]) for n in all_nums if n and str(n)[0] in '123456789']
    
    if first_digits:
        counts = pd.Series(first_digits).value_counts(normalize=True).sort_index() * 100
        benford_theory = pd.Series({1:30.1, 2:17.6, 3:12.5, 4:9.7, 5:7.9, 6:6.7, 7:5.8, 8:5.1, 9:4.6})
        
        fig, ax = plt.subplots(figsize=(7, 3.5))
        fig.patch.set_facecolor(bg_color)
        setup_ax(ax, 'PRAWO BENFORDA (WYKRYWANIE ANOMALII)')
        
        # TEORIA JAKO LINIA (Żeby nie zasłaniała)
        ax.plot(benford_theory.index, benford_theory.values, color='#787b86', linestyle='--', marker='x', label='Teoria', linewidth=1.5, zorder=5)
        
        # DANE FIRMY JAKO SŁUPKI
        ax.bar(counts.index, counts.values, color=line_blue, alpha=0.8, label='Firma', zorder=2)
        
        ax.set_xticks(range(1,10))
        ax.legend(frameon=False, fontsize=8, labelcolor=text_color)
        charts['benford'] = save_fig(fig)
    else:
        # Jeśli za mało danych, pusty wykres
        fig, ax = plt.subplots(figsize=(7, 3.5))
        fig.patch.set_facecolor(bg_color); ax.set_facecolor(bg_color)
        ax.text(0.5, 0.5, "Zbyt mało danych do analizy Benforda", color='white', ha='center')
        charts['benford'] = save_fig(fig)

    return charts

# ======================================================
# 4. GŁÓWNA LOGIKA
# ======================================================
def analyze(ticker):
    t = ticker.upper().strip()
    search = f"{t}.WA" if len(t)<=4 and not t.endswith('.') else t
    
    try:
        stock = yf.Ticker(search)
        fin = stock.financials.T; bal = stock.balance_sheet.T; cf = stock.cashflow.T
        
        hist = stock.history(period='1d')
        if hist.empty: return None
        current_price = hist['Close'].iloc[-1]
        mcap = stock.info.get('marketCap', 0)
        
        if fin.empty: return None

        def g(df, keys):
            for k in keys:
                if k in df.columns: return pd.to_numeric(df[k], errors='coerce').groupby(pd.to_datetime(df[k].index).year).first()
            return None

        d = {}
        d['Przychody'] = g(fin, ['Total Revenue', 'Operating Revenue'])
        d['Zysk_Netto'] = g(fin, ['Net Income', 'Net Income Common Stockholders'])
        d['EBIT'] = g(fin, ['EBIT', 'Operating Income'])
        d['Odsetki'] = g(fin, ['Interest Expense'])
        d['Aktywa'] = g(bal, ['Total Assets'])
        d['Aktywa_Obrotowe'] = g(bal, ['Current Assets'])
        d['Zobowiazania'] = g(bal, ['Total Liabilities Net Minority Interest', 'Total Liabilities'])
        d['Zobowiazania_Krotkie'] = g(bal, ['Current Liabilities'])
        d['Kapital_Wlasny'] = g(bal, ['Stockholders Equity'])
        d['Zyski_Zatrzymane'] = g(bal, ['Retained Earnings'])
        d['Zapasy'] = g(bal, ['Inventory'])
        d['Cash_Flow'] = g(cf, ['Operating Cash Flow'])

        df = pd.DataFrame(d).sort_index(ascending=True).fillna(0)
        df['Rok'] = df.index.astype(int)
        df = df.replace(0, 0.001)

        df['ROE'] = (df['Zysk_Netto']/df['Kapital_Wlasny'])*100
        df['ROA'] = (df['Zysk_Netto']/df['Aktywa'])*100
        df['ROS'] = (df['Zysk_Netto']/df['Przychody'])*100
        df['Current_Ratio'] = df['Aktywa_Obrotowe']/df['Zobowiazania_Krotkie']
        df['Quick_Ratio'] = (df['Aktywa_Obrotowe']-df['Zapasy'])/df['Zobowiazania_Krotkie']
        df['Debt_Ratio'] = df['Zobowiazania']/df['Aktywa']
        
        df['Z_Score_Val'] = 6.56*((df['Aktywa_Obrotowe']-df['Zobowiazania_Krotkie'])/df['Aktywa']) + \
                            3.26*(df['Zyski_Zatrzymane']/df['Aktywa']) + \
                            6.72*(df['EBIT']/df['Aktywa']) + \
                            1.05*(df['Kapital_Wlasny']/df['Zobowiazania'])
        
        df['ROA_Prev'] = df['ROA'].shift(1).fillna(0)
        df['CFO_Prev'] = df['Cash_Flow'].shift(1).fillna(0)
        df['Lev_Prev'] = df['Debt_Ratio'].shift(1).fillna(0)
        df['Liq_Prev'] = df['Current_Ratio'].shift(1).fillna(0)
        df['Margin_Prev'] = df['ROS'].shift(1).fillna(0)
        def calc_f(row):
            s=0
            if row['ROA']>0: s+=1
            if row['Cash_Flow']>0: s+=1
            if row['ROA']>row['ROA_Prev']: s+=1
            if row['Cash_Flow']>row['Zysk_Netto']: s+=1
            if row['Debt_Ratio']<row['Lev_Prev']: s+=1
            if row['Current_Ratio']>row['Liq_Prev']: s+=1
            if row['ROS']>row['Margin_Prev']: s+=1
            return s
        df['F_Score'] = df.apply(calc_f, axis=1)

        latest = df.iloc[-1]
        dcf_data = calculate_dcf(stock, stock.cashflow, df)
        
        feats = [[latest['Current_Ratio'], latest['Debt_Ratio'], latest['ROA']/100, latest['ROS']/100]]
        try: ml_prob = ml_model.predict_proba(feats)[0][1] * 100
        except: ml_prob = 50.0

        charts = create_static_charts(df)

        return {
            'ticker': t,
            'current_price': round(current_price, 2),
            'mcap': mcap,
            'latest': latest.to_dict(),
            'history': df.sort_index(ascending=False).to_dict('records'),
            'ml_prob': ml_prob,
            'dcf': dcf_data,
            'charts': charts,
            'f_score': latest['F_Score']
        }

    except Exception as e:
        print(f"Błąd: {e}")
        return None

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    limit_reached = False
    
    if 'checks_today' not in session:
        session['checks_today'] = 0
        session['last_check_date'] = datetime.date.today().isoformat()
    
    if session['last_check_date'] != datetime.date.today().isoformat():
        session['checks_today'] = 0
        session['last_check_date'] = datetime.date.today().isoformat()

    if request.method == 'POST':
        if session['checks_today'] >= 500:
            limit_reached = True
        else:
            t = request.form.get('ticker')
            if t: 
                result = analyze(t)
                session['checks_today'] += 1
                session.modified = True

    return render_template('index.html', result=result, limit_reached=limit_reached, checks_used=session['checks_today'])

if __name__ == '__main__':
    app.run(debug=True, port=5001)