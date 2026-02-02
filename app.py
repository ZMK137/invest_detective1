import matplotlib
matplotlib.use('Agg')  # Tryb serwerowy
import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, render_template, request, session, redirect, url_for, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from scipy.io import arff
import os
import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask_caching import Cache

app = Flask(__name__)
app.secret_key = 'TAJNY_KLUCZ_SESJI_INVEST_DETECTIVE' 

# ======================================================
# 1. KONFIGURACJA CACHE (PAMIĘĆ PODRĘCZNA)
# ======================================================
cache = Cache(app, config={'CACHE_TYPE': 'SimpleCache', 'CACHE_DEFAULT_TIMEOUT': 900})

# ======================================================
# 2. KONFIGURACJA BAZY DANYCH (SMART SWITCH)
# ======================================================
basedir = os.path.abspath(os.path.dirname(__file__))
# Lokalnie: users.db, Na serwerze: PostgreSQL
db_url = os.environ.get('DATABASE_URL', 'sqlite:///' + os.path.join(basedir, 'users.db'))

# Poprawka dla Rendera (wymagają postgresql:// zamiast postgres://)
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = db_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'home'

# ======================================================
# 3. MODEL UŻYTKOWNIKA
# ======================================================
class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    plan = db.Column(db.String(50), default='free')
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Inicjalizacja bazy (Bezpieczna dla Gunicorna)
with app.app_context():
    try:
        db.create_all()
    except Exception as e:
        pass # Jeśli inny worker już stworzył bazę, to ignorujemy błąd

# ======================================================
# 4. POMOCNICZE: SESJA
# ======================================================
def get_session_info():
    """Zwraca info o limitach (wyłączone dla wygody)"""
    if 'checks_today' not in session:
        session['checks_today'] = 0
    return False, session['checks_today']

# ======================================================
# 5. MODEL ML (AI)
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
# 6. MODEL DCF (WYCENA)
# ======================================================
def calculate_dcf(stock_obj, cashflow, df_history):
    try:
        if cashflow is None or cashflow.empty: return None
        ocf = cashflow.loc['Operating Cash Flow'].iloc[0] if 'Operating Cash Flow' in cashflow.index else 0
        capex = cashflow.loc['Capital Expenditure'].iloc[0] if 'Capital Expenditure' in cashflow.index else 0
        if ocf == 0: return None
        fcf = ocf + capex 
        
        shares = stock_obj.info.get('sharesOutstanding', None)
        if not shares: return None

        cagr = 0.03
        if len(df_history) >= 2:
            try:
                rev_now = df_history['Przychody'].iloc[0]
                rev_past = df_history['Przychody'].iloc[-1]
                years = len(df_history) - 1
                if years > 0 and rev_past > 0:
                    cagr = (rev_now / rev_past) ** (1/years) - 1
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
            'shares': shares
        }
    except Exception as e:
        return None

# ======================================================
# 7. WYKRESY
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

    try:
        # Z-Score
        fig, ax = plt.subplots(figsize=(7, 3.5))
        fig.patch.set_facecolor(bg_color)
        setup_ax(ax, 'ALTMAN Z-SCORE')
        z = df['Z_Score_Val'].tolist()
        ax.axhline(1.8, color='#f23645', linestyle='--', linewidth=1, alpha=0.7)
        ax.axhline(2.99, color='#089981', linestyle='--', linewidth=1, alpha=0.7)
        ax.plot(years, z, marker='o', color=line_blue, linewidth=1.5, markersize=4)
        charts['z_score'] = save_fig(fig)

        # Rentowność
        fig, ax = plt.subplots(figsize=(7, 3.5))
        fig.patch.set_facecolor(bg_color)
        setup_ax(ax, 'RENTOWNOŚĆ')
        ax.plot(years, df['ROE'], label='ROE', color=line_blue, linewidth=1.5)
        ax.plot(years, df['ROA'], label='ROA', color='#d1d4dc', linewidth=1.5, linestyle=':')
        ax.legend(frameon=False, fontsize=8, labelcolor=text_color)
        charts['prof'] = save_fig(fig)

        # Płynność
        fig, ax = plt.subplots(figsize=(7, 3.5))
        fig.patch.set_facecolor(bg_color)
        setup_ax(ax, 'PŁYNNOŚĆ')
        ax.fill_between(years, df['Current_Ratio'], color=line_blue, alpha=0.1)
        ax.plot(years, df['Current_Ratio'], color=line_blue, linewidth=1.5)
        ax.axhline(1.0, color='#f23645', linestyle='--', linewidth=1)
        charts['liq'] = save_fig(fig)

        # Benford
        all_nums = []
        exclude = ['Rok', 'F_Score', 'Z_Score_Val', 'ml_prob']
        for col in df.select_dtypes(include=[np.number]).columns:
            if col not in exclude:
                vals = df[col].dropna().abs()
                vals = vals[vals > 10]
                all_nums.extend(vals.astype(str).tolist())
        first_digits = [int(str(n)[0]) for n in all_nums if n and str(n)[0] in '123456789']
        
        fig, ax = plt.subplots(figsize=(7, 3.5))
        fig.patch.set_facecolor(bg_color)
        if first_digits:
            setup_ax(ax, 'PRAWO BENFORDA')
            counts = pd.Series(first_digits).value_counts(normalize=True).sort_index() * 100
            benford_theory = pd.Series({1:30.1, 2:17.6, 3:12.5, 4:9.7, 5:7.9, 6:6.7, 7:5.8, 8:5.1, 9:4.6})
            ax.plot(benford_theory.index, benford_theory.values, color='#787b86', linestyle='--', marker='x', label='Teoria')
            ax.bar(counts.index, counts.values, color=line_blue, alpha=0.8, label='Firma')
            ax.set_xticks(range(1,10))
            ax.legend(frameon=False, fontsize=8, labelcolor=text_color)
        else:
            setup_ax(ax, 'PRAWO BENFORDA (BRAK DANYCH)')
        charts['benford'] = save_fig(fig)
        
    except Exception as e:
        print(f"Błąd wykresów: {e}")
        plt.close('all')

    return charts

# ======================================================
# 8. GŁÓWNA ANALIZA (CACHED)
# ======================================================
def process_financial_data(data_frames, is_quarterly=False):
    """
    Przygotowuje dane finansowe, łącząc je w jeden DataFrame.
    Args:
        data_frames (dict): Słownik z DataFrame'ami (fin, bal, cf).
        is_quarterly (bool): Jeśli True, grupuje dane kwartalnie, inaczej rocznie.
    """
    d = {}
    
    def get_series(df, keys):
        for k in keys:
            if k in df.columns:
                series = pd.to_numeric(df[k], errors='coerce')
                # Grupowanie danych rocznie lub kwartalnie
                if is_quarterly:
                    # Używamy floor('Q') aby uzyskać początek kwartału
                    grouper = pd.to_datetime(series.index).to_period('Q')
                else:
                    grouper = pd.to_datetime(series.index).year
                return series.groupby(grouper).first()
        return None

    fin = data_frames.get('fin', pd.DataFrame())
    bal = data_frames.get('bal', pd.DataFrame())
    cf = data_frames.get('cf', pd.DataFrame())

    d['Przychody'] = get_series(fin, ['Total Revenue', 'Operating Revenue'])
    d['Zysk_Netto'] = get_series(fin, ['Net Income', 'Net Income Common Stockholders'])
    d['EBIT'] = get_series(fin, ['EBIT', 'Operating Income'])
    d['Odsetki'] = get_series(fin, ['Interest Expense'])
    d['Aktywa'] = get_series(bal, ['Total Assets'])
    d['Aktywa_Obrotowe'] = get_series(bal, ['Current Assets'])
    d['Zobowiazania'] = get_series(bal, ['Total Liabilities Net Minority Interest', 'Total Liabilities'])
    d['Zobowiazania_Krotkie'] = get_series(bal, ['Current Liabilities'])
    d['Kapital_Wlasny'] = get_series(bal, ['Stockholders Equity'])
    d['Zyski_Zatrzymane'] = get_series(bal, ['Retained Earnings'])
    d['Zapasy'] = get_series(bal, ['Inventory'])
    d['Cash_Flow'] = get_series(cf, ['Operating Cash Flow'])

    df = pd.DataFrame(d).sort_index(ascending=True).fillna(0)
    
    # Dla danych kwartalnych, konwertujemy PeriodIndex na string
    if is_quarterly and isinstance(df.index, pd.PeriodIndex):
        df.index = df.index.strftime('%Y-Q%q')
        
    return df

def calculate_indicators(df):
    """Oblicza wskaźniki finansowe na podstawie danych wejściowych."""
    if df.empty:
        return pd.DataFrame(), pd.Series()

    df_calc = df.copy()
    df_calc = df_calc.replace(0, 0.001) # Unikamy dzielenia przez zero

    # Wskaźniki
    df_calc['ROE'] = (df_calc['Zysk_Netto'] / df_calc['Kapital_Wlasny']) * 100
    df_calc['ROA'] = (df_calc['Zysk_Netto'] / df_calc['Aktywa']) * 100
    df_calc['ROS'] = (df_calc['Zysk_Netto'] / df_calc['Przychody']) * 100
    df_calc['Current_Ratio'] = df_calc['Aktywa_Obrotowe'] / df_calc['Zobowiazania_Krotkie']
    df_calc['Quick_Ratio'] = (df_calc['Aktywa_Obrotowe'] - df_calc['Zapasy']) / df_calc['Zobowiazania_Krotkie']
    df_calc['Debt_Ratio'] = df_calc['Zobowiazania'] / df_calc['Aktywa']
    
    # Z-Score (wersja dla firm niefinansowych, produkcyjnych)
    df_calc['Z_Score_Val'] = 6.56 * ((df_calc['Aktywa_Obrotowe'] - df_calc['Zobowiazania_Krotkie']) / df_calc['Aktywa']) + \
                             3.26 * (df_calc['Zyski_Zatrzymane'] / df_calc['Aktywa']) + \
                             6.72 * (df_calc['EBIT'] / df_calc['Aktywa']) + \
                             1.05 * (df_calc['Kapital_Wlasny'] / df_calc['Zobowiazania'])
    
    # F-Score
    df_calc['ROA_Prev'] = df_calc['ROA'].shift(1).fillna(0)
    df_calc['CFO_Prev'] = df_calc['Cash_Flow'].shift(1).fillna(0)
    df_calc['Lev_Prev'] = df_calc['Debt_Ratio'].shift(1).fillna(0)
    df_calc['Liq_Prev'] = df_calc['Current_Ratio'].shift(1).fillna(0)
    df_calc['Margin_Prev'] = df_calc['ROS'].shift(1).fillna(0)
    
    def calc_f(row):
        s = 0
        if row['ROA'] > 0: s += 1
        if row['Cash_Flow'] > 0: s += 1
        if row['ROA'] > row['ROA_Prev']: s += 1
        if row['Cash_Flow'] > row['Zysk_Netto']: s += 1
        if row['Debt_Ratio'] < row['Lev_Prev']: s += 1
        if row['Current_Ratio'] > row['Liq_Prev']: s += 1
        if row['ROS'] > row['Margin_Prev']: s += 1
        return s
    df_calc['F_Score'] = df_calc.apply(calc_f, axis=1)

    # Zwracamy pełną historię oraz ostatni wiersz (najnowsze dane)
    latest = df_calc.iloc[-1] if not df_calc.empty else pd.Series()
    return df_calc, latest

# ======================================================
# 8. GŁÓWNA ANALIZA (CACHED)
# ======================================================
@cache.memoize(timeout=900)
def analyze(ticker):
    t = ticker.upper().strip()
    search = f"{t}.WA" if len(t) <= 4 and not t.endswith('.') else t
    print(f"🔄 Analyzing {t} (Fresh run)...")
    
    try:
        stock = yf.Ticker(search)

        # Pobieranie danych rocznych i kwartalnych
        annual_data = {'fin': stock.financials.T, 'bal': stock.balance_sheet.T, 'cf': stock.cashflow.T}
        quarterly_data = {'fin': stock.quarterly_financials.T, 'bal': stock.quarterly_balance_sheet.T, 'cf': stock.quarterly_cashflow.T}

        # Poluzowanie warunku: jeśli brak danych finansowych, nie przerywamy, aby zwrócić stock.info
        # if annual_data['fin'].empty:
        #     return None
        
        # Przetwarzanie i obliczenia
        df_annual_raw = process_financial_data(annual_data, is_quarterly=False)
        df_quarterly_raw = process_financial_data(quarterly_data, is_quarterly=True)

        df_annual, latest_annual = calculate_indicators(df_annual_raw)
        df_quarterly, latest_quarterly = calculate_indicators(df_quarterly_raw)

        # Używamy najnowszych danych kwartalnych jako głównych "latest"
        latest = latest_quarterly if not latest_quarterly.empty else latest_annual

        # Pobieranie ceny i kapitalizacji
        try:
            hist = stock.history(period='1d')
            current_price = hist['Close'].iloc[-1] if not hist.empty else 0
            mcap = stock.info.get('marketCap', 0)
        except:
            current_price = 0
            mcap = 0

        # Model DCF (używa danych rocznych dla stabilności)
        dcf_data = calculate_dcf(stock, stock.cashflow, df_annual)
        
        # Predykcja ML (używa najnowszych dostępnych danych)
        ml_prob = 50.0
        if not latest.empty:
            feats = [[latest.get('Current_Ratio', 0), latest.get('Debt_Ratio', 0), latest.get('ROA', 0)/100, latest.get('ROS', 0)/100]]
            try:
                ml_prob = ml_model.predict_proba(feats)[0][1] * 100
            except:
                pass

        # Dodanie kolumn indeksowych (niezbędne dla szablonu i wykresów)
        if not df_annual.empty: df_annual['Rok'] = df_annual.index
        if not df_quarterly.empty: df_quarterly['index'] = df_quarterly.index

        # Wykresy (bazują na danych rocznych)
        charts = create_static_charts(df_annual) if not df_annual.empty else {}

        return {
            'ticker': t,
            'current_price': round(current_price, 2),
            'mcap': mcap,
            'latest': latest.to_dict() if not latest.empty else {},
            'history_annual': df_annual.sort_index(ascending=False).to_dict('records'),
            'history_quarterly': df_quarterly.sort_index(ascending=False).to_dict('records'),
            'ml_prob': ml_prob,
            'dcf': dcf_data,
            'charts': charts,
            'f_score': latest.get('F_Score', 0) if not latest.empty else 0,
            'raw_info': stock.info
        }

    except Exception as e:
        print(f"Error: {e}")
        return None

# ======================================================
# 9. ROUTING 
# ======================================================

@app.route('/register', methods=['POST'])
def register():
    email = request.form.get('email')
    password = request.form.get('password')
    
    # Sprawdzamy czy user istnieje
    user = User.query.filter_by(email=email).first()
    if user:
        # Błąd? Zostajemy na stronie, pokazujemy błąd
        return render_template('index.html', error="Taki email już istnieje!", limit_reached=False, checks_used=0)
    
    # Sukces? Tworzymy usera
    new_user = User(email=email, password=generate_password_hash(password, method='scrypt'))
    db.session.add(new_user)
    db.session.commit()
    
    login_user(new_user)
    
    # WAŻNE: Przekieruj na stronę główną (adres zmieni się na /)
    return redirect(url_for('home'))

@app.route('/login', methods=['POST'])
def login():
    email = request.form.get('email')
    password = request.form.get('password')
    
    user = User.query.filter_by(email=email).first()
    
    if user and check_password_hash(user.password, password):
        login_user(user)
        # WAŻNE: Sukces = Przekierowanie na czystą główną
        return redirect(url_for('home'))
    else:
        # Błąd = Zostajemy tu i wyświetlamy info
        return render_template('index.html', error="Błędny email lub hasło", limit_reached=False, checks_used=0)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

# Ścieżka dla strony głównej (Landing Page)
@app.route('/', methods=['GET', 'POST']) 
def home():
    if request.method == 'POST':
        ticker = request.form.get('ticker', '').upper()
        if not ticker:
            return render_template('mainpage.html')

        analysis_result = analyze(ticker)
        
        limit_reached, checks_used = get_session_info()
        session['checks_today'] = checks_used + 1

        if analysis_result:
            return render_template('index.html', 
                                   result=analysis_result, 
                                   checks_used=session['checks_today'], 
                                   limit_reached=limit_reached)
        else:
            return render_template('index.html', 
                                   error=f"Nie znaleziono danych dla tickera '{ticker}'. Sprawdź symbol i spróbuj ponownie.", 
                                   result=None,
                                   checks_used=checks_used, 
                                   limit_reached=limit_reached)
            
    return render_template('mainpage.html')


# Ścieżka dla sugestii wyszukiwarki
@app.route('/search_suggestions')
def search_suggestions():
    query = request.args.get('q', '').upper()
    if len(query) < 2:
        return jsonify(suggestions=[])
    
    # Lista testowa (później możesz ją rozbudować)
    all_tickers = [
        {'symbol': 'XTB.WA', 'name': 'XTB Spółka Akcyjna'},
        {'symbol': 'CDR.WA', 'name': 'CD Projekt'},
        {'symbol': 'PKO.WA', 'name': 'PKO BP'},
        {'symbol': 'AAPL', 'name': 'Apple Inc.'},
        {'symbol': 'TSLA', 'name': 'Tesla Inc.'}
    ]
    
    filtered = [s for s in all_tickers if query in s['symbol'] or query in s['name'].upper()]
    return jsonify(suggestions=filtered[:5])
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)