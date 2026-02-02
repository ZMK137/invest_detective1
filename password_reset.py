from app import app, db, User
from werkzeug.security import generate_password_hash

# 1. Konfiguracja
TARGET_EMAIL = "szymonziem03@gmail.com" 
NEW_PASSWORD = "Lubiecycki"             

# 2. Operacja na bazie
with app.app_context():
    user = User.query.filter_by(email=TARGET_EMAIL).first()
    
    if user:
        print(f"Znaleziono użytkownika: {user.email}")
        # Nadpisujemy stare hasło nowym hashem
        user.password = generate_password_hash(NEW_PASSWORD, method='scrypt')
        db.session.commit()
        print(f"✅ SUKCES! Twoje nowe hasło to: {NEW_PASSWORD}")
    else:
        print(f"❌ BŁĄD: Nie znaleziono użytkownika o emailu {TARGET_EMAIL}")