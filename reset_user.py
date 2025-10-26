# reset_user.py
# one-time helper to (re)create a user with a properly hashed password

from docuchat_backend.app.database import SessionLocal
from docuchat_backend.app import models
from docuchat_backend.app.auth import get_password_hash

EMAIL = "umit.ipek@gmail.com"
PLAINTEXT_PASSWORD = "12345678"  # this is what you'll log in with

db = SessionLocal()

try:
    user = db.query(models.User).filter(models.User.email == EMAIL).first()
    if user:
        print(f"User {EMAIL} exists. Updating password hash.")
        user.password_hash = get_password_hash(PLAINTEXT_PASSWORD)
    else:
        print(f"User {EMAIL} not found. Creating new user.")
        user = models.User(
            email=EMAIL,
            password_hash=get_password_hash(PLAINTEXT_PASSWORD),
        )
        db.add(user)

    db.commit()
    print("Done. You can now log in with that email + password.")
finally:
    db.close()
