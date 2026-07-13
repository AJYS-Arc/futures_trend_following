
from dotenv import load_dotenv
import os
import databento as db 


def _load_key() -> str:
    try:
        load_dotenv()
    except ImportError:
        pass  # dotenv is optional; env var may be set another way
    key = os.environ.get("DATABENTO_API_KEY")

    if not key:
        raise RuntimeError(
            "DATABENTO_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return key


def get_client():
    return db.Historical(_load_key())