"""Create the private monitoring schema with an operator-provided database URL."""
import os

from monitoring.settings import Settings
from monitoring.store import Store


def main():
    settings = Settings.from_env()
    url = os.getenv("MIGRATION_DATABASE_URL") or settings.database_url
    if not url:
        print("DATABASE_URL is not configured; monitoring remains unavailable.")
        return
    store = Store(url, migrate=True)
    assert store.ping()
    store.close()
    print("Monitoring schema version 1 is ready.")


if __name__ == "__main__":
    main()
