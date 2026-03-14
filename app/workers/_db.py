"""Shared DB session helper for Celery workers."""

from contextlib import contextmanager


def get_worker_db():
    """Get a synchronous DB session for Celery worker use."""
    from app.db.session import get_sync_session
    return get_sync_session()


@contextmanager
def worker_db_session():
    """Context manager that yields a sync DB session and closes it."""
    db = get_worker_db()
    try:
        yield db
    finally:
        db.close()
