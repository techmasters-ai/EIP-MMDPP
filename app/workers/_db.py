"""Shared DB session helper for Celery workers."""


def get_worker_db():
    """Get a synchronous DB session for Celery worker use."""
    from app.db.session import get_sync_session
    return get_sync_session()
