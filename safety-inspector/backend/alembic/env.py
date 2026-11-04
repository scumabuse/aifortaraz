from alembic import context
from app.database import Base, engine
from app import models  # noqa: F401

target_metadata = Base.metadata

if context.is_offline_mode():
    context.configure(
        url=str(engine.url), target_metadata=target_metadata, literal_binds=True
    )
    with context.begin_transaction():
        context.run_migrations()
else:
    with engine.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata, compare_type=True
        )
        with context.begin_transaction():
            context.run_migrations()
