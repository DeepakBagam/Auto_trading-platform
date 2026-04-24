from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from db.models import Base
from utils.config import get_settings

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Target the application's declarative metadata so autogenerate works.
target_metadata = Base.metadata

# ---------------------------------------------------------------------------
# Override the sqlalchemy.url from the application settings so we never have
# to hard-code credentials in alembic.ini.
# ---------------------------------------------------------------------------
_settings = get_settings()
_database_url = _settings.database_url

config.set_main_option("sqlalchemy.url", _database_url)

# ---------------------------------------------------------------------------
# SQLite-specific connect args
# ---------------------------------------------------------------------------
_is_sqlite = _database_url.startswith("sqlite")


def _get_connect_args() -> dict:
    if _is_sqlite:
        return {"check_same_thread": False}
    return {}


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL and not an Engine, though an
    Engine is acceptable here as well.  By skipping the Engine creation we
    don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the script output.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine and associate a connection
    with the context.
    """
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = _database_url

    connect_args = _get_connect_args()

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        connect_args=connect_args,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
            # SQLite requires batch mode to ALTER TABLE (rename, drop column, etc.)
            render_as_batch=_is_sqlite,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
