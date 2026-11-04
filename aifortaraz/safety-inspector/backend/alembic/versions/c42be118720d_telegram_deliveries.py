"""Durable Telegram delivery queue."""

from alembic import op
import sqlalchemy as sa

revision = "c42be118720d"
down_revision = "81dea44abe81"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "telegram_deliveries",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "incident_id",
            sa.String(36),
            sa.ForeignKey("incidents.id", ondelete="CASCADE"),
        ),
        sa.Column("camera_id", sa.String(36)),
        sa.Column("type_code", sa.String(50)),
        sa.Column("recipient", sa.String(32), nullable=False),
        sa.Column("label", sa.String(100), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("error", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("next_attempt_at", sa.DateTime(timezone=True)),
        sa.Column("sent_at", sa.DateTime(timezone=True)),
    )
    op.create_index(
        "ix_telegram_due", "telegram_deliveries", ["status", "next_attempt_at"]
    )
    op.create_index(
        "ix_telegram_cooldown",
        "telegram_deliveries",
        ["camera_id", "type_code", "recipient", "created_at"],
    )


def downgrade():
    op.drop_table("telegram_deliveries")
