"""append-only audit_log table with immutability trigger

Revision ID: 0003
Revises: 0002
Create Date: 2026-07-09

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "audit_log",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        # no FK: audit entries must outlive the memories they describe (M4.6
        # deletes the memory; the record that it existed and was deleted stays)
        sa.Column("memory_id", UUID(as_uuid=True), nullable=False),
        sa.Column("action", sa.Text(), nullable=False),
        sa.Column("actor_type", sa.Text(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    # provenance lookups per memory (M4.5 surfaces these on retrieval)
    op.create_index("ix_audit_log_memory_id", "audit_log", ["memory_id"])

    # append-only by construction: row-level UPDATE/DELETE raise; TRUNCATE stays
    # possible as a deliberate admin action (and for test cleanup)
    op.execute(
        """
        CREATE FUNCTION audit_log_immutable() RETURNS trigger AS $$
        BEGIN
            RAISE EXCEPTION 'audit_log is append-only';
        END
        $$ LANGUAGE plpgsql
        """
    )
    op.execute(
        """
        CREATE TRIGGER audit_log_no_rewrite
        BEFORE UPDATE OR DELETE ON audit_log
        FOR EACH ROW EXECUTE FUNCTION audit_log_immutable()
        """
    )


def downgrade() -> None:
    op.drop_table("audit_log")
    op.execute("DROP FUNCTION audit_log_immutable()")
