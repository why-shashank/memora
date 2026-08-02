"""entities, alias index and memory_entities; audit_log can name an entity

Revision ID: 0004
Revises: 0003
Create Date: 2026-08-02

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "entities",
        sa.Column(
            "id",
            UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("type", sa.Text(), nullable=False),
        sa.Column("canonical_name", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )

    # PK (entity_type, alias_key) is the whole point: one entity per name, enforced by
    # the database so concurrent resolvers collide instead of creating duplicates —
    # and partitioned by type so a shared email can't fuse a person into their
    # employer (S5).
    op.create_table(
        "entity_aliases",
        sa.Column("entity_type", sa.Text(), primary_key=True),
        sa.Column("alias_key", sa.Text(), primary_key=True),
        sa.Column(
            "entity_id",
            UUID(as_uuid=True),
            sa.ForeignKey("entities.id", ondelete="CASCADE"),
            nullable=False,
        ),
    )
    op.create_index("ix_entity_aliases_entity_id", "entity_aliases", ["entity_id"])

    op.create_table(
        "memory_entities",
        sa.Column(
            "memory_id",
            UUID(as_uuid=True),
            sa.ForeignKey("memories.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "entity_id",
            UUID(as_uuid=True),
            sa.ForeignKey("entities.id", ondelete="CASCADE"),
            primary_key=True,
        ),
    )
    # M2.8's leg walks entity -> memories; M4.6 deletion-by-subject walks the same way
    op.create_index("ix_memory_entities_entity_id", "memory_entities", ["entity_id"])

    # An entity merge is a trust decision, so it belongs in the same append-only trail
    # as memory events rather than a parallel log. Existing rows all carry a memory_id,
    # so the CHECK validates without a rewrite; the immutability trigger is row-level
    # (BEFORE UPDATE OR DELETE) and does not fire on DDL.
    op.add_column("audit_log", sa.Column("entity_id", UUID(as_uuid=True), nullable=True))
    op.alter_column("audit_log", "memory_id", nullable=True)
    op.create_check_constraint(
        "ck_audit_log_names_a_subject",
        "audit_log",
        "(memory_id IS NULL) <> (entity_id IS NULL)",
    )
    op.create_index("ix_audit_log_entity_id", "audit_log", ["entity_id"])


def downgrade() -> None:
    op.drop_index("ix_audit_log_entity_id", table_name="audit_log")
    op.drop_constraint("ck_audit_log_names_a_subject", "audit_log", type_="check")
    # entity events have no memory_id and so cannot satisfy the restored NOT NULL.
    # Dropping them needs the append-only trigger out of the way — it blocks our own
    # DELETEs too, which is the point of it.
    op.execute("DROP TRIGGER audit_log_no_rewrite ON audit_log")
    op.execute("DELETE FROM audit_log WHERE memory_id IS NULL")
    op.alter_column("audit_log", "memory_id", nullable=False)
    op.execute(
        """
        CREATE TRIGGER audit_log_no_rewrite
        BEFORE UPDATE OR DELETE ON audit_log
        FOR EACH ROW EXECUTE FUNCTION audit_log_immutable()
        """
    )
    op.drop_column("audit_log", "entity_id")
    op.drop_table("memory_entities")
    op.drop_table("entity_aliases")
    op.drop_table("entities")
