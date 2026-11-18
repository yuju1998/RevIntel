"""Index CRM text for account-scoped semantic search (rerunnable).

    PYTHONPATH=backend python -m app.cli.index_crm [--account-id UUID]

Uses DATABASE_URL and the EMBEDDING_* settings (environment or `.env`). Unchanged sources
are skipped, changed ones re-embedded, and chunks of deleted or re-parented sources
removed, so running it again is safe and cheap.
"""

import argparse
import sys
import time
import uuid
from collections.abc import Callable, Sequence

from sqlalchemy.orm import Session

from app.ai.embeddings import EmbeddingError, EmbeddingProvider, SentenceTransformerProvider
from app.core.config import get_database_settings, get_embedding_settings
from app.db.session import create_db_engine, create_session_factory
from app.repositories.crm_sources import CrmSourceRepository
from app.repositories.document_chunk import DocumentChunkRepository
from app.services.crm_indexing_service import CrmIndexingService


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Index CRM text for semantic search.")
    parser.add_argument("--account-id", type=uuid.UUID, help="index only this account")
    parser.add_argument("--page-size", type=int, default=500, help="sources per batch")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    session_factory: Callable[[], Session] | None = None,
    embeddings: EmbeddingProvider | None = None,
) -> int:
    args = _parser().parse_args(argv)
    if embeddings is None:
        settings = get_embedding_settings()
        embeddings = SentenceTransformerProvider(
            settings.embedding_model,
            settings.embedding_dimension,
            batch_size=settings.embedding_batch_size,
        )
    engine = None
    if session_factory is None:
        engine = create_db_engine(get_database_settings())
        session_factory = create_session_factory(engine)
    started = time.perf_counter()
    try:
        with session_factory() as session:
            stats = CrmIndexingService(
                session,
                CrmSourceRepository(session),
                DocumentChunkRepository(session),
                embeddings,
                page_size=args.page_size,
                progress=print,
            ).index(account_id=args.account_id)
    except EmbeddingError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        if engine is not None:
            engine.dispose()
    print(
        f"Embedded {stats.texts_embedded:,} distinct texts with {embeddings.model_name} "
        f"in {time.perf_counter() - started:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
