"""Processors module exports.

Only the GA subject-matching grammar is left here, and only as an interim step: matching
resolutions to thesaurus subjects belongs in the ingestion job, which sees the raw repeated
MARC `991.d` fields. See T11 in plans/app_postgres_migration_plan.md.
"""

from .ga_processor import GAResolutionProcessor

__all__ = ['GAResolutionProcessor']
