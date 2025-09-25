"""Legacy module retained for backwards compatibility.

The refactored project exposes all model builders via
``sentiment_analysis.models``.
"""
from __future__ import annotations

from sentiment_analysis.models.bert import BertEncoderClassifier

__all__ = ["BertEncoderClassifier"]
