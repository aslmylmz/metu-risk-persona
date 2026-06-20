API Reference
=============

This page is generated automatically from the source docstrings.

Scoring engine
--------------

.. currentmodule:: scoring.bart

.. py:data:: COLOR_PROFILES

   Mapping of each balloon color to its risk tier and maximum pump capacity
   ``N``: ``purple`` (low, ``N = 128``), ``teal`` (medium, ``N = 32``), and
   ``orange`` (high, ``N = 8``). Shared throughout the engine.

.. py:data:: MIN_COLLECTED_FALLBACK
   :value: 2

   Minimum number of collected (non-exploded) balloons a color must have before
   the engine falls back to using all of that color's trials for its
   behavioral-intention metrics.

Public functions
^^^^^^^^^^^^^^^^^

.. autofunction:: score_bart

.. autofunction:: validate_bart_session

Internal computation helpers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The functions below are private (underscore-prefixed) implementation details.
They are documented here as a technical reference for the metric definitions;
they are not part of the public API and may change between releases.

.. automodule:: scoring.bart
   :members:
   :private-members:
   :exclude-members: score_bart, validate_bart_session, COLOR_PROFILES, MIN_COLLECTED_FALLBACK
   :undoc-members:
   :show-inheritance:

Data schemas
------------

.. automodule:: scoring.schemas
   :members:
   :show-inheritance:
   :member-order: bysource

Event validators
----------------

.. automodule:: scoring.schemas.game_events
   :members:
   :exclude-members: AssessmentResponse, BARTMetrics, ColorMetrics, EventPayload, GameEvent, GameSession, GameType, NormalizedScore
