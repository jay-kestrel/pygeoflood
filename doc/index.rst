.. pygeoflood documentation master file, created by
   sphinx-quickstart on Fri May 24 10:07:38 2024.

**pygeoflood** docs
========================

**pygeoflood** is a suite of terrain-analysis tools for mapping flood
inundation in near-real time: the maintained successor of GeoFlood/GeoNet,
implementing Height Above Nearest Drainage (HAND) fluvial flood mapping,
channel extraction, and coastal and pluvial compound modes.

.. note::

   This site documents the **Kestrel fork** of pygeoflood
   (`jay-kestrel/pygeoflood`), which adds native Rust kernels, a CLI, a
   benchmark harness, and a golden-value equivalence suite on top of
   `upstream pygeoflood <https://github.com/passah2O/pygeoflood>`_.
   ``pip install pygeoflood`` installs *upstream's* release — see
   `Getting Started`_ for fork install instructions.

Getting Started
###############

Install (Python 3.10–3.12; 3.13 is not supported). For pluvial
(`fill_spill_merge`) mapping you additionally need the
`mdp0023/richdem fork <https://github.com/mdp0023/richdem>`_:

.. code-block:: bash

   # upstream's PyPI release — NOT this fork
   pip install pygeoflood

   # this fork, from source (needs a C linker and Rust 1.85.1+)
   pip install git+https://github.com/jay-kestrel/pygeoflood

   # richdem fork, only for fill_spill_merge (pluvial mode)
   git clone https://github.com/mdp0023/richdem.git richdem
   cd richdem/wrappers/pyrichdem
   pip install "setuptools<81"
   pip install "numpy==2.3.5"
   pip install --no-deps .

Then run the full fluvial flood-mapping workflow on your DEM:

.. code-block:: python

   from pygeoflood import pyGeoFlood, PGF_Config

   pgf = pyGeoFlood(
       dem_path="DEM.tif",
       config=PGF_Config({"calculate_flood_stage": {"custom_Q": 500}}),
   )
   pgf.run_fim_workflow()

Every stage prints its resolved parameters and writes deterministic outputs
under ``project_dir``. Three read-only CLI helpers come with the package:

.. code-block:: bash

   pygeoflood version         # package, Python, dependency, WBT versions
   pygeoflood list-options    # every method/option PGF_Config recognizes
   pygeoflood check-inputs DEM.tif [--flowline F.shp] [--catchment C.shp]

Background
##########

**Lineage.** pygeoflood descends from GeoNet (channel extraction from
lidar via nonlinear diffusion and geodesic paths) and GeoFlood (HAND-based
inundation mapping at scale), and keeps both citation lineages — see
`Acknowledgements`_ and the README.

**Method.** The fluvial workflow conditions the DEM with Perona–Malik
nonlinear diffusion, derives terrain attributes (slope, curvature), then
hands the gridded hydrology to the external
`WhiteboxTools <https://www.whiteboxgeo.com/>`_ binary (depression filling,
MFD flow accumulation, D8 flow direction, basin delineation). A flow
skeleton and endpoints yield the channel network; segmenting it and
computing Height Above Nearest Drainage gives the HAND raster; a
stage–rating curve per channel segment plus a streamflow forecast (or a
``custom_Q``) yields the inundation map.

**This fork.** The two hottest kernels — ``anisodiff`` and the ``process_cells``
SRC accumulation — are compiled Rust (``pygeoflood._rust``, PyO3 + Rayon,
abi3 wheels for Python 3.10–3.12). Every port is gated by a golden-value
equivalence suite, and benchmark runs record full environment manifests so
performance claims carry their context. The as-built details are in the
repository's `ARCHITECTURE.md
<https://github.com/jay-kestrel/pygeoflood/blob/main/ARCHITECTURE.md>`_.

User Guide
##########

**Workflows.** ``run_ce_workflow()`` runs channel extraction to
``extract_channel_network``; ``run_fim_workflow()`` continues through
``calculate_hand``, segmentation, catchment delineation, the stage–rating
curve, flood stage, and inundation. Each stage writes a deterministic file
(``{project_dir}/{dem-stem}_{product}.{ext}``), so workflows resume in
place. Two steps are bypassed by upstream design
(``calculate_geodesic_distance``, ``identify_channel_heads``).

**Options.** Pass a ``PGF_Config`` to the ``pyGeoFlood`` constructor.
Option resolution precedence is: explicitly passed arguments, then config
values, then signature defaults. ``pygeoflood list-options`` enumerates
every recognized ``(method, option)`` pair; unrecognized names warn and are
ignored. See the `API Reference`_ for per-method details.

**Performance.** Time the canonical workload with the benchmark harness
(``python -m benchmarks``; profiles ``crop`` and ``full``, cold/warm runs,
per-stage and per-WhiteboxTools timings, environment manifests) — protocol
and official baseline in ``benchmarks/README.md``.

**Correctness.** ``tests/test_golden_equivalence.py`` regenerates the
golden corpus and gates every output class (HAND and inundation rasters,
SRC and flood-stage CSVs, WBT integer rasters) against the pinned sha256
manifest; regeneration protocol in ``tests/golden/README.md``.

**Troubleshooting.**

- ``ImportError`` from ``fill_spill_merge`` → install the richdem fork
  (Getting Started above).
- WhiteboxTools version drift → point ``WBT_PATH`` at a pinned **v2.4.0**
  binary, the version the equivalence suite pins.
- ``import pygeoflood`` fails after editing ``rust/`` → editable installs
  do not recompile the extension; re-run ``pip install -e .``.
- Cargo cannot find the Python shared library → set ``LD_LIBRARY_PATH`` to
  Python's ``LIBDIR`` (see CONTRIBUTING.md).

Example workflows in python
###########################

.. toctree::
   :maxdepth: 2

   Example gallery <ex>

Contributing
#############

.. toctree::
   :maxdepth: 1

   Contributing bug reports and development <contribute>

Development setup and the full test matrix also live in the repository's
`CONTRIBUTING.md
<https://github.com/jay-kestrel/pygeoflood/blob/main/CONTRIBUTING.md>`_.

API Reference
#############

.. toctree::
   :maxdepth: 1

   pygeoflood API reference <apiref/apiref>


Acknowledgements
################

.. toctree::
   :maxdepth: 1

   Acknowledgements <acknowledgements>

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
