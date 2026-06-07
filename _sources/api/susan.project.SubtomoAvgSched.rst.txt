susan.project.SubtomoAvgSched
=============================

.. warning::

   **Experimental / in development.**  The scheduler protocol, built-in
   schedulers in :mod:`susan.project.Schedulers`, and the factory presets
   (``make_*_refinement``) may change between releases.  Use
   :class:`~susan.project.SubtomoAvg.SubtomoAvg` directly for stable
   manual loops.

.. autoclass:: susan.project.SubtomoAvgSched.SubtomoAvgSched
   :show-inheritance:
   :no-members:

   .. rubric:: Automated Loop

   .. automethod:: run

   .. rubric:: Factory Presets

   .. automethod:: make_3d_refinement
   .. automethod:: make_2d_refinement
   .. automethod:: make_mixed_ctf_2d
