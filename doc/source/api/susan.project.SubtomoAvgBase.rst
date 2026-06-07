susan.project.SubtomoAvgBase
============================

.. autoclass:: susan.project.SubtomoAvg.SubtomoAvgBase
   :show-inheritance:
   :no-members:

   .. warning::

      **Developer-only.**  ``SubtomoAvgBase`` is internal infrastructure
      (project paths and read-only queries) shared by the public classes.
      Application code should use
      :class:`~susan.project.SubtomoAvg.SubtomoAvg` to run a project or
      :class:`~susan.project.SubtomoAvg.SubtomoAvgMonitor` to inspect one,
      not this base class directly.

   .. rubric:: Path Helpers

   .. automethod:: iteration_dir
   .. automethod:: iteration_files
   .. automethod:: path_map
   .. automethod:: path_halfmap
   .. automethod:: path_mask
   .. automethod:: path_refstxt
   .. automethod:: path_ptcls
   .. automethod:: path_map_rec

   .. rubric:: Data Access

   .. automethod:: get_map
   .. automethod:: get_mask
   .. automethod:: get_ptcls
   .. automethod:: get_cc
   .. automethod:: get_fsc
   .. automethod:: map_change
