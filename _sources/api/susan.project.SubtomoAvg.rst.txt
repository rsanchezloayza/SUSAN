susan.project.SubtomoAvg
========================

.. autoclass:: susan.project.SubtomoAvg.SubtomoAvg
   :show-inheritance:
   :no-members:

   .. rubric:: Iteration Execution

   .. automethod:: run_iteration
   .. automethod:: execute_iteration

   .. note::

      :meth:`execute_iteration` is an alias of :meth:`run_iteration`, kept
      for backward compatibility with :class:`~susan.project.STA.STA`.

   .. automethod:: setup_iteration
   .. automethod:: run_estimation
   .. automethod:: select_particles
   .. automethod:: run_reconstruction
   .. automethod:: run_postprocessing

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
   .. automethod:: get_ptcls
   .. automethod:: get_cc
   .. automethod:: get_fsc
   .. automethod:: map_change
