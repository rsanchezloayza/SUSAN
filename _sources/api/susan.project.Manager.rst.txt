susan.project.STA
=================

.. autoclass:: susan.project.STA.STA
   :show-inheritance:
   :no-members:

   .. note::

      Retained for **backward compatibility**.  New projects should use
      :class:`~susan.project.SubtomoAvg.SubtomoAvg` (or
      :class:`~susan.project.SubtomoAvgSched.SubtomoAvgSched` /
      :class:`~susan.project.SubtomoAvgN2N.SubtomoAvgN2N`), which supersede
      this interface.

   .. rubric:: Main Method

   .. automethod:: execute_iteration

   .. rubric:: Step-by-step Execution

   .. automethod:: setup_iteration
   .. automethod:: exec_estimation
   .. automethod:: exec_particle_selection
   .. automethod:: exec_averaging
   .. automethod:: exec_postprocessing

   .. rubric:: Data Access

   .. automethod:: get_map
   .. automethod:: get_ptcls
   .. automethod:: get_cc
   .. automethod:: get_fsc
   .. automethod:: get_name_ptcls
   .. automethod:: get_name_refstxt
   .. automethod:: get_names_map
   .. automethod:: get_names_mask
   .. automethod:: get_names_halfmaps
   .. automethod:: get_iteration_dir
   .. automethod:: get_iteration_files
