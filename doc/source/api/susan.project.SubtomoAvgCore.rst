susan.project.SubtomoAvgCore
============================

.. autoclass:: susan.project.SubtomoAvg.SubtomoAvgCore
   :show-inheritance:
   :no-members:

   .. warning::

      **Developer-only.**  ``SubtomoAvgCore`` defines the overridable
      pipeline steps that sit between project setup and output.  To
      customise a single step, subclass
      :class:`~susan.project.SubtomoAvg.SubtomoAvg` and override the
      relevant method rather than using this class directly.

   .. rubric:: Iteration Execution

   .. automethod:: setup_iteration
   .. automethod:: run_estimation
   .. automethod:: select_particles
   .. automethod:: run_reconstruction
   .. automethod:: run_postprocessing
   .. automethod:: run_iteration
