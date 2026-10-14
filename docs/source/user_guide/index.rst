.. meta::
   :description: The ns-sdn user guide: the model in its equations, how to read the decomposition it emits, the two forecast protocols, the spectral-state coordinates, the data conventions, and what to be careful about.

User guide
==========

The user guide explains how to do spectral decomposition with ns-sdn,
one question at a time. Each page gives the relevant part of the model
in its standard notation, runs a verified example, and ends with the
nuances that are easy to get wrong. The :doc:`API reference <../api/index>`
has the exhaustive argument lists; the guide is where to learn what each
path means and when to trust it.

New to the library? Read :doc:`../getting_started/quickstart` first,
then :doc:`model` and :doc:`decomposition`. Those two pages explain what
the network computes and what the result hands back, which every other
page assumes.

Find the right page
-------------------

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - Question
     - Page
   * - What does the network compute, head by head, and how is it trained?
     - :doc:`model`
   * - What is each path on the result, and how do they add up?
     - :doc:`decomposition`
   * - One-step versus recursive, filtered versus predicted: which number means what?
     - :doc:`forecasting`
   * - How do I summarize sixteen components, and test a regime-change claim?
     - :doc:`spectral_state`
   * - What shape do inputs take? How are returned arrays indexed and scaled?
     - :doc:`conventions`
   * - What behaves differently from what the papers lead me to expect?
     - :doc:`nuances`

.. toctree::
   :caption: The model
   :maxdepth: 1

   model
   decomposition

.. toctree::
   :caption: Using a fit
   :maxdepth: 1

   forecasting
   spectral_state

.. toctree::
   :caption: Foundations
   :maxdepth: 1

   conventions
   nuances
