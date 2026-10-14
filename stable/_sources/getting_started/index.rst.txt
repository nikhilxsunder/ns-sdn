.. meta::
   :description: Install ns-sdn, fit a first spectral decomposition, and learn the model-fit-result pattern the library follows.

Getting started
===============

ns-sdn is one model behind one pattern. :class:`~ns_sdn.spectral_decomposition.NSSDN`
holds the series and the specification, ``fit()`` trains the network, and
the :class:`~ns_sdn.spectral_decomposition.NSSDNResult` it returns is a
frozen record of everything the fit is: the decomposition as arrays, the
one-step fit and its residuals, the trained network behind a numpy
interface, and a summary that states what the numbers assume. Learn the
pattern once and the whole library reads the same way; readers of
:mod:`cultivars` already know it.

The three pages below take about fifteen minutes in order. Readers who
know the model from the papers can start at the quickstart; readers who
want to know why the library is shaped the way it is can start at the
concepts.

.. grid:: 1 1 3 3
   :gutter: 3

   .. grid-item-card:: Installation
      :link: installation
      :link-type: doc

      Supported Pythons, the torch dependency, optional extras, pinning,
      and a development checkout.

   .. grid-item-card:: Quickstart
      :link: quickstart
      :link-type: doc

      Fit a decomposition to a series with a known answer, read every
      path, run the one-step and recursive protocols, compute the
      spectral state, compare two fits.

   .. grid-item-card:: Core concepts
      :link: concepts
      :link-type: doc

      Models, results, summaries, the two alignments, the training
      schedule, data conventions, and the error hierarchy.

.. toctree::
   :hidden:
   :maxdepth: 1

   installation
   quickstart
   concepts

Where next
----------

After these pages, the :doc:`user guide <../user_guide/index>` derives
the model and explains how to read what it emits, the
:doc:`examples <../auto_examples/index>` run the published DGS10
analysis on live FRED data, and the :doc:`API reference <../api/index>`
documents every public class with its mathematics and a runnable example.
