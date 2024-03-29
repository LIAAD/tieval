tieval
------

.. toctree::
   :maxdepth: 2
   :titlesonly:
   :hidden:

   tieval
   hands_on

A framework for the development of temporally aware models.

.. image:: ../imgs/tieval.png


Installation
============

The package is available on PyPI:

.. code-block:: bash

   pip install tieval

.. note::
    ``tieval`` requires Python 3.8 or above.

Usage
=====

To understand its usability refer to the notebooks available in :doc:`hands_on` section.


Data
====

Throughout the last two decades many datasets have been developed to train this task.
text2timeline provides an easy interface to download the available corpus.

To know more about the module run the following code on the terminal.

.. code-block:: bash

   python -m tieval download --help

How to ...
==========

In this section we summarize how to perform the most useful operations in text2timeline.

download a dataset.
+++++++++++++++++++
.. code-block:: python

   from tieval import datasets
   datasets.download("TimeBank")


load a dataset.
+++++++++++++++
.. code-block:: python

   from tieval import datasets
   te3 = datasets.read("TempEval-3")

load a model.
+++++++++++++
.. code-block:: python

    from tieval import models
    heideltime = models.identification.HeidelTime()

make predictions.
+++++++++++++++++
.. code-block:: python

   predictions = heideltime.predict(te3.test)

evaluate predictions.
+++++++++++++++++++++
.. code-block:: python

   from tieval import evaluate
   evaluator = evaluate.Evaluator(te3.test)
   result = evaluator.timex_identification(predictions)

Contributing
============

1. Fork github repository_
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request

Meta
====

Hugo Sousa - hugo.o.sousa@inesctec.pt

This framework is part of the Text2Story_ project which is financed by the ERDF – European Regional Development Fund through the North Portugal Regional Operational Programme (NORTE 2020), under the PORTUGAL 2020 and by National Funds through the Portuguese funding agency, FCT - Fundação para a Ciência e a Tecnologia within project PTDC/CCI-COM/31857/2017 (NORTE-01-0145-FEDER-03185)

.. _Text2Story: https://text2story.inesctec.pt/
.. _repository: https://github.com/LIAAD/tieval
.. _PyPI:
