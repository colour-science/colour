Colour's Benchmark Suite
------------------------

.. image:: http://img.shields.io/badge/benchmarked%20by-asv-blue.svg?style=flat

This is a benchmark suite for the Colour package, the benchmarking is done using the Airspeed Velocity library and can be used to benchmark a commit, compare between commits and as a form of regression testing.

When adding a benchmark for a new module please create a new folder with its name, all files must be named as their counterpart in Colour's code.

Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Installation
^^^^^^^^^^^^

You'll need to have ASV installed to run these benchmarks and you'll also need ``virtualenv`` installed if you don't use conda.

``$ pip install asv``

or in conda environment:

``$ conda install -c conda-forge asv``

``$ pip install virtualenv`` *(optional)*

Running the benchmarks
~~~~~~~~~~~~~~~~~~~~~~

The conf files currently run on conda as a default and runs the benchmarks on your local copy of the Colour repository.

To get it to run on virtualenv, all you'll need is to change the ``environment_type`` value in the ``asv.conf.json`` to ``virtualenv``.

You can also run the benchmarks on Colour's github repository, to do that you'll need to change the ``repo`` value to the link of this repository and change the ``branches`` value to the branch you want to benchmark.

The benchmark results will be stored in your ``results/`` folder and can be changed in the conf file.

You can run the benchmarks using

``asv run``

to run all the benchmarks, you can also run specific benchmarks like this

``asv run --bench <file_name>.<class_name>``

For example

``asv run --bench benchmarks.isUniform``

The class name is an optional parameter and can be omitted (along with the .) if you need to run the whole file.


Comparing benchmarks between commits
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can run your benchmarks on specific commits using

``asv run -s 1 <commit_number>``

You can then compare the benchmark outputs between two commits using

``asv compare <commit1> <commit2>``

See `asv documentation <https://asv.readthedocs.io/en/stable/using.html>`__ for additional information on how to use the library.

Writing benchmarks
^^^^^^^^^^^^^^^^^^

To write benchmarks for Colour, please read `this <https://asv.readthedocs.io/en/stable/writing_benchmarks.html>`_ document on how to write benchmarks, benchmarks related to the same module should be in the same folder, and for each file in that module only one benchmark file should be created.

