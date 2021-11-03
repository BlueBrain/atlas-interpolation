Vendors
=======
Some dependencies are not available as packages and therefore had to be
vendored. The vendoring is done using the
`py-vendor <https://pypi.org/project/py-vendor>`__ utility. It's installed
automatically together with the ``dev`` extras. You can also install it by hand
via ``pip install py-vendor==0.1.2``.

The vendoring is then done using the following command (add ``--force`` to
overwrite existing folders):

.. code-block:: shell

    py-vendor run --config py-vendor.yaml

See the ``py-vendor.yaml`` file for details on the vendor sources and files.
