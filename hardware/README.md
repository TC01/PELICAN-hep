# PELICAN on Hardware

This document covers the use of hardware inference tools to deploy a PELICAN
neural network onto hardware, e.g. a Xilinx FPGA.

# Using Vitis AI

These instructions assume the use of a pre-trained PELICAN model, and therefore
that there is no reason/requirement to use the GPU docker containers. There seem
to be some differences between the pre-built pytorch CPU container and what you
get from following the Vitis AI build instructions, so I recommend just pulling
the latest CPU container from dockerhub.

## Setup

The Vitis AI github repo can be fetched using:

```
$ git clone https://github.com/Xilinx/Vitis-AI.git
$ cd Vitis-AI
```

You can then run the pytorch CPU docker container using the following:

```
./docker_run.sh xilinx/vitis-ai-pytorch-cpu:latest
```

However, you don't actually *need* the Vitis AI git repo if you just want to use
the existing docker containers-- you can just fetch the container directly. Below
are some notes for running this using singularity.

### Singularity vs Docker

[Singularity/Apptainer](https://apptainer.org/) is a docker-compatible containerization
framework widely used in high energy physics on clusters. Compared to docker, the
security model is a bit different-- singularity containers automatically have access
to a user's `/home` directory, and root access is not required to run containers. Note
that "singularity" was the original name of the project but due to a fork there are
now two competing (but cross-compatible) implementations, "apptainer" and "singularity".
(The "singularity" command should be available regardless of which one you are using).

These instructions may be useful if, for whatever reason, you want to run one of
the Vitis AI containers under singularity rather than docker-- perhaps because you're
working on a cluster or analysis facility, or perhaps because you have singularity
installed locally and don't want to set up docker, or perhaps because you don't have
root access.

First: singularity can run a docker container, but it will do this by downloading
the docker image and converting the format. This requires unpacking the docker image,
which by default is done under `/tmp`. The Vitis AI images are quite large, and `/tmp`
may not have enough space depending on how much RAM is available on the local system.
If you get an error when running `singularity build` you should do the following first:

```
sudo mkdir -p /opt/singularity/
export SINGULARITY_TMPDIR=/opt/singularity/
```

Then, you can fetch and build the container:

```
singularity pull docker://xilinx/vitis-ai-pytorch-cpu:latest
singularity build --sandbox --fakeroot vitis-ai-pytorch.sif docker://xilinx/vitis-ai-pytorch-cpu:latest
```

The `--sandbox` argument will generate a *directory* called `vitis-ai-pytorch.sif/` in the
local folder that contains the container's filesystem, and allow you to make edits (both
from inside and outside the container); if you don't pass `--sandbox` this won't be possible,
the image will be read-only.

The script that initializes Vitis AI (and will be executed when you run the container) is stored
inside the SIF at:

```
vitis-ai-pytorch.sif/etc/bash.bashrc
```

I found that it was necessary to edit this file and comment out the line beginning with `sudo chown`;
otherwise you'll get an error.

You can then start things up by simply running:

```
singularity run --writable vitis-ai-pytorch.sif
```

You'll get the usual "Vitis AI" activation message. You can then activate the conda environment:

```
Apptainer> conda activate vitis-ai-pytorch
(vitis-ai-pytorch) Apptainer>
```

If you `ls`, you'll notice that you have full access to your home directory (as opposed to
the docker setup).

One advantage of this approach is that, if you find yourself needing/wanting to patch the
Vitis AI code, it is easy to do so-- see the next section.

### Patching Vitis AI

In order to run the Vitis AI model inspector over PELICAN without changing PELICAN itself,
I found it was necessary to patch the inspector in a few places.

This is *very fragile*, and it would be nice to submit these changes upstream, but for
the time being you can apply them to the `vitis-ai-pytorch-cpu:latest` container image
directly using the following (assuming the Singularity installation):

```
cd vitis-ai-pytorch.sif/
patch -p1 < ../vitis-ai-inspector.patch
```

The patch does a few things:

* Enable support for dictionary arguments to PyTorch models when running the inspector.
* `raise` the full traceback when the inspector crashes for debugging purposes.
* Handle the case when the inspector encounters a node has no parents instead of just
assuming that they're there (still unsure how this can happen).

Assuming you're using the "latest" container for Vitis AI 3.5 the patch should apply.
If it doesn't, you can look at the diff and make the changes yourself. If you modify
PELICAN to not use a dictionary but instead a list of tuple of tensors, and pre-process
the input data accordingly, this entire step is hopefully unnecessary.

### Python 3.8 vs Python 3.9

The existing Vitis AI containers use Python 3.8. However, PELICAN is only compatible
with Python 3.9 and above. On close inspection it seems there are only two actual
Python 3.9 features that need to be dealt with:

* The use of some type hinting in `src/models/lorentz_metric.py`; I fixed this by
adding a `from __future__ import annotations` to the top of this file.
* The extensive use of the `argparse.BooleanOptionalAction` action in
`src/trainer/args.py`, which is only in python 3.9.

The easiest way to work around this latter issue that I found was to simply fetch
a copy of [Python 3.9's argparse](https://github.com/python/cpython/blob/3.9/Lib/argparse.py)
and import that. In principle we could bundle this here, for now I haven't done that.
Instead, fetch the source file from the Python git repo using (something like) the
following command:

```
wget -O src/argparse9.py https://raw.githubusercontent.com/python/cpython/3.9/Lib/argparse.py
```

Then I modified the relevant imports accordingly:

```
import sys

if sys.version_info.minor <= 8:
    from src import argparse9 as argparse
else:
    import argparse
```

Of course, we could also solve this problem by updating the containers, or by installing
the Vitis AI packages "manually" on a newer Python.

## Quantization and Inspection

The `quantize_pelican.py` script takes the same arguments as the training script. For
consistency it should be launched using the same arguments used to build the pre-trained
version of the model. For the `'515_100p-b_best.pt'` PyTorch file, that would be:

```
./quantize_pelican.py --device cpu --datadir=data/sample_data/run12/ \
--nobj=80 --nobj-avg=49 \
--num-epoch=35 --num-train=-1 --num-valid=60000 --batch-size=250 --prefix=515_100p-b \
--optim=adamw --lr-decay-type=warm --activation=leakyrelu --no-factorize --masked \
--scale=1 --lr-init=0.0025 --lr-final=1e-6 --drop-rate=0.025 --drop-rate-out=0.025 \
--weight-decay=0.005 --reproducible --no-fix-data --no-summarize --config=M \
--config-out=M --no-mlp-out --no-add-beams --no-activate-agg-out
```

It would almost certainly be possible to *not* require all of these arguments (the
arguments also seem to be stored as part of the pytorch file)-- but for the moment
we're still using the argument parser from the training script.

The script requires the pre-trained PyTorch model file to be located in the
current working directory. For now, the name of that file is hardcoded at the top
of the script as:

```
torchfile = '515_100p-b_best.pt'
```

The script will:

* Instantiate the model, and load the model state dictionary from the pytorch file.
* Set up an instance of the model inspector, using `DPUCVDX8G_ISA3_C32B6` as the DPU
(note: I'm not *totally* sure this is the right version... this is definitely a Versal
DPU, though) and run it over the model.
* Run the quantizer.
* Run the test suite over the quantized model.
* Attempt to write out the results in various formats (this step doesn't seem to work
at the moment).

# Using Other Tools

TODO: add information here about non-Vitis AI approaches, if we ever get round
to them.

# Resources:

* [CERN SoC workshop talk](https://indico.cern.ch/event/1253805/#33-xilinx-versal-acapsoc-for-r): many
more useful links included in these slides.
