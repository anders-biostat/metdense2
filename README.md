Here, I am developing code to handle single-cell bisulfite-sequencing data, using a data fromat that might be an alternative to the one we use in [MethSCAn](https://github.com/anders-biostat/MethSCAn).

#### Files

The script `convert.py` converts the data format produced by MethSCAn to this new format.

The `metdense2.py` file defines two classes to read this data.

#### Data format

The data format is as follows:
- All data is provided in a directory with several files.
- There is a file `metadata.json` that contains the following
  - a string named `format` with content `metdense`
  - a string named `format-version`. For now, I put there "0.0.-1" to indicate that the format is not finalized yet.
  - a string named `genome` that may contain information on the genome assembly used (e.g., `GRCm38`)
  - a vector of strings with the identifiers of the cells (one name per cell). Wherever cells are indexed, these indices are understood as referring to this string vector.
  - a dictionary with the names of all the chromosomes that appear in the reference, with their length
- For each chromosome, there are three files:
  - The file `nn.pos` (where `nn` is the chromosome name) is a binary file with the positions of all CpG sites that appear in the chromosome (possibly skipping those which
    have no coverage in any chromosome). The file is a vector of unsigned 32-bit integers, i.e., it can be opened with, e.g., `numpy.memmap( "nn.pos", numpy.uint32 )`. The
    number of 32-bit integers in this file (i.e., its size in bytes divided by 4) is called $n_\text{sites}$ below.
  - The file `nn.dat` with the actual methylation calls. This file contains $n_\text{cells} \cdot \lceil n_\text{sites} / 16 \rceil$ 32-bit integers (where $\lceil\cdot\rceil$ is the ceiling operation), i.e., $\lceil n_\text{sites} / 16 \rceil$ 32-bit integers per CpG site. These contain the methylation calls for the cell, each 32-bit word containing calls for 16 cells, starting with the calls for cell 0 to cell 15 in the first word. The two least signficant bits contain the call for cell 0, etc. 
      - The calls are 0 for "no call" (i.e., no coverage), 1 for "unmethylated", 2 for "methylated", and 3 for "ambiguous" (which means that there were mutiple reads that did not agree).
  - The file `nn.smd` conatins smoothed data for teh chromosome (TO DO: Explain.)

#### Classes

The class `MetDenseDataset` holds memmaps to a directory with files as just described. Given an object `md`, use, e.g., `md["X",176, 23]` to get the call for chromosome `X`, CpG site 176, cell 23.

The object also contains a dictionary `chroms` that contain objects of type `Chromosome`.

These have a method `get`, which is used by the above access, which can hence also be written `md.chroms["X"].get( 176, 23 )`. The 0-based basepair position of that CpG site can be obtained by `md.chroms["X"].posv[176]`.

The `Chromosome` class is a numba jitclass, allowing for fast code using numba.
