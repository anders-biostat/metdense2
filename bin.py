import json
import numpy as np
import pandas as pd
import numba
from matplotlib import pyplot as plt

from metdense2 import *
from hilbert import *

with open("scnmt_data__CpG_filtered.metdense/metadata.json") as f:
    info = json.load(f)

meta = pd.read_table("21-11-10_cell-annotation.tsv").set_index("cell_id_dna").reindex( info["cells"] )

cell_idcs_qNSC = np.where( meta["celltype"]=="qNSC" )[0]
cell_idcs_aNSC = np.where( meta["celltype"]=="oligodendrocyte" )[0]

md = MetdenseDataset( "scnmt_data__CpG_filtered.metdense" )
chrom = md.chroms["9"]

smoothed_bins = chrom.bin_smoothed( 200 )
vals_qNSC = chrom.bin_residual_sums( cell_idcs_qNSC, 200 )
vals_aNSC = chrom.bin_residual_sums( cell_idcs_aNSC, 200 )

#plt.scatter( smoothed_bins + vals_qNSC, smoothed_bins + vals_aNSC, s=.4 )
plt.imshow( vec2hilbert(vals_qNSC, 10 ) )
plt.show()