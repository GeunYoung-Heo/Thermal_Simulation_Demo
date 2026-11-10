"""Thermal Digital Twin demo.

Pipeline (each stage is a module):
    scene           virtual "real world" warehouse (ground truth, used only to fake sensors)
    scanner         simulated panoramic RGB-D scan  -> point cloud
    segmentation    simulated SAM3 masks -> multi-view voting -> 3D instances with IDs
    reconstruction  TSDF fusion + marching cubes -> messy, labelled mesh
    twin            reconstruction -> editable thermal model description (devices / BCs / zone)
    thermal         voxel conduction solver (steady state)
    optimize        device placement search + operating temperature LP
    io              save / load pipeline outputs and simulation results
    viz             plotly figures used by the Streamlit app
"""

__version__ = "0.1.0"
