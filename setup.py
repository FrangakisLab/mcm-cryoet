from skbuild import setup 

setup(
    name="mcm-cryoet",
    version="1.0.0",
    description="Mean curvature motion for cryo-electron tomograms.",
    author='Achilleas S. Frangakis, Utz H. Ermel',
    license="GPLv3",
    packages=['pymcm'],
    python_requires=">=3.9",
    scripts=['scripts/mcm_levelset.py',
             'scripts/mcm_3D.py',
             'scripts/geodesic_trace.py',
             'scripts/mcm_open.py',
             'scripts/mcm_close.py'],
)