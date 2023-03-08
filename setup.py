from skbuild import setup 

setup(
    name="mcm-cryoet",
    version="1.0.0",
    description="MCM",
    author='Achilleas Frangakis, Utz Heinrich Ermel',
    license="GPLv3",
    packages=['pymcm'],
    python_requires=">=3.9",
    scripts=['scripts/mcm_levelset.py', 'scripts/mcm_3D.py', 'scripts/geodesic_trace.py'],
    data_files=[('bin', 'mcm_3D', 'mcm_3D_cuda', 'mcm_levelset', 'mcm_levelset_cuda', 'geodesic_trace', 'geodesic_trace_cuda')]
)