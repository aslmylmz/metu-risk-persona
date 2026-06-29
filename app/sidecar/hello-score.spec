# PyInstaller spec for the hello-score smoke binary.
#
# Freezes app/sidecar/hello_score.py (which imports the scipy-free `scoring`
# package) into a single executable. This is the SPEC §9/§18 de-risk artifact:
# proving numpy bundles and runs frozen on Windows before any UI is built.
#
# Build from anywhere:  pyinstaller app/sidecar/hello-score.spec
# Output:               dist/hello-score[.exe]

import os

from PyInstaller.utils.hooks import collect_submodules

# SPECPATH is injected by PyInstaller: the directory containing this spec.
script = os.path.join(SPECPATH, "hello_score.py")

# Pull in the whole scoring package (its submodules are imported dynamically
# enough that we list them explicitly rather than relying on graph discovery).
hiddenimports = collect_submodules("scoring")

a = Analysis(
    [script],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # The engine is scipy-free (issue 05); keep these heavy libs out of the binary.
    excludes=["scipy", "matplotlib", "pandas"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="hello-score",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
