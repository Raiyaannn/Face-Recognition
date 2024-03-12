from cx_Freeze import setup, Executable

setup(
    name="Face Recognition App",
    version="1.0",
    description="Detects face and protects your privacy",
    executables=[Executable("appFace.py")],
)
