from pathlib import Path

for f in Path(".").glob("*x4.*"):
    new_name = f.name.replace("x4.", ".")
    f.rename(f.with_name(new_name))
    print(f"{f.name} -> {new_name}")