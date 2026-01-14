from glob import glob
from os.path import basename

# get all the py file in the current directory
py_files = glob("*.py")

# filter out this file (get the name from variable __file__)
py_files = [f for f in py_files if f != basename(__file__)]

# create a latex document listing the files named file_list.tex
with open("file_list.tex", "w") as f:
    f.write(r"\newcommand{\filelist}{" + "%\n")
    f.write(",%\n".join(py_files) + "%\n")
    f.write("}%\n")