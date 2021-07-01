'''
This is a simple file to clean generated python files from ipynb files using ipython nbconvert.
It basically just removes instances of get_ipython() in the python file.
'''

import os
import sys


def clean_ipynb_to_py_file(file_lines):
    '''
    There are certain things in the ipynb to py conversion that need
    to be removed for the py file to run.
    '''
    good_lines = []
    for i, line in enumerate(file_lines):
        if 'get_ipython()' in line:
            print(f"\t Found instance of get_ipython() on line {i}, dropping")
        else:
            good_lines.append(line)
    return good_lines


if __name__ == '__main__':
    folder = sys.argv[1]
    print(f"Cleaning {folder}")

    files = os.listdir(folder)
    for f in files:
        if f[-3:] != '.py':
            continue
        
        fp = os.path.join(folder, f)
        pyfile_r = open(fp, 'r')
        print(f"\t File: {f}")
        pylines_cleaned = clean_ipynb_to_py_file( pyfile_r.readlines() )
        pyfile_r.close()

        pyfile_w = open(fp, 'w')
        pyfile_w.writelines(pylines_cleaned)
        pyfile_w.close()

