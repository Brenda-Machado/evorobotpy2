import subprocess

def run_r_script():
    command = 'Rscript'
    path2script = 'path/to/script.R'
    cmd = [command, path2script]

    x = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = x.communicate()

    print('Output:', output)
    print('Error:', error)

run_r_script()