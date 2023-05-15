import subprocess

def run_r_script():
    command = 'Rscript'
    path2script = '~/evorobotpy2/script.R'
    cmd = [command, path2script]

    with open('output.txt', 'w') as f:
        x = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        output, error = x.communicate()

    print('Output:', output)
    print('Error:', error)
