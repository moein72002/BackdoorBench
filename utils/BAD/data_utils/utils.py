import subprocess
import stat, os

def run_download_bash_file(script_path):
    st = os.stat(script_path)
    os.chmod(script_path, st.st_mode | stat.S_IEXEC)
    subprocess.run(['bash', script_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    