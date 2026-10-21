"""Fail-closed sandbox: Linux seccomp subprocess or hardened Docker container."""
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

WORKER=Path(__file__).with_name('sandbox_worker.py')


def extract_code(text):
    blocks=re.findall(r'```(?:python|py)?\s*\n(.*?)```',text,re.S)
    return '\n\n'.join(blocks).strip() if blocks else text.strip()


def check_backend(backend):
    # Run an actual success AND a forbidden filesystem probe, never silently fall back.
    ok=execute('def f(): return 2',['assert f()==2'],backend=backend)
    forbidden=execute("open('/etc/passwd').read()",['assert True'],backend=backend)
    if not ok['passed'] or forbidden['passed']:
        raise RuntimeError(f'Sandbox unavailable or ineffective: success={ok}, filesystem probe={forbidden}')


def execute(code,tests,setup='',backend='seccomp',timeout=5):
    marker='PASS_'+uuid.uuid4().hex
    payload=dict(code=code,tests=tests,setup=setup,marker=marker,seccomp=backend=='seccomp')
    if backend=='seccomp':
        if sys.platform!='linux': raise RuntimeError('Linux required for seccomp; use --sandbox docker')
        command=[sys.executable,'-I','-S',str(WORKER)]
    elif backend=='docker':
        container_name='rl-eval-'+uuid.uuid4().hex
        command=['docker','run','--name',container_name,'--rm','--network=none','--read-only','--cap-drop=ALL',
                 '--security-opt=no-new-privileges','--pids-limit=16','--memory=512m','--cpus=1',
                 '--user=65534:65534','--tmpfs','/tmp:rw,noexec,nosuid,size=16m','-i',
                 '-v',f'{WORKER.resolve()}:/worker.py:ro','rl-code-sandbox:local','python','-I','-S','/worker.py']
        # Seccomp remains enabled inside Docker: identical restrictions and imports.
        payload['seccomp']=True
    else: raise ValueError(backend)
    with tempfile.TemporaryDirectory() as tmp:
        with tempfile.TemporaryFile() as output:
            proc=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=output,stderr=output,cwd=tmp,
                                  env=os.environ.copy() if backend=='docker' else {'PATH':os.environ.get('PATH',''),'HOME':tmp},start_new_session=True)
            try:
                proc.communicate(json.dumps(payload).encode(),timeout=timeout)
                status='passed' if proc.returncode==0 else 'failed'
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL); proc.communicate(); status='timeout'
                if backend=='docker':
                    subprocess.run(['docker','rm','-f',container_name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=10)
            output.seek(0); data=output.read(8192).decode(errors='replace')
            if proc.returncode in (70,125,126,127) or 'SANDBOX_INIT_ERROR:' in data:
                raise RuntimeError(f'Sandbox infrastructure failed: {data[-1000:]}')
            passed=proc.returncode==0 and marker in data.splitlines()
            return dict(passed=passed,status=status if not (status=='passed' and not passed) else 'missing_marker',
                        returncode=proc.returncode,diagnostic=data[-1000:].replace(marker,'<marker>'))
