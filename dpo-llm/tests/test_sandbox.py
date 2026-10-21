import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pytest
from sandbox import execute,extract_code,check_backend


def test_code_extraction():
    assert extract_code('Here:\n```python\ndef f(): return 2\n```')=='def f(): return 2'


@pytest.mark.skipif(sys.platform!='linux',reason='Linux seccomp integration; Docker test is run separately on macOS')
def test_seccomp_isolation():
    check_backend('seccomp')
    assert execute('def f(): return 2',['assert f()==2'])['passed']
    assert not execute('def f(): return 3',['assert f()==2'])['passed']
    assert not execute('while True: pass',['assert True'],timeout=1)['passed']
    assert not execute("import socket; socket.socket()",['assert True'])['passed']
    assert not execute("import os; os.system('true')",['assert True'])['passed']
    assert not execute('import os; os._exit(0)',['assert False'])['passed']
