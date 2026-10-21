"""Isolated worker. Linux seccomp is installed BEFORE any candidate code executes."""
import sys,json,ctypes,ctypes.util,resource,math,itertools,collections,functools,heapq,bisect,re,string,statistics,decimal,fractions,operator,random,datetime,calendar,array,typing,numbers,cmath,copy


def restrict():
    resource.setrlimit(resource.RLIMIT_CPU,(2,2))
    resource.setrlimit(resource.RLIMIT_AS,(512*1024**2,512*1024**2))
    resource.setrlimit(resource.RLIMIT_FSIZE,(1024*1024,1024*1024))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    if sys.platform!='linux':
        raise RuntimeError('seccomp requires Linux; use the Docker backend on macOS')
    lib=ctypes.CDLL(ctypes.util.find_library('seccomp') or 'libseccomp.so.2',use_errno=True)
    lib.seccomp_init.argtypes=[ctypes.c_uint32]; lib.seccomp_init.restype=ctypes.c_void_p
    lib.seccomp_rule_add.argtypes=[ctypes.c_void_p,ctypes.c_uint32,ctypes.c_int,ctypes.c_uint]
    lib.seccomp_syscall_resolve_name.argtypes=[ctypes.c_char_p];lib.seccomp_syscall_resolve_name.restype=ctypes.c_int
    lib.seccomp_load.argtypes=[ctypes.c_void_p]
    # Default kill; no file opens, networking, exec, fork, ptrace or filesystem mutation.
    ctx=lib.seccomp_init(0x80000000)
    if not ctx: raise RuntimeError('seccomp_init failed')
    allowed=['read','write','close','fstat','lseek','brk','mmap','mprotect','munmap','mremap','madvise',
             'rt_sigaction','rt_sigprocmask','rt_sigreturn','sigaltstack','futex','clock_gettime','gettimeofday',
             'getpid','gettid','getrandom','exit','exit_group','sched_yield','restart_syscall']
    for name in allowed:
        nr=lib.seccomp_syscall_resolve_name(name.encode())
        if nr>=0 and lib.seccomp_rule_add(ctx,0x7fff0000,nr,0)!=0:
            raise RuntimeError('seccomp_rule_add failed')
    if lib.seccomp_load(ctx)!=0: raise RuntimeError('seccomp_load failed')


def main():
    payload=json.loads(sys.stdin.read())
    # Compile trusted test expressions before applying the filter.
    code=compile(payload['code'],'<candidate>','exec')
    tests=[compile(t,'<test>','exec') for t in payload['tests']]
    setup=compile(payload.get('setup',''),'<setup>','exec')
    if payload.get('seccomp',True):
        try:
            restrict()
        except BaseException as e:
            print('SANDBOX_INIT_ERROR: '+str(e),file=sys.stderr,flush=True)
            sys.exit(70)
    else:
        resource.setrlimit(resource.RLIMIT_CPU,(2,2))
        resource.setrlimit(resource.RLIMIT_FSIZE,(1024*1024,1024*1024))
    ns={}
    try:
        exec(setup,ns); exec(code,ns)
        for t in tests: exec(t,ns)
    except BaseException:
        sys.exit(1)
    # Parent requires the unique marker; early sys.exit(0) does not count as a pass.
    print(payload['marker'],flush=True)

if __name__=='__main__': main()
