
import fcntl
import os
from contextlib import contextmanager

@contextmanager
def atomic_file_lock(file_path, mode='r'):
    """Garante acesso exclusivo ao arquivo usando fcntl (Zero Entropia)."""
    f = open(file_path, mode)
    try:
        # LOCK_EX para escrita, LOCK_SH para leitura
        lock_type = fcntl.LOCK_EX if 'w' in mode or 'a' in mode else fcntl.LOCK_SH
        fcntl.flock(f, lock_type)
        yield f
    finally:
        fcntl.flock(f, fcntl.LOCK_UN)
        f.close()
