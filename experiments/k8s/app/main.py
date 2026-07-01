import hashlib
import os
import threading
import time
import ctypes
from fastapi import FastAPI, Query

CPU_WORK_ITERATIONS: int = int(os.getenv("CPU_WORK_ITERATIONS", 50000))
MEMORY_ALLOC_MB: int = int(os.getenv("MEMORY_ALLOC_MB", 0))

_memory_buffer: bytearray = bytearray(MEMORY_ALLOC_MB * 1024 * 1024)
_buffer_lock: threading.Lock = threading.Lock()

app = FastAPI(title="target-app")

@app.get("/health")
def health():
  return {"status": "ok"}
import asyncio
from concurrent.futures import ProcessPoolExecutor

_cpu_pool = ProcessPoolExecutor(max_workers=4)

def _cpu_bound_work(iterations: int):
  data = b"work"
  for _ in range(iterations):
    data = hashlib.sha256(data).digest()
  return data

for _ in range(4):
  _cpu_pool.submit(int, 0).result()

@app.get("/work")
async def work():
  start = time.perf_counter()
  loop = asyncio.get_running_loop()
  
  try:
    data = await loop.run_in_executor(_cpu_pool, _cpu_bound_work, CPU_WORK_ITERATIONS)
  except Exception as e:
    import sys
    print(f"Process pool broken (likely child process OOM killed). Exiting to trigger pod restart: {e}", file=sys.stderr)
    os._exit(1)
  
  elapsed_ms = (time.perf_counter() - start) * 1000
  digest_prefix = data.hex()[:8]
  return {"elapsed_ms": elapsed_ms, "digest_prefix": digest_prefix}
  
@app.post("/set-memory")
def set_memory(mb: int = Query(..., ge=0, description="Buffer size in MB (0 = free)")):
  global _memory_buffer
  with _buffer_lock:
    _memory_buffer = bytearray()
    import gc
    gc.collect()
    
    try:
      import ctypes
      ctypes.CDLL('libc.so.6').malloc_trim(0)
    except Exception:
      pass

    size = mb * 1024 * 1024
    _memory_buffer = bytearray(size)
    
    if size > 0:
      for i in range(0, size, 4096):
        _memory_buffer[i] = 1
        
  return {"allocated_mb": mb}
