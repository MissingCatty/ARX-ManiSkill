"""Check actual CUDA initialization before starting planner or simulator."""
import json
import subprocess
from scripts.curobo_arx.data_factory.common import ROOT


def probe_cuda(device, environment):
    # Probe the driver directly: importing the entire training environment can
    # exceed the health-check timeout on a freshly rebooted, cold filesystem.
    script = '''import ctypes, json, os
lib = ctypes.CDLL("libcuda.so.1")
context = ctypes.c_void_p()
memory = ctypes.c_uint64()
def checked(name, *args):
    code = getattr(lib, name)(*args)
    if code:
        error = ctypes.c_char_p()
        lib.cuGetErrorName(code, ctypes.byref(error))
        raise RuntimeError(name + ": " + str(code) + " " + str(error.value))
try:
    checked("cuInit", 0)
    count = ctypes.c_int()
    checked("cuDeviceGetCount", ctypes.byref(count))
    wanted = os.environ["CUDA_VISIBLE_DEVICES"].removeprefix("GPU-").replace("-", "").lower()
    selected = None
    for index in range(count.value):
        ident = (ctypes.c_ubyte * 16)()
        checked("cuDeviceGetUuid", ctypes.byref(ident), index)
        if bytes(ident).hex() == wanted:
            selected = index
            break
    if selected is None:
        raise RuntimeError("Requested GPU UUID is not visible to CUDA")
    checked("cuCtxCreate_v2", ctypes.byref(context), 0, selected)
    checked("cuMemAlloc_v2", ctypes.byref(memory), ctypes.c_size_t(4096))
    checked("cuCtxSynchronize")
    name = ctypes.create_string_buffer(256)
    checked("cuDeviceGetName", name, 256, selected)
    print(json.dumps({"pass":True,"name":name.value.decode(),"probe":"CUDA driver context and 4096-byte allocation"}))
except Exception as exc:
    print(json.dumps({"pass":False,"error":str(exc)}))
    raise SystemExit(1)
finally:
    if memory.value:lib.cuMemFree_v2(memory)
    if context.value:lib.cuCtxDestroy_v2(context)
'''
    try:
        result = subprocess.run([str(ROOT.parent/'curobo/.venv/bin/python'), '-c', script],
            env=environment, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired as exc:
        return {'pass':False,'gpu_uuid':device['uuid'],'exit_code':None,
                'error':'CUDA context probe timed out before service launch','timeout_seconds':exc.timeout}
    try:receipt = json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):receipt = {'pass':False, 'error':'No CUDA probe receipt'}
    receipt.update(gpu_uuid=device['uuid'], exit_code=result.returncode, stderr=result.stderr[-4000:])
    receipt['pass'] = bool(receipt.get('pass') and result.returncode == 0)
    return receipt
