"""
Evita que Windows se suspenda por inactividad mientras dura un trabajo largo.

Una suspensión a mitad de corrida pausa el cálculo, falsea los tiempos medidos y corta la
conexión de Colab. Esto pide al sistema que siga despierto solo mientras viva este proceso:
no cambia ninguna configuración de energía. Cerrar la tapa sí suspende el equipo igual.

Uso:
    python keep_awake.py bash run_experiments.sh    # mientras dure ese comando
    python keep_awake.py --pid 1234                 # mientras viva ese proceso
    python keep_awake.py --hours 8                  # durante ese tiempo (Colab, por ejemplo)
"""

import argparse
import ctypes
import subprocess
import sys
import time

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
SYNCHRONIZE = 0x00100000
INFINITE = 0xFFFFFFFF


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int)
    parser.add_argument("--hours", type=float)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    kernel32 = ctypes.windll.kernel32
    kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    if args.pid:
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        handle = kernel32.OpenProcess(SYNCHRONIZE, False, args.pid)
        if handle:
            kernel32.WaitForSingleObject(handle, INFINITE)
        return 0
    if args.hours:
        time.sleep(args.hours * 3600)
        return 0
    return subprocess.run(args.command).returncode


if __name__ == "__main__":
    sys.exit(main())
