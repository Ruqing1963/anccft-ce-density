"""Task G: memory watchdog — kills only taskG_run.py processes whose private memory exceeds the limit (GB).
usage: python taskG_watchdog.py limitGB hours"""
import sys
import time

import psutil

lim = float(sys.argv[1]) * 2 ** 30
end = time.time() + float(sys.argv[2]) * 3600
seen = False
while time.time() < end:
    mine = []
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            if p.info["cmdline"] and any("taskG_run.py" in c for c in p.info["cmdline"]):
                mine.append(p)
        except (psutil.Error, TypeError):
            pass
    if mine:
        seen = True
    elif seen:
        break
    for p in mine:
        try:
            pm = p.memory_info().private
            if pm > lim:
                p.kill()
                with open("out_taskG_watchdog.txt", "a") as fh:
                    fh.write(f"killed {p.pid} private={pm} at {time.ctime()}\n")
        except psutil.Error:
            pass
    time.sleep(10)
