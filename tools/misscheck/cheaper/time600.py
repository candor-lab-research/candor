import sys, time
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import invariance as inv
t = time.perf_counter(); inv.run("cap", prune=False, trials=600); print("600 trials cap full: %.1f s" % (time.perf_counter() - t))
