"""The peak-floor and reach-ceiling check's tables: candor's search as it is (grow, D3grow), with peak
layer from 0.4 (floor-) and with reach to 1.0 as well (ends-), judged as summary.py judges.

    uv run python ends_summary.py
"""

from summary import POSITION, RANGE, table

if __name__ == "__main__":
    print("== Without thinking (judged = prompts judged per trial over 600 trials)\n")
    table(RANGE, ["grow", "floor-grow", "ends-grow"], False)
    print("== Thinking, where the direction is read a parameter\n")
    table(POSITION, ["D3grow", "floor-D3grow", "ends-D3grow"], True)
