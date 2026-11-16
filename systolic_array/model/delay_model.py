#golden reference model for the delay/skew module 


def expected_out (history, cycle, depth): 
    """ the expected out_data from the read only section after the "rising edge" cycle"""

    latency = depth - 1 if depth > 0 else 0 
    idx = cycle - latency 
    if idx < 0: 
        return 0     #pipleine not full yet 
    return history[idx]


if __name__ == "__main__":
    fails = 0

    def chk(name, got, exp):
        global fails
        if got != exp:
            print(f"FAIL {name:<32} got={got} exp={exp}")
            fails += 1
        else:
            print(f"pass {name:<32} = {got}")

    history = [10, 20, 30, 40, 50]
    # DEPTH 0 and 1 look the same at the ReadOnly point: the value just driven
    chk("depth 0 bypass", [expected_out(history, i, 0) for i in range(5)], history)
    chk("depth 1", [expected_out(history, i, 1) for i in range(5)], history)
    chk("depth 2", [expected_out(history, i, 2) for i in range(5)], [0, 10, 20, 30, 40])
    chk("depth 4", [expected_out(history, i, 4) for i in range(5)], [0, 0, 0, 10, 20])
    chk("depth 8 not full yet", [expected_out(history, i, 8) for i in range(5)], [0] * 5)

    print("\nALL PASS" if fails == 0 else f"\n{fails} FAILURES")

    raise SystemExit(1 if fails else 0)
