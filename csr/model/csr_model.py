"""Register model of csr/rtl/csr.sv, transaction level.

Implements top/host_interface_contract.md section 2-4 directly: reset values,
masks, the busy and host_mode rules, wide-word staging with valid bits, the
write CRCs (protocol.crc16 over the committed bytes), LAUNCH_COUNT, CYCLES
and the sticky bits. The TB runs it beside the RTL and compares every
response, every read value and every memory write the CSR commits.

Responses are AXI codes: 0 OKAY, 2 SLVERR, 3 DECERR.
"""

from protocol import (A_ACT_WCRC, A_B_WCRC, A_BUILD_ID, A_CMD0, A_CTRL, A_CYCLES, A_ERR_CODE,
                      A_ID, A_LAUNCH, A_P_WCRC, A_PARAMS0, A_PARAMS1, A_SCRATCH, A_STATUS,
                      A_VERSION, ACT_BASE, B_BASE, CMD_FIELDS, ID_VALUE, P_BASE, P_WIDTH, VERSION,
                      crc16, slot_shift, word_bytes)

OKAY, SLVERR, DECERR = 0, 2, 3


class CsrModel:

    def __init__(self, n=8, addr_width=12, read_latency=1, nbuf=2, b_addr_w=13, p_addr_w=6,
                 k_max=1024, dim_w=8, cw_w=7, build_id=0):
        self.n, self.aw, self.lat, self.nbuf = n, addr_width, read_latency, nbuf
        self.b_aw, self.p_aw, self.k_max, self.build_id = b_addr_w, p_addr_w, k_max, build_id
        self.sel_w = (nbuf - 1).bit_length() if nbuf > 1 else 1
        self.word_w = 8 * n
        self.widths = [dim_w] * 4 + [cw_w] * 2 + [3, 3, 2, 2, 8, 8, 1] + [self.sel_w] * 2 + \
                      [addr_width] * 2 + [b_addr_w, p_addr_w]
        self.reset()

    def reset(self):
        self.cmd = [0] * len(CMD_FIELDS)
        self.scratch = 0
        self.host_mode = 0
        self.done = self.err = 0
        self.cycles = 0
        self.launch = 0
        self.wcrc = {"act": 0xFFFF, "b": 0xFFFF, "p": 0xFFFF}
        self.stage = {k: {"vld": set(), "idx": 0, "slots": {}} for k in ("act", "b", "p")}
        # what the CSR is expected to do to the outside, in order
        self.commits = []            # ("act", sel, addr, word) / ("b", addr, word) / ("p", addr, word)
        self.starts = 0
        self.soft_resets = 0

    # -- helpers -------------------------------------------------------------
    def params0(self):
        return (self.b_aw << 24) | (self.nbuf << 20) | (self.aw << 12) | (self.lat << 8) | self.n

    def params1(self):
        return (self.k_max << 8) | self.p_aw

    def _window(self, addr):
        if addr >> 8 == 0:
            return "reg", None, None
        top = addr >> 20
        name = {1: "act", 2: "b", 3: "p"}.get(top)
        if name is None:
            return None, None, None
        width = P_WIDTH if name == "p" else self.word_w
        sh = slot_shift(width)
        word = (addr & 0xFFFFF) >> 2
        return name, word >> sh, word & ((1 << sh) - 1)

    def _index_ok(self, name, idx):
        if name == "act":
            return idx >> (self.sel_w + self.aw) == 0 and (idx >> self.aw) < self.nbuf
        return idx >> (self.b_aw if name == "b" else self.p_aw) == 0

    def status_word(self, busy):
        return (self.host_mode << 4) | (self.err << 3) | (self.done << 2) | ((not busy) << 1) | busy

    def reg_value(self, off, busy, err_code):
        vals = {A_ID: ID_VALUE, A_VERSION: VERSION, A_PARAMS0: self.params0(),
                A_PARAMS1: self.params1(), A_CTRL: self.host_mode << 2,
                A_STATUS: self.status_word(busy), A_ERR_CODE: err_code, A_CYCLES: self.cycles,
                A_SCRATCH: self.scratch, A_BUILD_ID: self.build_id,
                A_ACT_WCRC: self.wcrc["act"], A_B_WCRC: self.wcrc["b"], A_P_WCRC: self.wcrc["p"],
                A_LAUNCH: self.launch}
        if off in vals:
            return vals[off]
        if A_CMD0 <= off < A_CMD0 + 4 * len(CMD_FIELDS) and off % 4 == 0:
            i = (off - A_CMD0) // 4
            v = self.cmd[i]
            if CMD_FIELDS[i] in ("zp_in", "zp_out"):
                v = v - 256 if v & 0x80 else v
                return v & 0xFFFFFFFF
            return v
        return None

    # -- bus -----------------------------------------------------------------
    def write(self, addr, data, busy=False):
        name, idx, slot = self._window(addr)
        if name is None:
            return DECERR
        if name == "reg":
            off = addr & 0xFF
            if off == A_CTRL:
                if busy:
                    if data & 2:
                        self.soft_resets += 1
                        self._soft_reset()
                        return OKAY
                    return SLVERR
                if data & 1:
                    self.starts += 1
                if data & 2:
                    self.soft_resets += 1
                    self._soft_reset()
                self.host_mode = (data >> 2) & 1
                if data & 8:
                    self.wcrc = {k: 0xFFFF for k in self.wcrc}
                return OKAY
            if off == A_STATUS:
                if data & 4: self.done = 0
                if data & 8: self.err = 0
                return OKAY
            if off == A_SCRATCH:
                if busy: return SLVERR
                self.scratch = data & 0xFFFFFFFF
                return OKAY
            if A_CMD0 <= off < A_CMD0 + 4 * len(CMD_FIELDS) and off % 4 == 0:
                if busy: return SLVERR
                i = (off - A_CMD0) // 4
                self.cmd[i] = data & ((1 << self.widths[i]) - 1)
                return OKAY
            return SLVERR if self.reg_value(off, busy, 0) is not None else DECERR
        # windows
        if not self._index_ok(name, idx):
            return DECERR
        if busy or (name == "act" and not self.host_mode):
            return SLVERR
        width = P_WIDTH if name == "p" else self.word_w
        slots = 1 << slot_shift(width)
        st = self.stage[name]
        if slot < slots - 1:
            if st["idx"] != idx:
                st["vld"] = set()
            st["idx"] = idx
            st["vld"].add(slot)
            st["slots"][slot] = data
            return OKAY
        ok = slots == 1 or (st["vld"] >= set(range(slots - 1)) and st["idx"] == idx)
        st["vld"] = set()
        if not ok:
            return SLVERR
        word = sum(st["slots"].get(s, 0) << (32 * s) for s in range(slots - 1)) | (data << (32 * (slots - 1)))
        word &= (1 << width) - 1
        self.wcrc[name] = crc16(word_bytes(word, width), self.wcrc[name])
        if name == "act":
            self.commits.append(("act", idx >> self.aw, idx & ((1 << self.aw) - 1), word))
        else:
            self.commits.append((name, idx, word))
        return OKAY

    def read(self, addr, busy=False, err_code=0, mem=None):
        """mem(sel, addr) -> act word, for ACT reads. returns (resp, data)"""
        name, idx, slot = self._window(addr)
        if name is None:
            return DECERR, 0
        if name == "reg":
            v = self.reg_value(addr & 0xFF, busy, err_code)
            return (OKAY, v) if v is not None else (DECERR, 0)
        if not self._index_ok(name, idx):
            return DECERR, 0
        if name in ("b", "p"):
            return SLVERR, 0
        if busy or self.host_mode:
            return SLVERR, 0
        word = mem(idx >> self.aw, idx & ((1 << self.aw) - 1))
        return OKAY, (word >> (32 * slot)) & 0xFFFFFFFF

    def link_err(self):
        for st in self.stage.values():
            st["vld"] = set()

    # -- what conv_layer does back --------------------------------------------
    def accepted_start(self):
        self.launch = (self.launch + 1) & 0xFFFFFFFF
        self.cycles = 0
        self.done = self.err = 0

    def finished(self, edges, err=False):
        self.cycles = edges
        if err: self.err = 1
        else:   self.done = 1

    def _soft_reset(self):
        self.cycles = 0
        self.done = self.err = 0


if __name__ == "__main__":
    fails = 0

    def chk(name, got, exp):
        global fails
        ok = got == exp
        print(f"{'pass' if ok else 'FAIL'} {name}" + ("" if ok else f": got {got!r}, expected {exp!r}"))
        fails += not ok

    m = CsrModel(n=8, nbuf=3)
    chk("ID", m.read(A_ID), (OKAY, ID_VALUE))
    chk("STATUS after reset (ready)", m.read(A_STATUS), (OKAY, 2))
    chk("unmapped register hole", m.read(0x38), (DECERR, 0))
    chk("unmapped region", m.read(0x500000), (DECERR, 0))
    chk("RO write refused", m.write(A_ID, 1), SLVERR)
    chk("KH masked to 3 bits", (m.write(A_CMD0 + 24, 0xFF), m.read(A_CMD0 + 24)), (OKAY, (OKAY, 7)))
    m.write(A_CMD0 + 40, 0x80)
    chk("ZP_IN reads back sign-extended", m.read(A_CMD0 + 40), (OKAY, 0xFFFFFF80))
    chk("ACT write without host_mode", m.write(ACT_BASE, 1), SLVERR)
    m.write(A_CTRL, 4)
    # N=8: two slots per word. bank 1, word 5
    a = ACT_BASE + ((((1 << 12) | 5) << 1) << 2)
    chk("slot 0 stages", m.write(a, 0x11111111), OKAY)
    chk("slot 1 commits", m.write(a + 4, 0x22222222), OKAY)
    chk("commit record", m.commits[-1], ("act", 1, 5, 0x2222222211111111))
    chk("act CRC", m.read(A_ACT_WCRC)[1], crc16((0x2222222211111111).to_bytes(8, "little")))
    chk("top slot alone refused", m.write(a + 4, 3), SLVERR)
    m.write(a, 1); m.link_err()
    chk("link error clears staging", m.write(a + 4, 3), SLVERR)
    a2 = ACT_BASE + ((((1 << 12) | 6) << 1) << 2)
    m.write(a, 1)
    chk("staged slot of another word does not count", m.write(a2 + 4, 3), SLVERR)
    bank3 = ACT_BASE + ((((3 << 12) | 0) << 1) << 2)
    chk("bank >= NBUF is unmapped", m.write(bank3, 0), DECERR)
    chk("B read is SLVERR", m.read(B_BASE), (SLVERR, 0))
    m.write(P_BASE, 0xFFFFFFFF)
    m.write(P_BASE + 4, 0xFFFFFFFF)
    chk("P commit drops bits above 53", m.commits[-1], ("p", 0, (1 << 53) - 1))
    chk("busy: command write refused", m.write(A_CMD0, 1, busy=True), SLVERR)
    chk("busy: soft reset accepted", m.write(A_CTRL, 2, busy=True), OKAY)
    chk("busy: start refused", m.write(A_CTRL, 1, busy=True), SLVERR)
    chk("busy: STATUS W1C accepted", m.write(A_STATUS, 4, busy=True), OKAY)
    chk("busy: window refused", m.write(B_BASE + 4, 0, busy=True), SLVERR)
    m.write(A_CTRL, 0)
    chk("ACT read slot 1", m.read(a + 4, mem=lambda s, i: 0xAABBCCDD11223344), (OKAY, 0xAABBCCDD))
    m.write(A_CTRL, 4)
    chk("ACT read in host mode refused", m.read(a, mem=lambda s, i: 0), (SLVERR, 0))
    m.accepted_start(); m.finished(123)
    chk("cycles and done", (m.read(A_CYCLES)[1], m.read(A_STATUS)[1] & 4, m.read(A_LAUNCH)[1]), (123, 4, 1))
    m.write(A_CTRL, 8)
    chk("clear_wcrc", m.read(A_ACT_WCRC)[1], 0xFFFF)

    print("\nALL PASS" if fails == 0 else f"\n{fails} FAILURES")
    raise SystemExit(1 if fails else 0)
