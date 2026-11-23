"""pyserial transport for protocol.run_sync: the real board.

xfer() writes a whole packet in one call. That matters: the bridge's
inter-byte timeout is TIMEOUT_BITS bit times, and a packet handed to the
FTDI chip in one USB transfer goes out back to back. wait_idle() sleeps past
the bridge's drain window and throws away whatever is in the input buffer.

The FTDI latency timer (16 ms by default) delays every response by up to
16 ms. pyserial cannot set it portably (macOS has no sysfs knob), so the
protocol batches instead: the command is one burst, loads are 256-word
bursts. See fpga/README.md.
"""

import time

RESPONSE_TIMEOUT_S = 0.05


class SerialTransport:

    def __init__(self, port: str, baud: int = 1_000_000, timeout_bits: int = 1000):
        import serial                       # pyserial, only needed on the bench
        self.baud, self.timeout_bits = baud, timeout_bits
        self.ser = serial.Serial(port, baudrate=baud, bytesize=8, parity="N", stopbits=1,
                                 timeout=0, write_timeout=1.0)
        self.ser.reset_input_buffer()
        self.bytes_out = self.bytes_in = 0

    def byte_time(self, n: int) -> float:
        return n * 10.0 / self.baud

    def xfer(self, packet: bytes, resp_len: int) -> bytes:
        self.ser.write(packet)
        self.ser.flush()
        self.bytes_out += len(packet)
        # the answer can only start after the packet is out; then it streams
        deadline = time.monotonic() + self.byte_time(len(packet) + resp_len) + RESPONSE_TIMEOUT_S
        buf = bytearray()
        while len(buf) < resp_len and time.monotonic() < deadline:
            chunk = self.ser.read(resp_len - len(buf))
            if chunk:
                buf += chunk
            else:
                time.sleep(0.0005)
        self.bytes_in += len(buf)
        return bytes(buf)

    def wait_idle(self) -> None:
        # the bridge drains until the line has been quiet TIMEOUT_BITS bit times
        time.sleep(2 * self.timeout_bits / self.baud + 0.005)
        self.ser.reset_input_buffer()

    def close(self) -> None:
        self.ser.close()
