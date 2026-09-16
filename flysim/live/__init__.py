"""M5a: flysim-live server — engine resident in memory, websocket control and telemetry.

The wire contract is ``docs/m5-protocol.md``; ``protocol.py`` is that document as code.
Nothing in this package modifies ``flysim/engine``, ``flysim/sensory`` or ``flysim/probe``:
the streaming input path reuses the filter coefficients and the probe sets from those
modules so that a live run and a scripted run see the same signal path.
"""
