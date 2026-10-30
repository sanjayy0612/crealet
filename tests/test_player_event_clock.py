"""Exercise the real recorder handlers with time advancing between clock reads."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_native_handlers_use_the_logged_event_time_for_stalls_and_startup():
    if shutil.which('node') is None:
        pytest.skip('Node required for actual JS handler check')
    source = Path('experiments/validation/policy-player.mjs').read_text()
    handlers = source[source.index('    const event ='):source.index("    video.addEventListener('playing'")]
    # A busy event loop can advance the clock between event logging and updating
    # counters. The archived event timestamps must remain the source of truth.
    program = '''
const assert = require('node:assert/strict');
let clock = 0;
const t = () => (clock += 0.04);
const video = {currentTime: 0, ended: false};
const capture = {events: []};
let started = false, startup = null, stallStart = null, stallSeconds = 0, stallEvents = 0;
''' + handlers + '''
playing(); waiting(); playing();
assert.ok(Math.abs(startup - capture.events[0].t_s) < 1e-9);
assert.ok(Math.abs(stallSeconds - (capture.events[2].t_s - capture.events[1].t_s)) < 1e-9);
assert.equal(stallEvents, 1);
'''
    subprocess.run(['node', '-e', program], check=True, capture_output=True, text=True)
