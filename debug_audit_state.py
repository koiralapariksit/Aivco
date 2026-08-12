import json
import tempfile
from pathlib import Path
from command_controller import CommandController
from memory.memory_controller import MemoryController
from memory.memory_detector import MemoryDetector
from Aivco import MEMORY_FILE

with tempfile.TemporaryDirectory() as tmpdir:
    tmp = Path(tmpdir)
    profile = tmp / 'profile.json'
    mem = MemoryController(profile)
    print('initial data keys:', list(mem._data.keys()))
    print('initial memories:', mem.all_records())

    controller = CommandController(mem, confirm_input=lambda prompt: 'yes')
    out = controller.handle('remember test_memory_1 = Python')
    print('remember test_memory_1', out)
    print('after one record keys:', list(mem.all_records().keys()))

    out = controller.handle('remember search_a = AIVCO [category=project] [importance=high]')
    print('remember search_a', out)
    print('after search_a keys:', list(mem.all_records().keys()))

    out = controller.handle('memory search AIVCO')
    print('memory search AIVCO output:\n', out)

    print('all memory keys final:', list(mem.all_records().keys()))
    print('stats output:\n', controller.handle('memory stats'))
    print('file content:\n', profile.read_text())
