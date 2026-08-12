from pathlib import Path
import tempfile
from memory.memory_controller import MemoryController

with tempfile.TemporaryDirectory() as tmpdir:
    profile = Path(tmpdir) / 'profile.json'
    mc = MemoryController(profile)
    print('had_existing_data', mc._had_existing_data)
    print('data keys', mc._data.keys())
    print('profile exists', profile.exists())
    print('file content:')
    print(profile.read_text())
