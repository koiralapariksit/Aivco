from pathlib import Path
import sys
sys.path.insert(0, str(Path('.').resolve()))
from command_controller import CommandController
from memory.memory_controller import MemoryController

m = MemoryController(Path('memory/user/profile.json'))
c = CommandController(m)
cmds = [
    'remember',
    'remember test',
    'remember = value',
    'remember test =',
    'remember test = ',
    'recall',
    'recall nonexistent',
    'recall category nonexistent',
    'recall important nonexistent',
    'memory search',
    'forget',
    'forget nonexistent',
]
for cmd in cmds:
    out = c.handle(cmd)
    print(repr(cmd), '=>', repr(out))
