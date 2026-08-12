import json
import sys
import tempfile
import builtins
from pathlib import Path

sys.path.insert(0, str(Path('.').resolve()))

import Aivco
from command_controller import CommandController
from memory.memory_controller import MemoryController
from memory.memory_detector import MemoryDetector

errors = []

class FakeLLM:
    def __init__(self):
        pass

    def generate(self, system_prompt, user_prompt):
        return f"reply to {user_prompt}"

Aivco.LLMController = FakeLLM

with tempfile.TemporaryDirectory() as tmpdir:
    tmp = Path(tmpdir)
    profile = tmp / 'profile.json'
    Aivco.MEMORY_FILE = profile
    memory = MemoryController(profile)
    controller = CommandController(memory, confirm_input=lambda prompt: 'yes')

    def run(cmd):
        return controller.handle(cmd)

    # Phase 2: memory storage
    if run('remember test_memory_1 = Python') != 'Saved: test_memory_1 = Python':
        errors.append(('remember simple', run('remember test_memory_1 = Python')))
    if memory.recall('test_memory_1') != 'Python':
        errors.append(('recall simple', memory.recall('test_memory_1')))
    rec = memory.get_record('test_memory_1')
    if rec is None or rec['category'] != 'general' or rec['importance'] != 'medium' or 'created_at' not in rec or 'updated_at' not in rec:
        errors.append(('record metadata', rec))

    if run('remember test_memory_2 = PressureOS [category=project] [importance=high]') != 'Saved: test_memory_2 = PressureOS':
        errors.append(('remember metadata', run('remember test_memory_2 = PressureOS [category=project] [importance=high]')))
    rec2 = memory.get_record('test_memory_2')
    if rec2 is None or rec2['category'] != 'project' or rec2['importance'] != 'high' or rec2['value'] != 'PressureOS':
        errors.append(('metadata record', rec2))

    if run('remember spaced_value = Hello world [category=fact] [importance=low]') != 'Saved: spaced_value = Hello world':
        errors.append(('spaces value', run('remember spaced_value = Hello world [category=fact] [importance=low]')))
    rec3 = memory.get_record('spaced_value')
    if rec3 is None or rec3['value'] != 'Hello world':
        errors.append(('spaces value stored', rec3))

    if run('remember punct_value = Hello, world! [category=interest] [importance=medium]') != 'Saved: punct_value = Hello, world!':
        errors.append(('punctuation value', run('remember punct_value = Hello, world! [category=interest] [importance=medium]')))
    rec4 = memory.get_record('punct_value')
    if rec4 is None or rec4['value'] != 'Hello, world!':
        errors.append(('punctuation stored', rec4))

    if run('remember test_memory_1 = Python 3 [category=education] [importance=high]') != 'Saved: test_memory_1 = Python 3':
        errors.append(('update existing', run('remember test_memory_1 = Python 3 [category=education] [importance=high]')))
    rec_update = memory.get_record('test_memory_1')
    if rec_update is None or rec_update['value'] != 'Python 3' or rec_update['category'] != 'education' or rec_update['importance'] != 'high':
        errors.append(('update existing record', rec_update))

    # Phase 3: search audit
    run('remember search_a = AIVCO [category=project] [importance=high]')
    run('remember search_b = Pythonista [category=education] [importance=medium]')
    run('remember search_c = PressureOS [category=project] [importance=high]')

    s = run('memory search AIVCO')
    if 'search_a = AIVCO' not in s:
        errors.append(('search AIVCO', s))
    s = run('memory search Python')
    if 'search_b = Pythonista' not in s:
        errors.append(('search Python', s))
    s = run('memory search PressureOS')
    if 'search_c = PressureOS' not in s:
        errors.append(('search PressureOS', s))
    s = run('memory search project')
    if 'search_a = AIVCO' not in s or 'search_c = PressureOS' not in s:
        errors.append(('search project', s))
    s = run('memory search nonexistent_random_string')
    if s != 'No matching memories found.':
        errors.append(('search nonexistent', s))
    s = run('memory search aivco')
    if 'search_a = AIVCO' not in s:
        errors.append(('search case-insensitive', s))

    # Phase 4: categories audit
    categories = ['identity', 'career', 'education', 'fact', 'general', 'goal', 'interest', 'preference', 'project']
    for idx, cat in enumerate(categories):
        k = f'cat_{cat}'
        v = f'Value {idx}'
        out = run(f'remember {k} = {v} [category={cat}] [importance=medium]')
        if out != f'Saved: {k} = {v}':
            errors.append((f'remember category {cat}', out))
        res = run(f'recall category {cat}')
        if f'{k} = {v}' not in res:
            errors.append((f'recall category {cat}', res))

    bad = run('recall category bogus')
    if 'Invalid category' not in bad:
        errors.append(('invalid category reject', bad))

    # Phase 5: importance audit
    importances = ['low', 'medium', 'high', 'critical']
    for imp in importances:
        k = f'imp_{imp}'
        out = run(f'remember {k} = imp {imp} [category=general] [importance={imp}]')
        if out != f'Saved: {k} = imp {imp}':
            errors.append((f'remember importance {imp}', out))
        res = run(f'recall important {imp}')
        if f'{k} = imp {imp}' not in res:
            errors.append((f'recall important {imp}', res))
    stats = run('memory stats')
    if 'low:' not in stats or 'medium:' not in stats or 'high:' not in stats or 'critical:' not in stats:
        errors.append(('memory stats importance', stats))
    badimp = run('remember bad_importance = test [importance=banana]')
    if 'Invalid importance' not in badimp:
        errors.append(('invalid importance reject', badimp))

    # Phase 6: persistence audit
    run('remember persistence_test = This must survive restart [category=fact] [importance=high]')
    mem2 = MemoryController(profile)
    if mem2.recall('persistence_test') != 'This must survive restart':
        errors.append(('persistence recall', mem2.recall('persistence_test')))
    if mem2.search('persistence') == {}:
        errors.append(('persistence search empty', mem2.search('persistence')))

    # Phase 7: automatic memory detection
    detector = MemoryDetector()
    detect_tests = [
        ('I am learning Python', 'education', 'Python', 'learning_python'),
        ('I prefer practical projects', 'preference', 'Practical projects', 'preference_practical_projects'),
        ('I like football', 'preference', 'Football', 'preference_football'),
        ('I am working on PressureOS', 'project', 'PressureOS', 'project_pressureos'),
        ('My goal is to become a data scientist', 'goal', 'Data scientist', 'goal_data_scientist'),
        ('I want to become an AI engineer', 'goal', 'AI engineer', 'goal_ai_engineer'),
        ('I am studying machine learning', 'education', 'Machine learning', 'learning_machine_learning'),
        ('I am building a new project called TestProject', 'project', 'A new project called TestProject', 'project_a_new_project_called_testproject'),
    ]
    for text, cat, val, key in detect_tests:
        cands = detector.detect(text)
        if len(cands) != 1:
            errors.append((f'detect count {text}', cands))
            continue
        cand = cands[0]
        if cand.category != cat or cand.value != val or cand.key != key:
            errors.append((f'detect content {text}', cand))

    # Phase 8: duplicate protection
    memory2 = MemoryController(profile)
    memory2.remember('learning_python', 'Python', category='education', importance='medium')
    original_input = builtins.input
    seq = iter(['I am learning Python', 'y', 'exit'])
    builtins.input = lambda prompt='': next(seq)
    try:
        Aivco.main()
    except StopIteration:
        pass
    finally:
        builtins.input = original_input
    memdup = MemoryController(profile)
    if memdup.get_record('learning_python')['value'] != 'Python':
        errors.append(('duplicate protection altered record', memdup.get_record('learning_python')))
    if detector.detect('I am currently learning Python') != []:
        errors.append(('unsupported sentence should not detect', detector.detect('I am currently learning Python')))
    cands = detector.detect('I prefer practical projects')
    if len(cands) != 1:
        errors.append(('unrelated detection suppressed', cands))

    # Phase 9: user confirmation audit
    profile3 = tmp / 'profile3.json'
    Aivco.MEMORY_FILE = profile3
    original_input = builtins.input
    seq = iter(['I am learning Rust', 'n', 'I am learning Rust', 'y', 'exit'])
    builtins.input = lambda prompt='': next(seq)
    try:
        Aivco.main()
    except StopIteration:
        pass
    finally:
        builtins.input = original_input
    mem3 = MemoryController(profile3)
    if mem3.recall('learning_rust') != 'Rust':
        errors.append(('confirmation final save', mem3.recall('learning_rust')))

    # Phase 10: forget/delete audit
    memory4 = MemoryController(tmp / 'profile4.json')
    controller4 = CommandController(memory4)
    if controller4.handle('remember deletion_test = temporary memory') != 'Saved: deletion_test = temporary memory':
        errors.append(('remember deletion_test', controller4.handle('remember deletion_test = temporary memory')))
    if memory4.recall('deletion_test') != 'temporary memory':
        errors.append(('recall deletion_test', memory4.recall('deletion_test')))
    if controller4.handle('forget deletion_test') != 'Forgotten: deletion_test':
        errors.append(('forget deletion_test', controller4.handle('forget deletion_test')))
    if memory4.recall('deletion_test') is not None:
        errors.append(('deletion still present', memory4.recall('deletion_test')))
    if 'No memory found' not in controller4.handle('forget nonexistent_key'):
        errors.append(('forget nonexistent', controller4.handle('forget nonexistent_key')))

    # Phase 11: edge cases
    invalids = [
        ('remember', 'Usage: remember <key> = <value>'),
        ('remember test', 'Usage: remember <key> = <value>'),
        ('remember = value', 'Usage: remember <key> = <value>'),
        ('remember test =', 'Usage: remember <key> = <value>'),
        ('remember test = ', 'Usage: remember <key> = <value>'),
        ('recall', 'Usage: recall <key>'),
        ('recall nonexistent', 'No memory found'),
        ('recall category nonexistent', 'Invalid category'),
        ('recall important nonexistent', 'Invalid importance'),
        ('memory search', 'Usage: memory search <query>'),
        ('forget', 'Usage: forget <key>'),
        ('forget nonexistent', 'No memory found'),
    ]
    for cmd, expected in invalids:
        out = controller.handle(cmd)
        if expected not in out:
            errors.append((f'invalid input {cmd}', out))
    if controller.handle('remember punctuation_test = Hello, world!') != 'Saved: punctuation_test = Hello, world!':
        errors.append(('punctuation test', controller.handle('remember punctuation_test = Hello, world!')))
    if controller.handle('remember spaces_test = Machine learning and artificial intelligence') != 'Saved: spaces_test = Machine learning and artificial intelligence':
        errors.append(('spaces test', controller.handle('remember spaces_test = Machine learning and artificial intelligence')))
    if controller.handle('remember number_test = 12345') != 'Saved: number_test = 12345':
        errors.append(('number test', controller.handle('remember number_test = 12345')))

    data = json.loads(profile.read_text(encoding='utf-8'))
    if not isinstance(data, dict) or data.get('schema_version') != '2.0':
        errors.append(('profile json validity', data))

print('errors_count', len(errors))
for e in errors:
    print('ERROR:', e)
if errors:
    raise SystemExit(1)
