import json
import tempfile
import builtins
from pathlib import Path

import Aivco
from command_controller import CommandController
from memory.memory_controller import MemoryController, VALID_CATEGORIES, VALID_IMPORTANCE
from memory.memory_detector import MemoryDetector

Aivco.LLMController = type('FakeLLM', (), {'__init__': lambda self: None, 'generate': lambda self, s,u: 'ok'})

def run_validation():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        profile = tmp / 'profile.json'
        Aivco.MEMORY_FILE = profile
        memory = MemoryController(profile)
        controller = CommandController(memory, confirm_input=lambda prompt: 'yes')
        detector = MemoryDetector()
        results = []

        out = controller.handle('remember qa_storage_test = hello_world [category=fact] [importance=medium]')
        results.append(('storage_result', out))
        rec = memory.get_record('qa_storage_test')
        results.append(('storage_record', rec))
        out2 = controller.handle('recall qa_storage_test')
        results.append(('storage_recall', out2))

        cat_results = {}
        for cat in sorted(VALID_CATEGORIES):
            key = f'qa_cat_{cat}'
            val = f'Value {cat}'
            out = controller.handle(f'remember {key} = {val} [category={cat}] [importance=medium]')
            cat_results[cat] = {'out': out, 'recall': controller.handle(f'recall category {cat}')}
        bad_cat = controller.handle('recall category invalidcat')
        results.append(('bad_cat', bad_cat))

        imp_results = {}
        for imp in sorted(VALID_IMPORTANCE):
            key = f'qa_imp_{imp}'
            val = f'importance_{imp}'
            out = controller.handle(f'remember {key} = {val} [category=general] [importance={imp}]')
            imp_results[imp] = {'out': out, 'recall': controller.handle(f'recall important {imp}')}
        bad_imp = controller.handle('recall important invalidimp')
        results.append(('bad_imp', bad_imp))

        controller.handle('remember search_key = AIVCO [category=project] [importance=high]')
        controller.handle('remember search_value = Python [category=education] [importance=medium]')
        controller.handle('remember search_cat = PressureOS [category=project] [importance=medium]')
        results.append(('search_a', controller.handle('memory search AIVCO')))
        results.append(('search_python', controller.handle('memory search Python')))
        results.append(('search_project', controller.handle('memory search project')))
        results.append(('search_none', controller.handle('memory search nonexistent_xyz')))
        results.append(('search_empty', controller.handle('memory search')))

        detect_inputs = [
            'I am learning Python',
            'I prefer practical projects',
            'I like football',
            "I don't like unnecessary complexity",
            'My goal is to become a data scientist',
            'I want to become an AI engineer',
            'I plan to become an AI engineer',
            'I am working on PressureOS',
            'I am building a new project called TestProject',
        ]
        detect_results = {}
        for text in detect_inputs:
            cands = detector.detect(text)
            detect_results[text] = [c.__dict__ for c in cands]
        no_detect_inputs = [
            'hello', 'how are you', 'what is Python', 'tell me about AI', 'I am tired', 'I am going to school',
            'Can you explain machine learning?', 'What is PressureOS?'
        ]
        no_detect_results = {text: [c.__dict__ for c in detector.detect(text)] for text in no_detect_inputs}

        dup_profile = tmp / 'dup_profile.json'
        dup_memory = MemoryController(dup_profile)
        dup_controller = CommandController(dup_memory, confirm_input=lambda prompt: 'yes')
        dup_controller.handle('remember duplicate_test = Python [category=education] [importance=medium]')
        dup_results = detector.detect('I am learning Python')
        results.append(('dup_detect', [c.__dict__ for c in dup_results]))
        results.append(('dup_saved', dup_memory.get_record('learning_python')))

        profile_conf = tmp / 'confirm_profile.json'
        Aivco.MEMORY_FILE = profile_conf
        orig_input = builtins.input
        seq = iter(['I am learning Rust', 'n', 'I am learning Rust', 'y', 'exit'])
        builtins.input = lambda prompt='': next(seq)
        try:
            Aivco.main()
        except StopIteration:
            pass
        finally:
            builtins.input = orig_input
        conf_mem = MemoryController(profile_conf).get_record('learning_rust')
        results.append(('confirm_saved', conf_mem))

        forget_profile = tmp / 'forget_profile.json'
        forget_memory = MemoryController(forget_profile)
        forget_controller = CommandController(forget_memory, confirm_input=lambda prompt: 'yes')
        out = forget_controller.handle('remember forget_test = delete me [category=fact] [importance=low]')
        recall_before = forget_memory.recall('forget_test')
        out_forget = forget_controller.handle('forget forget_test')
        recall_after = forget_memory.recall('forget_test')
        out_forget_non = forget_controller.handle('forget nonexistent_key')
        results.append(('forget', (out, recall_before, out_forget, recall_after, out_forget_non)))

        data = json.loads(profile.read_text(encoding='utf-8'))
        results.append(('profile_integrity', data))

        print(json.dumps({'results': results, 'cats': cat_results, 'imps': imp_results, 'detect': detect_results, 'no_detect': no_detect_results}, indent=2))

if __name__ == '__main__':
    run_validation()
