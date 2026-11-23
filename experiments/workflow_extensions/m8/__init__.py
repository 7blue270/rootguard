from .policy import choose, run

def load_original(directory):
    """Return reference context and frozen default plan for the original BUN/Cr release."""
    import json
    from pathlib import Path
    from ..finite_cache import data
    from ..m7.engine import Context
    from ..m7.run import SEED
    here=Path(__file__).resolve().parent
    protocol=json.loads((here/'original_protocol.json').read_text())
    if data.input_hashes(directory,['BUN_CR'])!=protocol['dataset']:
        raise ValueError('Expected the original PhysioNet BUN/Cr benchmark')
    records={r['subject_id']:r for r in data.load(directory,['BUN_CR'])}
    context=Context('BUN_CR',[records[s] for s in protocol['references']],SEED)
    return context,json.loads((here/'original_bun.json').read_text())
