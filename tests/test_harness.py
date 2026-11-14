import json
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip('lm_eval')
pytest.importorskip('tokenizers')
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

from exealm import Llama, get_config
from exealm.eval.harness import ExeaLMHarness
from exealm.train.pretrain import make_optimizer, save


def request(*args):
    return SimpleNamespace(args=args)


@pytest.fixture
def adapter(tmp_path):
    old_threads = torch.get_num_threads()
    torch.manual_seed(1)
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(['a b c hello world\n日本語のテスト' * 20],
                            trainers.BpeTrainer(vocab_size=270, min_frequency=2,
                                                special_tokens=['<|endoftext|>'],
                                                initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok.save(str(tmp_path / 'tokenizer.json'))
    cfg = get_config('tiny', vocab_size=tok.get_vocab_size())
    model = Llama(cfg)
    opt = make_optimizer(model, 0.001, 0.1)
    save(str(tmp_path / 'model.pt'), model, opt, 0, cfg, 'llama')
    yield ExeaLMHarness(tmp_path / 'model.pt', tmp_path, max_length=16)
    torch.set_num_threads(old_threads)


@torch.inference_mode()
def direct(model, prefix, targets):
    inp = torch.tensor([prefix + targets[:-1]])
    logits = model(inp)[0][0, -len(targets):].float()
    ids = torch.tensor(targets)
    return float(logits.log_softmax(-1).gather(1, ids[:, None]).sum()), bool((logits.argmax(-1) == ids).all())


def test_continuation_likelihood_matches_direct_forward_and_preserves_order(adapter):
    context, continuation = 'hello', ' world'
    ctx, cont = adapter._encode_pair(context, continuation)
    expected = direct(adapter.model, ctx, cont)
    scored = adapter.loglikelihood([request(context, continuation), request('', 'a'), request('x', '')])
    assert scored[0][0] == pytest.approx(expected[0], abs=1e-5)
    assert scored[0][1] == expected[1]
    other = direct(adapter.model, [adapter.eot_token_id], adapter.tok_encode('a'))
    assert scored[1][0] == pytest.approx(other[0], abs=1e-5)
    assert scored[2] == (0., True)
    assert adapter.loglikelihood([request('', '')]) == [(0., True)]


def test_trailing_context_whitespace_uses_harness_pair_convention(adapter):
    first = adapter.loglikelihood([request('hello ', 'world')])
    second = adapter.loglikelihood([request('hello', ' world')])
    assert first == second


def test_long_continuation_scores_every_target_with_bounded_inputs(adapter):
    adapter.max_length = 4
    context = adapter.tok_encode('hello world ' * 4)
    continuation = adapter.tok_encode('a b c ' * 8)
    lengths = []
    handle = adapter.model.register_forward_pre_hook(lambda m, args: lengths.append(args[0].shape[1]))
    actual = adapter._score(context, continuation)
    handle.remove()
    expected = 0.
    for start in range(0, len(continuation), 4):
        part = continuation[start:start + 4]
        prefix = (context + continuation[:start])[-(5 - len(part)):]
        expected += direct(adapter.model, prefix, part)[0]
    assert actual[0] == pytest.approx(expected, abs=1e-5)
    assert len(lengths) == (len(continuation) + 3) // 4 and max(lengths) <= 4


def test_rolling_likelihood_counts_each_token_and_resets_documents(adapter):
    adapter.max_length = 4
    # A constant uniform model makes total likelihood depend only on token count.
    with torch.no_grad():
        adapter.model.head.weight.zero_()
    texts = ['a b c hello world ' * 4, '日本語', '']
    scores = adapter.loglikelihood_rolling([request(t) for t in texts])
    for text, score in zip(texts, scores):
        expected = -len(adapter.tok_encode(text)) * torch.log(torch.tensor(float(adapter.model.cfg.vocab_size))).item()
        assert score == pytest.approx(expected, abs=1e-4)
    assert scores[-1] == 0


def test_generation_stops_across_token_boundaries_at_eos_and_at_limit(adapter):
    ids = [adapter.tokenizer.token_to_id(c) for c in 'xyz']
    assert None not in ids

    class Scripted(torch.nn.Module):
        def __init__(self, output):
            super().__init__()
            self.output = iter(output)

        def forward(self, idx):
            assert 0 < idx.shape[1] <= adapter.max_length
            logits = torch.full((1, idx.shape[1], adapter.tokenizer.get_vocab_size()), -100.)
            logits[0, -1, next(self.output)] = 100
            return logits, None

    adapter.model = Scripted(ids)
    assert adapter.generate_until([request('hello' * 30, dict(until=['yz'], max_gen_toks=3))]) == ['x']
    adapter.model = Scripted([ids[0], adapter.eot_token_id])
    assert adapter.generate_until([request('', dict(max_gen_toks=5))]) == ['x']
    adapter.model = Scripted(ids)
    assert adapter.generate_until([request('', dict(max_gen_toks=2))]) == ['xy']
    assert adapter.generate_until([request('', dict(max_gen_toks=0))]) == ['']
    with pytest.raises(ValueError, match='greedy'):
        adapter.generate_until([request('', dict(do_sample=True))])
    with pytest.raises(ValueError, match='unsupported'):
        adapter.generate_until([request('', dict(top_p=.8))])


def test_tokenizer_identity_and_checkpoint_validation(adapter, tmp_path):
    assert len(adapter.get_model_info()['checkpoint_sha256']) == 64
    assert adapter.tokenizer_hash in adapter.tokenizer_name
    wrong = Tokenizer(models.WordLevel({'<|endoftext|>': 0, 'x': 1}, unk_token='x'))
    wrong.save(str(tmp_path / 'wrong.json'))
    with pytest.raises(ValueError, match='vocabulary'):
        ExeaLMHarness(tmp_path / 'model.pt', tmp_path / 'wrong.json')


def test_offline_harness_multiple_choice_task(adapter, tmp_path):
    from lm_eval import simple_evaluate
    from lm_eval.api.task import ConfigurableTask
    data = tmp_path / 'task.jsonl'
    data.write_text(json.dumps(dict(question='hello', choices=[' world', ' a'], answer=0)) + '\n')
    task = ConfigurableTask(config=dict(task='offline_exealm', dataset_path='json',
                                       dataset_kwargs=dict(data_files={'test': str(data)}, cache_dir=str(tmp_path / 'cache')),
                                       test_split='test', output_type='multiple_choice',
                                       doc_to_text='{{question}}', doc_to_choice=lambda d: d['choices'],
                                       doc_to_target='{{answer}}', num_fewshot=0))
    result = simple_evaluate(model=adapter, tasks=[task], limit=1, bootstrap_iters=0, log_samples=False)
    assert result['results']['offline_exealm']['acc,none'] in (0., 1.)
    assert result['config']['tokenizer_sha256'] == adapter.tokenizer_hash


def test_module_cli_writes_results_with_callable_task_metadata(adapter, tmp_path):
    import yaml
    from exealm.eval.harness import main
    tasks = tmp_path / 'tasks'
    tasks.mkdir()
    data = tasks / 'data.jsonl'
    data.write_text(json.dumps(dict(question='hello', choices=[' world', ' a'], answer=0)) + '\n')
    (tasks / 'helper.py').write_text('def identity(dataset):\n    return dataset\n')
    config = dict(task='exealm_cli_fixture', dataset_path='json',
                  dataset_kwargs=dict(data_files={'test': str(data)}, cache_dir=str(tasks / 'cache')),
                  test_split='test', output_type='multiple_choice', doc_to_text='question',
                  doc_to_target='answer', doc_to_choice='choices', num_fewshot=0)
    (tasks / 'fixture.yaml').write_text(yaml.safe_dump(config) + '\nprocess_docs: !function helper.identity\n')
    output = tmp_path / 'result.json'
    main(['--ckpt', str(tmp_path / 'model.pt'), '--tokenizer', str(tmp_path),
          '--tasks', 'exealm_cli_fixture', '--include_path', str(tasks), '--limit', '1',
          '--max_length', '16', '--out', str(output)])
    saved = json.loads(output.read_text())
    assert saved['results']['exealm_cli_fixture']['acc,none'] in (0., 1.)
    assert 'identity' in saved['configs']['exealm_cli_fixture']['process_docs']
