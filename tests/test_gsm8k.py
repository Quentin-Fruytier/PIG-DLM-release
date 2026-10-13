"""The GSM8K prompts and graders on hand-made text (no dataset, no model)."""
import pytest

from experiments.tasks import gsm8k

PROBLEM = dict(task_id="gsm8k/0", question="Ann has 3 apples and buys 4 more. How many apples?",
               answer="She has 3 + 4 = <<3+4=7>>7 apples.\n#### 7")
TRAIN = [dict(question=f"q{i}?", answer=f"r{i} = <<{i}>>{i}.\n#### {i}") for i in range(12)]


def test_gold_strips_thousands_separators():
    assert gsm8k.gold("... \n#### 1,234") == "1234"
    assert gsm8k.gold(PROBLEM["answer"]) == "7"


def test_strict_is_the_first_hash_line_and_flex_the_last_number():
    assert gsm8k.extract("3 + 4 = 7.\n#### 7") == ("7", "7")
    assert gsm8k.extract("It costs $1,000.50 in total.\n#### 1,000.50.") == ("1000.50", "1000.50")
    assert gsm8k.extract("So 12 minus 15 gives -3.") == (None, "-3")
    assert gsm8k.extract("#### 7 and then I checked again and got 9") == ("7", "9")
    assert gsm8k.extract("no digits here") == (None, None)


def test_generation_is_cut_at_the_stop_sequences():
    assert gsm8k.extract("#### 7\nQuestion: next one\nAnswer: #### 9") == ("7", "7")
    assert gsm8k.extract("the answer is 7<|eot_id|>#### 9") == (None, "7")


def test_score_records_both_extractors():
    assert gsm8k.score("llada", PROBLEM, "3 + 4 = 7\n#### 7") == {"flex": True, "strict": True}
    assert gsm8k.score("llada", PROBLEM, "So she has 7 apples.") == {"flex": True, "strict": False}
    assert gsm8k.score("nemotron", PROBLEM, "\\boxed{8}") == {"flex": False, "strict": False}


def test_fewshot_block_is_five_fixed_exemplars_in_the_lm_eval_format():
    block = gsm8k.build_fewshot(TRAIN, 5)
    assert block == gsm8k.build_fewshot(TRAIN, 5)                         # the seed fixes the choice
    parts = block.split("\n\n")
    assert len(parts) == 5 and len(set(parts)) == 5
    for part in parts:
        q, a, h = part.split("\n")
        assert q.startswith("Question: q") and a.startswith("Answer: r") and h.startswith("#### ")
        assert "<<" in a                                                   # the reasoning keeps the calculator marks


def test_prompts_follow_each_models_protocol():
    assert gsm8k.prompt_text("nemotron", PROBLEM) == PROBLEM["question"]   # zero-shot, the question as is
    text = gsm8k.prompt_text("llada", PROBLEM, exemplars="EXEMPLARS")
    assert text == "EXEMPLARS\n\nQuestion: " + PROBLEM["question"] + "\nAnswer:"
    assert gsm8k.SHOTS == {"llada": 5, "nemotron": 0} and gsm8k.N == 500


def test_unknown_model_has_no_shot_count():
    with pytest.raises(KeyError):
        gsm8k.prompt_text("dream", PROBLEM)
