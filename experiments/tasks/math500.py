"""MATH-500 (HuggingFaceH4/MATH-500, all 500 problems, greedy) for the PIG decoders.

Prompts. Each prompt is wrapped in the model's chat template as the user turn:
    LLaDA-8B-Instruct          four-shot with lm-eval's Minerva exemplars in the Problem:/Solution: format
                               (the fixed list in lm_eval/tasks/minerva_math/utils.py), then
                               "Problem:\\n<problem>\\n\\nSolution:".
    Nemotron-Labs-Diffusion-3B zero-shot: the problem followed by "\\nPlease reason step by step.".

Grading. The generation is cut at the stop sequences; the answer is the "Final Answer: The final answer
is X" span if present, else the last \\boxed{...} (strict), else the last inline-math span or the last
number (flexible only). Prediction and gold are Minerva-normalised and compared by string, then by sympy
(difference simplifies to zero). Gold is the normalised `answer` field. The strict score is the primary
metric for LLaDA (boxed compliance is near 100%); both are recorded.

The extractor is the one the paper's numbers were produced with, kept verbatim: a "Final Answer" span that
itself contains \\boxed{...} is not unwrapped, so such an answer scores 0 even when correct.
"""
import re

NAME = "math500"
N = 500
SHOTS = {"llada": 4, "nemotron": 0}
METRICS = ("strict", "flex")
PRIMARY = "strict"
STOPS = ("\nProblem:", "Problem:", "<|endoftext|>", "<|eot_id|>")  # lm-eval until: "Problem:"


def load_problems():
    """The 500 problems in order: dicts with task_id, problem and answer."""
    from datasets import load_dataset

    return [dict(task_id=f"math500/{i}", problem=ex["problem"], answer=ex["answer"])
            for i, ex in enumerate(load_dataset("HuggingFaceH4/MATH-500", split="test"))]


def last_boxed(text):
    """Return the content of the LAST \\boxed{...}, brace-balanced."""
    i = text.rfind("\\boxed")
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth, k = 0, j
    while k < len(text):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1:k]
        k += 1
    return None


def normalize(a):
    """Minerva-style normalisation for string comparison of MATH answers."""
    if a is None:
        return None
    s = a.strip()
    for d in ("\\(", "\\)", "\\[", "\\]"):        # inline/display math delimiters
        s = s.replace(d, "")
    s = re.sub(r"\\(?:left|right|!|,|;|\s)", "", s)
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mbox\{([^}]*)\}", r"\1", s)
    s = s.replace("^\\circ", "").replace("^{\\circ}", "")
    s = s.replace("\\$", "").replace("$", "").replace("\\%", "").replace("%", "")
    s = re.sub(r"\\?\b(?:dollars|cm|m|km|kg|inches|feet|units|degrees)\b", "", s)
    s = s.replace(" ", "").rstrip(".").rstrip()
    s = re.sub(r"^\\{?(.*?)\\}?$", r"\1", s) if s.startswith("{") and s.endswith("}") else s
    s = re.sub(r"\.0+$", "", s)                       # 3.0 -> 3
    s = re.sub(r"^(-?)0+(\d)", r"\1\2", s)            # 007 -> 7
    return s


def is_equiv(a, b):
    """lm-eval is_equiv: string match, else sympy-difference-simplifies-to-zero."""
    if a is None or b is None:
        return False
    if a == b:
        return True
    try:
        from sympy.parsing.latex import parse_latex
        from sympy import simplify
        return bool(simplify(parse_latex(a) - parse_latex(b)) == 0)
    except Exception:
        return False


def extract(text):
    """-> (strict, flex) normalised answers; strict is None when nothing is boxed or declared final."""
    cut = text
    for st in STOPS:
        i = cut.find(st)
        if i != -1:
            cut = cut[:i]
    m = re.search(r"Final Answer: The final answer is(.*?). I hope it is correct.", cut)
    if m:                                              # lm-eval get_unnormalized_answer()
        v = normalize(m.group(1))
        if v:
            return v, v
    strict = normalize(last_boxed(cut))                # lm-eval last_boxed_only_string()
    if strict is not None:
        return strict, strict
    m = re.findall(r"\$([^$]{1,60})\$", cut)          # flexible: last inline-math span
    if m:
        return None, normalize(m[-1])
    m2 = re.findall(r"-?[\d,]*\.?\d+", cut)
    return None, (normalize(m2[-1]) if m2 else None)


# The MINERVA 4-shot exemplars, i.e. lm_eval/tasks/minerva_math/utils.py list_fewshot_samples(): Fast-dLLM
# reports "MATH (4-shot)" via lm-eval, so the same fixed set and the same "Problem:/Solution:" template.
MINERVA_FEWSHOT = [
    ("Find the domain of the expression  $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.",
     "The expressions inside each square root must be non-negative. Therefore, $x-2 \\ge 0$, so "
     "$x\\ge2$, and $5 - x \\ge 0$, so $x \\le 5$. Also, the denominator cannot be equal to zero, "
     "so $5-x>0$, which gives $x<5$. Therefore, the domain of the expression is $\\boxed{[2,5)}$.\n"
     "Final Answer: The final answer is $[2,5)$. I hope it is correct."),
    ("If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find "
     "$\\det (\\mathbf{A} \\mathbf{B}).$",
     "We have that $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B}) = "
     "(2)(12) = \\boxed{24}.$\n"
     "Final Answer: The final answer is $24$. I hope it is correct."),
    ("Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights "
     "instead, how many times must Terrell lift them in order to lift the same total weight?",
     "If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\\cdot 12\\cdot20="
     "480$ pounds of weight.  If he lifts two 15-pound weights instead for $n$ times, he will "
     "lift a total of $2\\cdot15\\cdot n=30n$ pounds of weight.  Equating this to 480 pounds, we "
     "can solve for $n$: \\begin{align*} 30n&=480\\\\ \\Rightarrow\\qquad n&=480/30=\\boxed{16} "
     "\\end{align*}\n"
     "Final Answer: The final answer is $16$. I hope it is correct."),
    ("If the system of equations\n\n\\begin{align*}\n6x-4y&=a,\\\\\n6y-9x &=b.\n\\end{align*}has "
     "a solution $(x, y)$ where $x$ and $y$ are both nonzero, find $\\frac{a}{b},$ assuming $b$ "
     "is nonzero.",
     "If we multiply the first equation by $-\\frac{3}{2}$, we obtain\n\n$$6y-9x=-\\frac{3}{2}a."
     "$$Since we also know that $6y-9x=b$, we have\n\n$$-\\frac{3}{2}a=b\\Rightarrow\\frac{a}{b}"
     "=\\boxed{-\\frac{2}{3}}.$$\n"
     "Final Answer: The final answer is $-\\frac{2}{3}$. I hope it is correct."),
]


def build_fewshot(k=4):
    return "\n\n".join(f"Problem:\n{p}\n\nSolution:\n{s}" for p, s in MINERVA_FEWSHOT[:k])


def prompt_text(model, problem):
    """LLaDA: the four exemplars, then "Problem:\\n<problem>\\n\\nSolution:". Nemotron: the problem, then
    "\\nPlease reason step by step." (zero-shot, the model's own evaluation practice)."""
    k = SHOTS[model]
    if k == 0:
        return problem["problem"] + "\nPlease reason step by step."
    return build_fewshot(k) + "\n\n" + f"Problem:\n{problem['problem'].strip()}\n\nSolution:"


def score(model, problem, completion):
    strict, flex = extract(completion)
    gold = normalize(problem["answer"])
    return {"strict": is_equiv(strict, gold), "flex": is_equiv(flex, gold)}
