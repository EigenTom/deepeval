import openai
from typing import List
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.benchmarks import BigBenchHard
from typing import Dict, Any, List
import asyncio

# ---- OpenAI client points to your local server -----------------------------
openai.api_key = "EMPTY"          # the server ignores auth, keep a dummy key
openai.base_url = "http://localhost:5001/v1"
model_name = "/data/yi/verl-tool/checkpoints/evaluation/Qwen2.5-Coder-7B-Inst-Interpreter-thinking-valid-tool"
system_prompt = """\
Answer the given coding question. You must conduct reasoning about the problem and then provide the final program as answer. 
During the thinking process, you can write test cases or test your current solutions using a testing tool. if you want to test any python code, writing it inside ```python and ``` tags following with "```output". 
The code between "```python" and "``````output" will then be executed, and the terminal output (standard output and standard error) will be provided to you. 
Each program between ```python and ``` tags are independent program. You can test Python codes as many times as you want. 
If you find no further code execution needed, you can then give your final solution in a markdown code block like this: ```python\nyour code here\n``` without appending anything. 
The final program will be evaluated against the hidden test cases. If the final program passes all the test cases, you will get a reward. If the final program fails any of the test cases, you will get a penalty.
"""

# ---------------------------------------------------------------------------

class AceCoder(DeepEvalBaseLLM):
    """
    DeepEval wrapper around an OpenAI-style chat model served at localhost:5001
    """
    def __init__(self, model_name: str = "gpt-4o-mini", max_tokens: int = 100, system_prompt: str = system_prompt):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt

    # DeepEval looks for this to know what to print in reports
    def get_model_name(self):
        return self.model_name

    def load_model(self):
        return self.model

    # ---------- single prompt ------------------------------------------------
    def generate(self, prompt: str) -> str:
        prompt = "Provided is the coding question:\n" + prompt
        
        response = openai.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},    
                {"role": "user", "content": prompt}
            ],
            temperature=0.95,
        )
        return response.choices[0].message.content.strip()

    # ---------- async prompt -------------------------------------------------
    async def a_generate(self, prompt: str) -> str:
        prompt = "Provided is the coding question:\n" + prompt
        
        response = await openai.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},    
                {"role": "user", "content": prompt}
            ],
            temperature=0.95,
        )
        return response.choices[0].message.content.strip()

    async def _batch_generate(self, prompts: List[str]) -> List[str]:
        return await asyncio.gather(*(self.a_generate(p) for p in prompts))

    # ---------- batch prompt (async fallback) ------------------------------
    def batch_generate(self, prompts: List[str]) -> List[str]:    
        return asyncio.run(self._batch_generate(prompts))
    
# ---------------------- run BigBenchHard benchmark ------------------------------
benchmark = BigBenchHard()
ace_coder = AceCoder(model_name="gpt-4o-mini", max_tokens=100)

results = benchmark.evaluate(model=ace_coder, batch_size=5)
print("Overall Score:", results)
