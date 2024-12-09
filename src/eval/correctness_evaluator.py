from llama_index.core import Settings
from llama_index.core.evaluation import CorrectnessEvaluator

from components.set_llm import set_llm

set_llm(model='llama3.2')
llm = Settings.llm
evaluator = CorrectnessEvaluator(llm=llm)

query = (
    "Can you explain the theory of relativity proposed by Albert Einstein in"
    " detail?"
)

reference = """
Certainly! Albert Einstein's theory of relativity consists of two main components: special relativity and general relativity. Special relativity, published in 1905, introduced the concept that the laws of physics are the same for all non-accelerating observers and that the speed of light in a vacuum is a constant, regardless of the motion of the source or observer. It also gave rise to the famous equation E=mc², which relates energy (E) and mass (m).

General relativity, published in 1915, extended these ideas to include the effects of gravity. According to general relativity, gravity is not a force between masses, as described by Newton's theory of gravity, but rather the result of the warping of space and time by mass and energy. Massive objects, such as planets and stars, cause a curvature in spacetime, and smaller objects follow curved paths in response to this curvature. This concept is often illustrated using the analogy of a heavy ball placed on a rubber sheet, causing it to create a depression that other objects (representing smaller masses) naturally move towards.

In essence, general relativity provided a new understanding of gravity, explaining phenomena like the bending of light by gravity (gravitational lensing) and the precession of the orbit of Mercury. It has been confirmed through numerous experiments and observations and has become a fundamental theory in modern physics.
"""

response = """
Certainly! Albert Einstein's theory of relativity consists of two main components: special relativity and general relativity. Special relativity, published in 1905, introduced the concept that the laws of physics are the same for all non-accelerating observers and that the speed of light in a vacuum is a constant, regardless of the motion of the source or observer. It also gave rise to the famous equation E=mc², which relates energy (E) and mass (m).

However, general relativity, published in 1915, extended these ideas to include the effects of magnetism. According to general relativity, gravity is not a force between masses but rather the result of the warping of space and time by magnetic fields generated by massive objects. Massive objects, such as planets and stars, create magnetic fields that cause a curvature in spacetime, and smaller objects follow curved paths in response to this magnetic curvature. This concept is often illustrated using the analogy of a heavy ball placed on a rubber sheet with magnets underneath, causing it to create a depression that other objects (representing smaller masses) naturally move towards due to magnetic attraction.
"""

result = evaluator.evaluate(
    query=query,
    response=response,
    reference=reference,
)
print(result.score)
print(result.feedback)