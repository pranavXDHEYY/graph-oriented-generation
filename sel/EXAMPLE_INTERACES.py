# decomposer.py
def decompose(prompt: str) -> list[Primitive]:
    """
    Input:  "I miss my hometown"
    Output: [
        Primitive(layer="0b", word="GRIEF", weight=0.8),
        Primitive(layer="0a", word="PLACE", weight=0.7),
        Primitive(layer="0b", word="NOSTALGIA", weight=0.9),
    ]
    """

# reasoner.py  
def reason(primitives: list[Primitive]) -> list[Concept]:
    """
    Input:  [GRIEF, PLACE, NOSTALGIA]
    Output: [
        Concept(name="exile", rule_class="D", confidence=0.85),
        Concept(name="homesickness", rule_class="A", confidence=0.92),
    ]
    """

# membrane.py
def render(concepts: list[Concept], context: str) -> str:
    """
    Input:  [exile, homesickness], "I miss my hometown"
    Output: "That longing for a place that shaped you —
             it's one of the most human feelings there is."
    """

# router.py
def process(prompt: str) -> str:
    primitives = decompose(prompt)
    concepts   = reason(primitives)
    response   = render(concepts, prompt)
    return response