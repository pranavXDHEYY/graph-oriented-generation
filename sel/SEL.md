/sel
  /core
    primitive_graph.json      ← the knowledge graph
    composition_rules.json    ← the rule taxonomy
    decomposer.py             ← prompt → primitives
    reasoner.py               ← primitives → concepts
    membrane.py               ← concepts → English
    router.py                 ← orchestrates the pipeline
  /data
    wierzbicka_primitives.json
    cowen_keltner_emotions.json
    ground_truth_layer1.json  ← from exp 19b/20
    validated_compositions.json
  /tests
    test_decomposer.py
    test_reasoner.py
    test_membrane.py
    test_pipeline.py
  /experiments
    ← your existing exp 1-22 stay in symbol_distillation
    ← SEL is the APPLICATION built from those findings
  README.md
  CLAUDE.md                   ← instructions for Claude Code