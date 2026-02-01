# Learning-With-AI
---
## The AST Manager
**The Challenge:** Building a bi-directional synchronization engine where a JSON object (the AST) is the "Single Source of Truth" for both a Visual Block Editor and a Text Editor.

### Learning Goals:
1. Master State Management patterns (specifically Redux or Proxy patterns).
2. Understand how to design a JSON Schema that is flexible enough for UI rendering but strict enough for compilation.
3. Learn how to handle Race Conditions when both editors try to update the state simultaneously.

---

## The Polyglot Transpiler

**The Challenge:** Converting a custom JSON AST into syntactically correct, production-ready code for multiple languages (Java/Groovy and Python) without using proprietary lock-in.

### Learning Goals:
1. Understand Compiler Theory basics: Tokenizing vs. Parsing vs. Printing.
2. Master the Visitor Pattern (the standard for walking Syntax Trees).
3. Learn how to build modular "Printers" that share core logic but output different syntax.
