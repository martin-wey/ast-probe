from tree_sitter import Language

Language.build_library(
  # Store the library in the `build` directory
  'grammars/languages.so',

  # Include one or more languages
  [
    'grammars/tree-sitter-python',
    'grammars/tree-sitter-javascript',
    'grammars/tree-sitter-go',
    'grammars/tree-sitter-php',
    'grammars/tree-sitter-ruby',
    'grammars/tree-sitter-java'
  ]
)
