import unittest

import matplotlib.pyplot as plt
import networkx as nx
from tree_sitter import Language, Parser

from src.data.code2ast import code2ast, get_tokens_ast
from src.data.utils import remove_comments_and_docstrings_python, \
    remove_comments_and_docstrings_java_js


def node_match_type_atts(n1, n2):
    return n1['type'] == n2['type']


code = """'''Compute the maximum'''
def max(a,b):
    s = "string"
    #compare a and b
    if a > b:
        return a
    return b
"""

code_pre_expected = """def max(a,b):
    s = "string"
    if a > b:
        return a
    return b"""

code_js = """function myFunction(p1, p2) {
/* multi-line
comments */
s = "string"
return p1 * p2;// The function returns the product of p1 and p2
}"""
plt.show()
code_js_pre_expected = """function myFunction(p1, p2) {

s = "string"
return p1 * p2;
}"""

code_go = """// Function to add two numbers
func addTwoNumbers(x, y int) int {
/*
sdsd
sdsds
sdsdsd
*/
s = "str"
sum := x + y
return sum
}"""

code_php = """function writeMsg() {
  echo 'Hello world!';
}"""

code_ruby = """def initialize(n, a)
# this is a comment
@name = n
# this is another comment
@surname = "smith"
@age  = a * DOG_YEARS

#!/usr/bin/ruby -w
# This is a single line comment.

=begin
This is a multiline comment and con spwan as many lines as you
like.
=end
end"""

code_java = """public void myMethod() {
    String mystr = "mystr";
}"""


PY_LANGUAGE = Language('grammars/languages.so', 'python')
JS_LANGUAGE = Language('grammars/languages.so', 'javascript')
GO_LANGUAGE = Language('grammars/languages.so', 'go')
PHP_LANGUAGE = Language('grammars/languages.so', 'php')
RUBY_LANGUAGE = Language('grammars/languages.so', 'ruby')
JAVA_LANGUAGE = Language('grammars/languages.so', 'java')
parser = Parser()
parser.set_language(PY_LANGUAGE)


class Code2ast(unittest.TestCase):

    def test_code2ast_java(self):
        plt.figure()
        plt.title('test_code2ast_java')
        parser = Parser()
        parser.set_language(JAVA_LANGUAGE)
        G, pre_code = code2ast(code_java, parser, lang='java')
        print(pre_code)
        nx.draw(nx.Graph(G), labels=nx.get_node_attributes(G, 'type'), with_labels=True)
        plt.show()
        print(G.nodes(data=True))
        tokens = get_tokens_ast(G, pre_code)
        print(tokens)
        self.assertTrue('"mystr"' in tokens)

    def test_code2ast_php(self):
        plt.figure()
        plt.title('test_code2ast_php')
        parser = Parser()
        parser.set_language(PHP_LANGUAGE)
        G, pre_code = code2ast(code_php, parser, lang='php')
        print(pre_code)
        nx.draw(nx.Graph(G), labels=nx.get_node_attributes(G, 'type'), with_labels=True)
        plt.show()
        print(G.nodes(data=True))
        tokens = get_tokens_ast(G, pre_code)
        print(tokens)
        self.assertTrue("'Hello world!'" in tokens)

    def test_code2ast_ruby(self):
        plt.figure()
        plt.title('test_code2ast_ruby')
        parser = Parser()
        parser.set_language(RUBY_LANGUAGE)
        G, pre_code = code2ast(code_ruby, parser, lang='ruby')
        print(pre_code)
        nx.draw(nx.Graph(G), labels=nx.get_node_attributes(G, 'type'), with_labels=True)
        plt.show()
        print(G.nodes(data=True))
        tokens = get_tokens_ast(G, pre_code)
        print(tokens)
        self.assertTrue('"smith"' in tokens)

    def test_code2ast_go(self):
        plt.figure()
        plt.title('test_code2ast_go')
        parser = Parser()
        parser.set_language(GO_LANGUAGE)
        G, pre_code = code2ast(code_go, parser, lang='go')
        print(pre_code)
        nx.draw(nx.Graph(G), labels=nx.get_node_attributes(G, 'type'), with_labels=True)
        plt.show()
        print(G.nodes(data=True))
        tokens = get_tokens_ast(G, pre_code)
        print(tokens)
        self.assertTrue('"str"' in tokens)

    def test_preprocessing(self):
        code_pre = remove_comments_and_docstrings_python(code)
        self.assertEqual(code_pre_expected, code_pre)
        code_pre = remove_comments_and_docstrings_java_js(code_js)
        self.assertEqual(code_js_pre_expected, code_pre)

    def test_code2ast_python(self):
        plt.figure()
        plt.title('test_code2ast_python')
        G, pre_code = code2ast(code, parser)
        nx.draw(nx.Graph(G), labels=nx.get_node_attributes(G, 'type'), with_labels=True)
        plt.show()
        self.assertEqual(31, len(G))
        tokens = get_tokens_ast(G, pre_code)
        print(tokens)
        self.assertTrue('"string"' in tokens)

    def test_js(self):
        plt.figure()
        plt.title('test_js I')
        parser = Parser()
        parser.set_language(JS_LANGUAGE)
        G, pre_code = code2ast(code_js, parser, 'javascript')
        print(pre_code)
        nx.draw(G, labels=nx.get_node_attributes(G, 'type'), with_labels=True)
        plt.show()
        tokens = get_tokens_ast(G, pre_code)
        print(tokens)
        self.assertTrue('"string"' in tokens)

    def test_str_ast(self):
        code = """def split_phylogeny(p, level="s"):
    level = level+"__"
    result = p.split(level)
    return result[0]+level+result[1].split(";")[0]"""
        G, pre_code = code2ast(code, parser)
        tokens = get_tokens_ast(G, pre_code)
        self.assertTrue('"__"' in tokens)
        self.assertTrue('";"' in tokens)
        self.assertTrue('"s"' in tokens)
        plt.figure()
        plt.title('test_str_ast')
        nx.draw(nx.Graph(G), labels=nx.get_node_attributes(G, 'type'), with_labels=True)
        plt.show()
