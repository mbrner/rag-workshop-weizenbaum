import re
from dataclasses import dataclass
from typing import Set, List, Union

# Constants
OPERATORS = {'&': 2, '|': 1, '!': 3}
TAG_PATTERN = re.compile(r'^[a-z][a-z0-9-_]*$')

class ParseError(ValueError):
    """Custom exception for parsing errors"""
    pass

@dataclass
class Token:
    """Represents a token in the expression"""
    value: str
    is_operator: bool = False

# AST Nodes
class Node:
    def evaluate(self, tags: Set[str]) -> bool:
        raise NotImplementedError

@dataclass
class BinaryNode(Node):
    left: Node
    right: Node

class AndNode(BinaryNode):
    def evaluate(self, tags: Set[str]) -> bool:
        return self.left.evaluate(tags) and self.right.evaluate(tags)

class OrNode(BinaryNode):
    def evaluate(self, tags: Set[str]) -> bool:
        return self.left.evaluate(tags) or self.right.evaluate(tags)

@dataclass
class NotNode(Node):
    child: Node
    def evaluate(self, tags: Set[str]) -> bool:
        return not self.child.evaluate(tags)

@dataclass
class TagNode(Node):
    name: str
    def evaluate(self, tags: Set[str]) -> bool:
        return self.name in tags

class ExpressionParser:
    @staticmethod
    def tokenize(expression: str) -> List[Token]:
        if not expression.strip():
            raise ParseError('Empty expression')

        tokens = re.findall(r'(\(|\)|\||&|!|[\w-]+|[^ \t\n\r\f\v\w&|!()]+)', expression)
        return [
            Token(t, t in OPERATORS or t in '()')
            for t in tokens
        ]

    @staticmethod
    def validate_tag(token: str) -> bool:
        return bool(TAG_PATTERN.match(token))

    @staticmethod
    def check_parentheses(tokens: List[Token]) -> None:
        count = 0
        for token in tokens:
            if token.value == '(':
                count += 1
            elif token.value == ')':
                count -= 1
            if count < 0:
                raise ParseError('Unmatched closing parenthesis')
        if count != 0:
            raise ParseError('Unmatched opening parenthesis')

    def parse(self, tokens: List[Token]) -> Node:
        stack = []
        postfix = []

        # Convert to postfix notation
        for token in tokens:
            if token.value == '(':
                stack.append(token)
            elif token.value == ')':
                while stack and stack[-1].value != '(':
                    postfix.append(stack.pop())
                if not stack:
                    raise ParseError('Mismatched parentheses')
                stack.pop()  # Remove '('
            elif token.is_operator:
                while (stack and stack[-1].value in OPERATORS and
                       OPERATORS[token.value] <= OPERATORS[stack[-1].value]):
                    postfix.append(stack.pop())
                stack.append(token)
            else:
                if not self.validate_tag(token.value):
                    raise ParseError(f'Invalid tag name: {token.value}')
                postfix.append(token)

        while stack:
            if stack[-1].value == '(':
                raise ParseError('Mismatched parentheses')
            postfix.append(stack.pop())

        # Build AST
        node_stack = []
        for token in postfix:
            if token.value == '&':
                right, left = node_stack.pop(), node_stack.pop()
                node_stack.append(AndNode(left, right))
            elif token.value == '|':
                right, left = node_stack.pop(), node_stack.pop()
                node_stack.append(OrNode(left, right))
            elif token.value == '!':
                node_stack.append(NotNode(node_stack.pop()))
            else:
                node_stack.append(TagNode(token.value))

        if not node_stack:
            raise ParseError('Invalid expression')
        return node_stack[0]

def normalize_tag(tag: str) -> str:
    """Normalize tag names to a consistent format"""
    tag = tag.lower()
    tag = re.sub(r'[^a-z0-9-_]+', '-', tag)
    tag = re.sub(r'-+', '-', tag.strip('-'))

    if not tag or not tag[0].isalpha():
        raise ParseError(f"Invalid tag name after normalization: {tag}")
    return tag

def evaluate(expression: str, tags: Set[str]) -> bool:
    """
    Evaluate a boolean expression with the given set of tags.

    Args:
        expression: A boolean expression string
        tags: Set of tag names that are considered True

    Returns:
        bool: Result of evaluating the expression

    Raises:
        ParseError: If the expression is invalid
    """
    parser = ExpressionParser()
    normalized_tags = {normalize_tag(tag) for tag in tags}
    tokens = parser.tokenize(expression)
    parser.check_parentheses(tokens)
    ast = parser.parse(tokens)
    return ast.evaluate(normalized_tags)


def evaluate_many(expression: str, tags: List[Set[str]]) -> List[bool]:
    """
    Evaluate a boolean expression with the given set of tags.

    Args:
        expression: A boolean expression string
        tags: Set of tag names that are considered True

    Returns:
        bool: Result of evaluating the expression

    Raises:
        ParseError: If the expression is invalid
    """
    parser = ExpressionParser()
    tokens = parser.tokenize(expression)
    parser.check_parentheses(tokens)
    ast = parser.parse(tokens)
    matches = []
    for tag_set in tags:
        normalized_tags = {normalize_tag(tag) for tag in tag_set}
        matches.append(ast.evaluate(normalized_tags))
    return matches

