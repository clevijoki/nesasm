use std::fmt;
use std::str;

struct Node<'a>
{
	filename: &'a str,
	line: u32,
	parent: Option<&'a Node<'a>>,
}

fn print_callstack(node: &Node, out: &mut fmt::Write) -> Result<u32, fmt::Error>
{
	let depth = match node.parent
	{
		Some(parent) => print_callstack(parent, out)?,
		None => 0,
	};

	for _ in 0..depth
	{
		out.write_str(" ")?;
	}

	out.write_fmt(format_args!("{}:{}\n", node.filename, node.line))?;
	return Ok(depth + 1);
}

fn new_node<'a>(filename: &'a str, parent: Option<&'a Node>) -> Node<'a>
{
	let line: u32 = 0;

	Node {
		filename,
		line,
		parent,
	}
}

struct Scope<'a>
{
    name:&'a str,
    parent:Option<&'a Scope<'a>>,
    children:Vec<Scope<'a>>,
}

#[derive(Debug)]
#[derive(PartialEq)]
enum Token<'a>
{
    LParen,
    RParen,
    Plus,
    Minus,
    Star,
    Slash,
    Equals,
    Bang,
    Colon,
    SemiColon,
    Dot,
    Hash,
    Comma,
    Symbol(&'a str),
    Numeric(&'a str),
    Unknown(char)
}

struct Lexer<'a>
{
    iter: str::Chars<'a>,
    newline:bool,
    spaces:u32,
    space_tab_count:u32,
    scope:u32,
    current:Option<Token<'a>>,
    peek:Option<Token<'a>>,
}

impl<'a> Lexer<'a>
{
    fn new(source:&'a str) -> Lexer
    {
        let mut result = Lexer { iter: source.chars(), newline: true, spaces: 0, space_tab_count: 4, scope: 0, current: None, peek: None };
        result.advance(); // skip one
        return result;
    }

    fn consume_some(&mut self, result:Token<'a>) -> Option<Token<'a>>
    {
        self.newline = false;
        self.scope = self.spaces / self.space_tab_count;
        self.iter.next();
        Some(result)
    }

    fn advance(&mut self)
    {
        self.current = self.peek;

        self.peek = match self.iter.clone().next()
        {
            Some('(') => self.consume_some(Token::LParen),
            Some(')') => self.consume_some(Token::RParen),
            Some('+') => self.consume_some(Token::Plus),
            Some('-') => self.consume_some(Token::Minus),
            Some('*') => self.consume_some(Token::Star),
            Some('/') => self.consume_some(Token::Slash),
            Some('=') => self.consume_some(Token::Equals),
            Some('!') => self.consume_some(Token::Bang),
            Some(':') => self.consume_some(Token::Colon),
            Some(';') => self.consume_some(Token::SemiColon),
            Some('.') => self.consume_some(Token::Dot),
            Some('#') => self.consume_some(Token::Hash),
            Some(',') => self.consume_some(Token::Comma),
            Some('\n') => 
            {
                self.newline = true;
                self.scope = 0;
                self.spaces = 0;
                self.iter.next();
                self.next()
            }
            Some(' ') =>
            {
                self.spaces += 1;
                self.iter.next();
                self.next()
            }
            Some('\t') =>
            {
                self.spaces += self.space_tab_count;
                self.iter.next();
                self.next()
            }
            Some(c) => 
            {
                if c == '_' || c.is_alphabetic() 
                {
                    let word_str = self.iter.clone().as_str();
                    let mut word_len = 0;

                    while self.iter.clone().next().map_or(false, |c| c == '_' || c.is_alphabetic() || c.is_numeric() ) {
                        word_len += 1;
                        self.iter.next();
                    }

                    Some(Token::Symbol(&word_str[..word_len]))
                }
                else if c.is_numeric()
                {
                    let word_str = self.iter.as_str();
                    let mut word_len = 0;
                    while self.iter.clone().next().map_or(false, |c| c.is_numeric() ) {
                        word_len += 1;
                        self.iter.next();
                    }

                    Some(Token::Numeric(&word_str[..word_len]))
                }
                else
                {
                    Some(Token::Unknown(self.iter.next().unwrap()))
                }
            },
            None => None
        };
    }
}


impl<'a> Iterator for Lexer<'a>
{
    type Item = Token<'a>;

    fn next(&mut self) -> Option<Self::Item>
    {
        self.advance();
        self.current
    }
}

impl<'a> Scope<'a>
{
    fn new(name:&'a str, parent:Option<&'a Scope>) -> Scope<'a>
    {
        let children = Vec::new();

        return Scope { name, parent, children };
    }
}

enum FlagState
{
    On,
    Off,
    Unknown,
}

enum ByteValue
{
    Constant(u8),
    Unknown,
}

// represents the state of the CPU when statically analyzing
struct CPU
{
    negative:FlagState,
    overflow:FlagState,
    always_set:FlagState,
    clear_if_interrupt:FlagState,
    decimal_mode:FlagState,
    interupt_disable:FlagState,
    zero:FlagState,
    carry:FlagState,

    a:ByteValue,
    x:ByteValue,
    y:ByteValue,
}

enum OpData
{
    Invalid,
    Immediate(u8),
    ZeroPage(u8),
    ZeroPageX(u8),
    Absolute(u16),
    AbsoluteX(u16),
    AbsoluteY(u16),
    IndirectX(u8),
    IndirectY(u8),
}

enum Expr
{

}

type EvalFunc = Fn(&OpCode, &mut CPU);

struct OpCode
{
    code:u8,
    data: OpData,
    evaluate:Box<EvalFunc>
}

impl OpCode
{
    fn new(code:u8, data:OpData, evaluate:Box<EvalFunc>) -> OpCode
    {
        return OpCode { code, data, evaluate };
    }    
}


struct State {}
fn new_state() -> State
{
    State {}
}

fn compile(_source: &str, _state: &mut State, _filename: &str)
{
}

#[cfg(test)]
mod tests
{
	use super::*;

	#[test]
	fn comment()
	{
	}

	#[test]
	fn op_adc()
	{
	}

	#[test]
	fn print_callstack()
	{
        let mut n1 = new_node("n1", None);
        n1.line = 1;
        let mut n2 = new_node("n2", Some(&n1));
        n2.line = 2;

        let mut s = String::new();

        ::print_callstack(&n2, &mut s).expect("Formatting failed");

		assert_eq!(s, "n1:1\n n2:2\n");
	}

    #[test]
    fn lexer()
    {
        let lexer = Lexer::new("(a + b*c)");

        let collection:Vec<Token> = lexer.collect();
        let expected = vec![Token::LParen, Token::Symbol("a"), Token::Plus, Token::Symbol("b"), Token::Star, Token::Symbol("c"), Token::RParen];

        assert_eq!(expected, collection);
    }

}
