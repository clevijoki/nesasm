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
#[derive(Clone)]
enum Token<'a>
{
    Word(&'a str),
    Numeric(&'a str),
    Symbol(char)
}

struct Lexer<'a>
{
    iter: str::Chars<'a>,
    newline:bool,
    lead_spaces:u32, // used to determine scope indenting
    space_tab_count:u32,
    peek_scope:u32,
    peek:Option<Token<'a>>,
}

impl<'a> Lexer<'a>
{
    fn new(source:&'a str) -> Lexer
    {
        let mut result = Lexer { iter: source.chars(), newline: true, lead_spaces: 0, space_tab_count: 4, peek_scope: 0, peek: None };
        result.peek = result.next_token(); // skip one
        return result;
    }

    fn some(&mut self, result:Token<'a>) -> Option<Token<'a>>
    {
        self.newline = false;
        self.peek_scope = self.lead_spaces / self.space_tab_count;
        Some(result)
    }

    fn next_token(&mut self) -> Option<Token<'a>>
    {
        match self.iter.clone().next()
        {
            Some('\n') => 
            {
                println!("Found newline:{} '{}'", self.lead_spaces, self.iter.as_str());
                self.newline = true;
                self.peek_scope = 0;
                self.lead_spaces = 0;
                self.iter.next();
                self.next_token()
            }
            Some(' ') =>
            {
                if self.newline 
                {
                    println!("Found space:{} '{}'", self.lead_spaces, self.iter.as_str());
                    self.lead_spaces += 1;                
                }

                self.iter.next();
                self.next_token()
            }
            Some('\t') =>
            {
                if self.newline
                {
                    println!("Found tab:{} '{}'", self.lead_spaces, self.iter.as_str());
                    self.lead_spaces += self.space_tab_count;
                }

                self.iter.next();
                self.next_token()
            }
            Some(c) => 
            {
                println!("Found symbol:{} '{}'", self.lead_spaces, self.iter.as_str());
                if c == '_' || c.is_alphabetic() 
                {
                    let word_str = self.iter.clone().as_str();

                    while self.iter.clone().next().map_or(false, |c| c == '_' || c.is_alphabetic() || c.is_numeric() ) {
                        self.iter.next();
                    }

                    let result = Token::Word(&word_str[..word_str.len()-self.iter.as_str().len()]);
                    self.some(result)
                }
                else if c.is_numeric()
                {
                    let word_str = self.iter.clone().as_str();
                    let mut word_len = 0;
                    while self.iter.clone().next().map_or(false, |c| c.is_numeric() ) {
                        word_len += 1;
                        self.iter.next();
                    }

                    self.some(Token::Numeric(&word_str[..word_len]))
                }
                else
                {
                    self.iter.next();
                    self.some(Token::Symbol(c))
                }
            },
            None => None
        }
    }
}

impl<'a> Iterator for Lexer<'a>
{
    type Item = (u32,Token<'a>);
    fn next(&mut self) -> Option<Self::Item>
    {
        let result = self.peek.clone();
        let scope = self.peek_scope;
        self.peek = self.next_token();

        match result {
            Some(item) => Some((scope, item)),
            None => None
        }
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

#[derive(Debug)]
#[derive(PartialEq)]
#[derive(Clone)]
#[derive(Copy)]
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

    memory:[ByteValue;64*1024],
}

trait Expression
{
    fn evaluate(cpu:&mut CPU);
}

#[derive(Clone)]
enum OpData
{
    Unknown,
    Immediate(u8),
    ZeroPage(u8),
    ZeroPageX(u8),
    ZeroPageY(u8),
    Absolute(u16),
    AbsoluteX(u16),
    AbsoluteY(u16),
    // IndirectIndexed(u8),
    // IndexedIndirect(u8),
}

enum OpCode
{
    INX,
    INY,
    DEX,
    DEY,
    TAX,
    TXA,
    TAY,
    TYA,
    LDA,
    LDX,
    LDY,
    STA,
    STY,
    STX,
}

fn add_value(base:ByteValue, val:u8) -> ByteValue
{
    match base 
    {
        ByteValue::Constant(x) => ByteValue::Constant(x + val),
        ByteValue::Unknown => ByteValue::Unknown
    }
}

fn add_zp_address(base:u8, offset:ByteValue) -> OpData
{
    match offset 
    {
        ByteValue::Constant(x) => OpData::Absolute((base + x) as u16),
        ByteValue::Unknown => OpData::Unknown
    }
}

fn add_address(base:u16, offset:ByteValue) -> OpData
{
    match offset 
    {
        ByteValue::Constant(x) => OpData::Absolute(base + (x as u16)),
        ByteValue::Unknown => OpData::Unknown
    }
}

struct Statement
{
    code:OpCode,
    data: OpData,
}

impl Statement
{
    fn new(code:OpCode, data:OpData) -> Statement
    {
        Statement { code, data }
    }
}

struct Program
{
    statements:Vec<Statement>
}

impl Program
{
    fn new() -> Program
    {
        Program { statements:Vec::new() }
    }
}

impl CPU
{
    fn new() -> CPU
    {
        CPU { 
            negative:FlagState::Off, 
            overflow:FlagState::Off, 
            always_set:FlagState::Off,
            clear_if_interrupt:FlagState::Off,
            decimal_mode:FlagState::Off,
            interupt_disable:FlagState::Off,
            zero:FlagState::Off,
            carry:FlagState::Off,
            a:ByteValue::Constant(0),
            x:ByteValue::Constant(0),
            y:ByteValue::Constant(0),
            memory:[ByteValue::Constant(0);64*1024] 
        }
    }

    fn invalidate_memory(&mut self)
    {
        for x in self.memory.iter_mut() 
        {
            *x = ByteValue::Unknown;
        }
    }

    fn load_indirect(&self, indirect:ByteValue) -> ByteValue
    {
        if let ByteValue::Constant(indirect_val) = indirect
        {
            if let ByteValue::Constant(b0) = self.load_value(OpData::ZeroPage(indirect_val))
            {
                if let ByteValue::Constant(b1) = self.load_value(OpData::ZeroPage(indirect_val + 1))
                {
                    return self.memory[((b1 as u16) << 8 + b0 as u16) as usize];
                }
            }

        }

        return ByteValue::Unknown;
    }

    fn load_value(&self, addr:OpData) -> ByteValue
    {
        match addr
        {
            OpData::Unknown => ByteValue::Unknown,
            OpData::Immediate(val) => ByteValue::Constant(val),
            OpData::ZeroPage(val) => self.memory[val as usize],
            OpData::ZeroPageX(base) => self.load_value(add_zp_address(base, self.x)),
            OpData::ZeroPageY(base) => self.load_value(add_zp_address(base, self.y)),
            OpData::Absolute(val) => self.memory[val as usize],
            OpData::AbsoluteX(base) => self.load_value(add_address(base, self.x)),
            OpData::AbsoluteY(base) => self.load_value(add_address(base, self.y)),
            // OpData::IndirectX(base) => self.load_indirect(self.x),
            // OpData::IndirectY(base) => self.load_indirect(self.y),
        }
    }

    fn store_indirect(&mut self, indirect:ByteValue, val:ByteValue)
    {
        if let ByteValue::Constant(indirect_val) = indirect
        {
            if let ByteValue::Constant(b0) = self.load_value(OpData::ZeroPage(indirect_val))
            {
                if let ByteValue::Constant(b1) = self.load_value(OpData::ZeroPage(indirect_val + 1))
                {
                    self.memory[((b1 as u16) << 8 + b0 as u16) as usize] = val;
                    return;
                }
            }

        }

        self.invalidate_memory();
    }

    fn store_value(&mut self, address:OpData, val:ByteValue)
    {
        match address 
        {
            OpData::Unknown => self.invalidate_memory(),
            OpData::Immediate(addr) => self.memory[addr as usize] = val,
            OpData::ZeroPage(addr) => self.memory[addr as usize] = val,
            OpData::ZeroPageX(base) => { let x = self.x.clone(); self.store_value(add_zp_address(base, x), val); },
            OpData::ZeroPageY(base) => { let y = self.y.clone(); self.store_value(add_zp_address(base, y), val); },
            OpData::Absolute(addr) => self.memory[addr as usize] = val,
            OpData::AbsoluteX(base) => { let x = self.x.clone(); self.store_value(add_address(base, x), val); },
            OpData::AbsoluteY(base) => { let y = self.y.clone(); self.store_value(add_address(base, y), val); },
            // OpData::IndirectX(base) => { let x = self.x.clone(); self.store_indirect(x, val); },
            // OpData::IndirectY(base) => { let y = self.y.clone(); self.store_indirect(y, val); },
        };
    }

    fn evaluate(&mut self, stmt:&Statement)
    {
        match stmt.code {
            OpCode::INX => self.x = add_value(self.x.clone(), 1),
            OpCode::DEX => self.x = add_value(self.x.clone(), 0xff),
            OpCode::INY => self.y = add_value(self.y.clone(), 1),
            OpCode::DEY => self.y = add_value(self.y.clone(), 0xff),
            OpCode::TAX => self.x = add_value(self.a.clone(), 0),
            OpCode::TAY => self.y = add_value(self.a.clone(), 0),
            OpCode::TXA => self.a = add_value(self.x.clone(), 0),
            OpCode::TYA => self.a = add_value(self.y.clone(), 0),
            OpCode::LDA => self.a = self.load_value(stmt.data.clone()),
            OpCode::LDX => self.x = self.load_value(stmt.data.clone()),
            OpCode::LDY => self.y = self.load_value(stmt.data.clone()),
            OpCode::STA => 
            { 
                let data = stmt.data.clone(); 
                let a_val = self.a.clone();

                self.store_value(data, a_val); 
            },
            OpCode::STX => 
            { 
                let data = stmt.data.clone(); 
                let x_val = self.x.clone(); 
                self.store_value(data, x_val);
            },
            OpCode::STY => 
            {
                let data = stmt.data.clone();
                let y_val = self.y.clone();
                self.store_value(data, y_val);
            },
        };
    }

    fn run(&mut self, program:&Program)
    {
        for stmt in program.statements.iter() 
        {
            self.evaluate(&stmt);
        }
    }
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

        let (scopes,collection):(Vec<u32>,Vec<Token>) = lexer.unzip();
        let expected = vec![Token::Symbol('('), Token::Word("a"), Token::Symbol('+'), Token::Word("b"), Token::Symbol('*'), Token::Word("c"), Token::Symbol(')')];

        assert_eq!(expected, collection);

        assert_eq!(vec![0,0,0,0,0,0,0], scopes);
    }

    #[test]
    fn lexer_indent()
    {
        let lexer = Lexer::new("outer:\n    inner:\n        jmp inner\n");

        let collection:Vec<(u32,Token)> = lexer.collect();
        let expected = vec![(0,Token::Word("outer")),(0,Token::Symbol(':')),(1,Token::Word("inner")),(1,Token::Symbol(':')),(2,Token::Word("jmp")),(2,Token::Word("inner"))];

        assert_eq!(expected, collection);
    }

    #[test]
    fn eval_inx()
    {
        let mut program = Program::new();

        program.statements = vec![Statement::new(OpCode::INX, OpData::Unknown)];

        let mut cpu = CPU::new();
        assert_eq!(ByteValue::Constant(0), cpu.x);

        cpu.run(&program);
        assert_eq!(ByteValue::Constant(1), cpu.x);
        assert_eq!(ByteValue::Constant(0), cpu.y);
        assert_eq!(ByteValue::Constant(0), cpu.a);
    }

    #[test]
    fn eval_iny()
    {
        let mut program = Program::new();

        program.statements = vec![Statement::new(OpCode::INY, OpData::Unknown)];

        let mut cpu = CPU::new();
        assert_eq!(ByteValue::Constant(0), cpu.y);

        cpu.run(&program);
        assert_eq!(ByteValue::Constant(0), cpu.x);
        assert_eq!(ByteValue::Constant(1), cpu.y);
        assert_eq!(ByteValue::Constant(0), cpu.a);
    }

    #[test]
    fn eval_lda_immediate()
    {
        let mut program = Program::new();

        program.statements = vec![Statement::new(OpCode::LDA, OpData::Immediate(42))];

        let mut cpu = CPU::new();
        assert_eq!(ByteValue::Constant(0), cpu.a);

        cpu.run(&program);
        assert_eq!(ByteValue::Constant(0), cpu.x);
        assert_eq!(ByteValue::Constant(0), cpu.y);
        assert_eq!(ByteValue::Constant(42), cpu.a);

    }

    #[test]
    fn eval_lda_zeropage()
    {
        let mut program = Program::new();


        program.statements = vec![Statement::new(OpCode::LDA, OpData::ZeroPage(42))];

        let mut cpu = CPU::new();
        cpu.memory[42] = ByteValue::Constant(17);

        assert_eq!(ByteValue::Constant(0), cpu.a);

        cpu.run(&program);
        assert_eq!(ByteValue::Constant(0), cpu.x);
        assert_eq!(ByteValue::Constant(0), cpu.y);
        assert_eq!(ByteValue::Constant(17), cpu.a);

    }
}
