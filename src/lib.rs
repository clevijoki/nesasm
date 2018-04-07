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

#[derive(Debug, PartialEq, Clone)]
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

#[derive(Debug, PartialEq, Clone, Copy)]
struct FlagState
{
    bits:u8
}

impl FlagState
{
    fn on() -> FlagState
    {
        FlagState { bits : 2 }
    }

    fn off() -> FlagState
    {
        FlagState { bits : 1 }
    }

    fn unknown() -> FlagState
    {
        FlagState { bits : 3 }
    }
}


#[derive(Debug, PartialEq, Clone, Copy)]
struct ByteValue
{
    bits:[u64;4],
}

impl ByteValue
{
    fn new(initial:u8) -> ByteValue
    {
        let mut bits:[u64;4] = [0,0,0,0];

        let idx = (initial as usize)>>6;
        let val = 1 << (initial & 0x3f);
        bits[idx] = val;

        ByteValue { bits }
    }

    fn unknown() -> ByteValue
    {
        ByteValue { bits:[0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff] }
    }
}

// represents the state of the CPU when statically analyzing
struct CPU
{
    carry:FlagState,
    zero:FlagState,
    interupt_disable:FlagState,
    decimal_mode:FlagState,
    brk:FlagState,
    negative:FlagState,
    overflow:FlagState,

    a:ByteValue,
    x:ByteValue,
    y:ByteValue,
    stack:ByteValue,

    memory:[ByteValue;64*1024],
}

trait Expression
{
    fn evaluate(cpu:&mut CPU);
}

#[derive(Debug, PartialEq, Clone, Copy)]
enum OpData
{
    Unknown,
    Accumulator,
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
    ADC,
    AND,
    ASL,
    ASR,
    INX,
    INY,
    DEX,
    DEY,
    EOR,
    ORA,
    TAX,
    TXA,
    TAY,
    TYA,
    TSX,
    TXS,
    LDA,
    LDX,
    LDY,
    ROR,
    ROL,
    STA,
    STY,
    STX,
}

struct Statement
{
    code:OpCode,
    data: OpData,
}

impl Statement
{
    fn new(code:OpCode) -> Statement
    {
        Statement { code, data:OpData::Unknown }
    }

    fn new_data(code:OpCode, data:OpData) -> Statement
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
            negative:FlagState::off(), 
            overflow:FlagState::off(), 
            brk:FlagState::off(),
            decimal_mode:FlagState::off(),
            interupt_disable:FlagState::off(),
            zero:FlagState::off(),
            carry:FlagState::off(),
            a:ByteValue::unknown(),
            x:ByteValue::unknown(),
            y:ByteValue::unknown(),
            stack:ByteValue::unknown(),
            memory:[ByteValue::unknown();64*1024] 
        }
    }

    fn add_value(base:ByteValue, val:ByteValue) -> ByteValue
    {
        match base 
        {
            ByteValue::Constant(x) => match val 
            {
                ByteValue::Constant(y) => ByteValue::Constant(((x as u32) + (y as u32) & 0xff) as u8),
                ByteValue::Unknown => ByteValue::Unknown,
            },
            ByteValue::Unknown => ByteValue::Unknown
        }
    }

    fn or_value(base:ByteValue, val:ByteValue) -> ByteValue
    {
        match base 
        {
            ByteValue::Constant(x) => match val 
            {
                ByteValue::Constant(y) => ByteValue::Constant(x | y),
                ByteValue::Unknown => ByteValue::Unknown,
            },
            ByteValue::Unknown => ByteValue::Unknown
        }
    }

    fn xor_value(base:ByteValue, val:ByteValue) -> ByteValue
    {
        match base 
        {
            ByteValue::Constant(x) => match val 
            {
                ByteValue::Constant(y) => ByteValue::Constant(x ^ y),
                ByteValue::Unknown => ByteValue::Unknown,
            },
            ByteValue::Unknown => ByteValue::Unknown
        }
    }

    fn and_value(base:ByteValue, val:ByteValue) -> ByteValue
    {
        match base 
        {
            ByteValue::Constant(x) => match val 
            {
                ByteValue::Constant(y) => ByteValue::Constant(x & y),
                ByteValue::Unknown => ByteValue::Unknown,
            },
            ByteValue::Unknown => ByteValue::Unknown
        }
    }

    fn shl_value(val:ByteValue) -> (ByteValue,FlagState)
    {
        match val
        {
            ByteValue::Constant(x) => ByteValue::Constant(((x << 1) & 0xff as u8, if x & 0x80 {FlagState::On} else {FlagState::Off}))
            ByteValue::Unknown => (ByteValue::Unknown, FlagState::Unknown)
        }
    }

    fn shr_value(val:ByteValue) -> ByteValue
    {
        match val
        {
            ByteValue::Constant(x) => ByteValue::Constant(x >> 1),
            ByteValue::Unknown => ByteValue::Unknown,
        }
    }

    // returns new value, carry flag, overflow flag, 
    fn add_with_carry(base:ByteValue, extra:ByteValue, carry:FlagState) -> (ByteValue,FlagState,FlagState)
    {
        if let ByteValue::Constant(base_val) = base
        {
            if let ByteValue::Constant(extra_val) = extra
            {
                let carry_val = match carry {
                    FlagState::On => 1,
                    FlagState::Off => 0,
                    FlagState::Unknown => return (ByteValue::Unknown,FlagState::Unknown,FlagState::Unknown),
                };

                let result = base_val as u32 + extra_val as u32 + carry_val;

                return (ByteValue::Constant((result & 0xff) as u8), 
                    if (result & 0xff) != result { FlagState::On } else { FlagState::Off },
                    if (base_val & 0x80) == 0 && (extra_val & 0x80) == 0 && (result & 0x80) != 0 { FlagState::On } else { FlagState::Off }
                    );
            }
        }

        return (ByteValue::Unknown, FlagState::Unknown, FlagState::Unknown);
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

    fn get_zero_flag(val:ByteValue) -> FlagState
    {
        match val
        {
            ByteValue::Constant(x) => if x == 0 { FlagState::On } else { FlagState::Off },
            ByteValue::Unknown => FlagState::Unknown
        }
    }

    fn get_negative_flag(val:ByteValue) -> FlagState
    {
        match val
        {
            ByteValue::Constant(x) => if (x & 0x80) != 0 { FlagState::On } else { FlagState::Off },
            ByteValue::Unknown => FlagState::Unknown
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
            OpData::Accumulator => self.a,
            OpData::Immediate(val) => ByteValue::Constant(val),
            OpData::ZeroPage(val) => self.memory[val as usize],
            OpData::ZeroPageX(base) => self.load_value(CPU::add_zp_address(base, self.x)),
            OpData::ZeroPageY(base) => self.load_value(CPU::add_zp_address(base, self.y)),
            OpData::Absolute(val) => self.memory[val as usize],
            OpData::AbsoluteX(base) => self.load_value(CPU::add_address(base, self.x)),
            OpData::AbsoluteY(base) => self.load_value(CPU::add_address(base, self.y)),
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
            OpData::Accumulator => self.a = val,
            OpData::Immediate(addr) => self.memory[addr as usize] = val,
            OpData::ZeroPage(addr) => self.memory[addr as usize] = val,
            OpData::ZeroPageX(base) => { let x = self.x; self.store_value(CPU::add_zp_address(base, x), val); },
            OpData::ZeroPageY(base) => { let y = self.y; self.store_value(CPU::add_zp_address(base, y), val); },
            OpData::Absolute(addr) => self.memory[addr as usize] = val,
            OpData::AbsoluteX(base) => { let x = self.x; self.store_value(CPU::add_address(base, x), val); },
            OpData::AbsoluteY(base) => { let y = self.y; self.store_value(CPU::add_address(base, y), val); },
            // OpData::IndirectX(base) => { let x = self.x; self.store_indirect(x, val); },
            // OpData::IndirectY(base) => { let y = self.y; self.store_indirect(y, val); },
        };
    }

    fn evaluate(&mut self, stmt:&Statement)
    {
        match stmt.code {
            OpCode::ADC =>
            {
                let mem = self.load_value(stmt.data.clone());
                let (a, carry, overflow) = CPU::add_with_carry(self.a, mem, self.carry);
                self.a = a;
                self.carry = carry;
                self.overflow = overflow;
                self.zero = CPU::get_zero_flag(self.a);
                self.negative = CPU::get_negative_flag(self.a);
            },
            OpCode::AND =>
            {
                let mem = self.load_value(stmt.data.clone());
                self.a = CPU::and_value(self.a, mem);
                self.zero = CPU::get_zero_flag(self.a);
                self.negative = CPU::get_negative_flag(self.a);
            },
            OpCode::ASL =>
            {
                let mem = self.load_value(stmt.data);
                self.a = CPU::shl_value(mem);
                self.zero = CPU::get_zero_flag(self.a);
                self.negative = CPU::get_negative_flag(self.a);
            },
            OpCode::ASR =>
            {
                let mem = self.load_value(stmt.data);
                self.a = CPU::shr_value(mem);
                self.zero = CPU::get_zero_flag(self.a);
                self.negative = CPU::get_negative_flag(self.a);
            },
            OpCode::ORA =>
            {
                let mem = self.load_value(stmt.data.clone());
                self.a = CPU::or_value(self.a, mem);
                self.zero = CPU::get_zero_flag(self.a);
                self.negative = CPU::get_negative_flag(self.a);
            },
            OpCode::EOR =>
            {
                let mem = self.load_value(stmt.data.clone());
                self.a = CPU::xor_value(self.a, mem);
                self.zero = CPU::get_zero_flag(self.a);
                self.negative = CPU::get_negative_flag(self.a);
            },
            OpCode::INX => 
            { 
                self.x = CPU::add_value(self.x, ByteValue::Constant(1)); 
                self.zero = CPU::get_zero_flag(self.x);
                self.negative = CPU::get_negative_flag(self.x);
            }, 
            OpCode::DEX => 
            {
                self.x = CPU::add_value(self.x, ByteValue::Constant(0xff));
                self.zero = CPU::get_zero_flag(self.x);
                self.negative = CPU::get_negative_flag(self.x);
            },
            OpCode::INY =>
            {
                self.y = CPU::add_value(self.y, ByteValue::Constant(1));
                self.zero = CPU::get_zero_flag(self.y);
                self.negative = CPU::get_negative_flag(self.y);
            },
            OpCode::DEY =>
            {
                self.y = CPU::add_value(self.y, ByteValue::Constant(0xff));
                self.zero = CPU::get_zero_flag(self.y);
                self.negative = CPU::get_negative_flag(self.y);
            } 
            OpCode::TAX => self.x = self.a,
            OpCode::TAY => self.y = self.a,
            OpCode::TXA => self.a = self.x,
            OpCode::TYA => self.a = self.y,
            OpCode::TSX => self.x = self.stack,
            OpCode::TXS => self.stack = self.x,
            OpCode::LDA => self.a = self.load_value(stmt.data.clone()),
            OpCode::LDX => self.x = self.load_value(stmt.data.clone()),
            OpCode::LDY => self.y = self.load_value(stmt.data.clone()),
            OpCode::STA => 
            { 
                let data = stmt.data.clone(); 
                let a_val = self.a;

                self.store_value(data, a_val); 
            },
            OpCode::STX => 
            { 
                let data = stmt.data.clone(); 
                let x_val = self.x; 
                self.store_value(data, x_val);
            },
            OpCode::STY => 
            {
                let data = stmt.data.clone();
                let y_val = self.y;
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
    fn eval_inc_registers()
    {
        let mut cpu = CPU::new();
        cpu.x = ByteValue::Constant(0);
        cpu.y = ByteValue::Constant(0xff);

        cpu.evaluate(&Statement::new(OpCode::INX));

        assert_eq!(ByteValue::Constant(1), cpu.x);
        assert_eq!(ByteValue::Constant(0xff), cpu.y);
        assert_eq!(FlagState::Off, cpu.zero);

        cpu.evaluate(&Statement::new(OpCode::INY));

        assert_eq!(ByteValue::Constant(1), cpu.x);
        assert_eq!(ByteValue::Constant(0), cpu.y);
        assert_eq!(FlagState::On, cpu.zero);

        assert_eq!(ByteValue::Unknown, cpu.a);
    }

    #[test]
    fn eval_dec_registers()
    {
        let mut cpu = CPU::new();
        cpu.x = ByteValue::Constant(2);
        cpu.y = ByteValue::Constant(1);

        cpu.evaluate(&Statement::new(OpCode::DEX));

        assert_eq!(ByteValue::Unknown, cpu.a);
        assert_eq!(ByteValue::Constant(1), cpu.x);
        assert_eq!(ByteValue::Constant(1), cpu.y);
        assert_eq!(FlagState::Off, cpu.zero);

        cpu.evaluate(&Statement::new(OpCode::DEY));

        assert_eq!(ByteValue::Unknown, cpu.a);
        assert_eq!(ByteValue::Constant(1), cpu.x);
        assert_eq!(ByteValue::Constant(0), cpu.y);
        assert_eq!(FlagState::On, cpu.zero);

        cpu.evaluate(&Statement::new(OpCode::DEX));

        assert_eq!(ByteValue::Unknown, cpu.a);
        assert_eq!(ByteValue::Constant(0), cpu.x);
        assert_eq!(ByteValue::Constant(0), cpu.y);
        assert_eq!(FlagState::On, cpu.zero);
    }

    #[test]
    fn eval_transfer()
    {
        let mut cpu = CPU::new();
        cpu.a = ByteValue::Constant(42);

        assert_eq!(ByteValue::Unknown, cpu.x);
        assert_eq!(ByteValue::Unknown, cpu.y);
        assert_eq!(ByteValue::Unknown, cpu.stack);

        cpu.evaluate(&Statement::new(OpCode::TAX));
        assert_eq!(ByteValue::Constant(42), cpu.a);
        assert_eq!(ByteValue::Constant(42), cpu.x);
        assert_eq!(ByteValue::Unknown, cpu.y);
        assert_eq!(ByteValue::Unknown, cpu.stack);

        cpu.evaluate(&Statement::new(OpCode::TAY));
        assert_eq!(ByteValue::Constant(42), cpu.a);
        assert_eq!(ByteValue::Constant(42), cpu.x);
        assert_eq!(ByteValue::Constant(42), cpu.y);
        assert_eq!(ByteValue::Unknown, cpu.stack);

        cpu.evaluate(&Statement::new(OpCode::TXS));
        assert_eq!(ByteValue::Constant(42), cpu.a);
        assert_eq!(ByteValue::Constant(42), cpu.x);
        assert_eq!(ByteValue::Constant(42), cpu.y);
        assert_eq!(ByteValue::Constant(42), cpu.stack);

        cpu.stack = ByteValue::Constant(17);

        cpu.evaluate(&Statement::new(OpCode::TSX));
        assert_eq!(ByteValue::Constant(42), cpu.a);
        assert_eq!(ByteValue::Constant(17), cpu.x);
        assert_eq!(ByteValue::Constant(42), cpu.y);
        assert_eq!(ByteValue::Constant(17), cpu.stack);

        cpu.evaluate(&Statement::new(OpCode::TXA));
        assert_eq!(ByteValue::Constant(17), cpu.a);
        assert_eq!(ByteValue::Constant(17), cpu.x);
        assert_eq!(ByteValue::Constant(42), cpu.y);
        assert_eq!(ByteValue::Constant(17), cpu.stack);

        cpu.evaluate(&Statement::new(OpCode::TYA));
        assert_eq!(ByteValue::Constant(42), cpu.a);
        assert_eq!(ByteValue::Constant(17), cpu.x);
        assert_eq!(ByteValue::Constant(42), cpu.y);
        assert_eq!(ByteValue::Constant(17), cpu.stack);
    }

    #[test]
    fn eval_and()
    {
        let mut cpu = CPU::new();

        cpu.a = ByteValue::Constant(0xAA);

        cpu.evaluate(&Statement::new_data(OpCode::AND, OpData::Immediate(0xf)));

        assert_eq!(ByteValue::Constant(0x0f & 0xAA), cpu.a);

        cpu.evaluate(&Statement::new_data(OpCode::AND, OpData::Immediate(0xf0)));

        assert_eq!(ByteValue::Constant(0), cpu.a);
        assert_eq!(FlagState::On, cpu.zero);
    }

    #[test]
    fn eval_or()
    {
        let mut cpu = CPU::new();

        cpu.a = ByteValue::Constant(0xAA);

        cpu.evaluate(&Statement::new_data(OpCode::ORA, OpData::Immediate(0xf)));

        assert_eq!(ByteValue::Constant(0x0f | 0xAA), cpu.a);

        cpu.evaluate(&Statement::new_data(OpCode::ORA, OpData::Immediate(0xf0)));

        assert_eq!(ByteValue::Constant(0xff), cpu.a);
        assert_eq!(FlagState::Off, cpu.zero);

        cpu.a = ByteValue::Constant(0);
        cpu.evaluate(&Statement::new_data(OpCode::ORA, OpData::Immediate(0)));

        assert_eq!(ByteValue::Constant(0), cpu.a);
        assert_eq!(FlagState::On, cpu.zero);
    }

    #[test]
    fn eval_xor()
    {
        let mut cpu = CPU::new();

        cpu.a = ByteValue::Constant(0xAA);

        cpu.evaluate(&Statement::new_data(OpCode::EOR, OpData::Immediate(0xf)));

        assert_eq!(ByteValue::Constant(0x0f ^ 0xAA), cpu.a);

        cpu.evaluate(&Statement::new_data(OpCode::EOR, OpData::Immediate(0xf0)));

        assert_eq!(ByteValue::Constant(0x55), cpu.a);
        assert_eq!(FlagState::Off, cpu.zero);

        cpu.evaluate(&Statement::new_data(OpCode::EOR, OpData::Immediate(0x55)));

        assert_eq!(ByteValue::Constant(0), cpu.a);
        assert_eq!(FlagState::On, cpu.zero);
    }

    #[test]
    fn eval_shl()
    {
        let mut cpu = CPU::new();

        cpu.a = ByteValue::Constant(0xAA);
        cpu.memory[42] = ByteValue::Constant(42);
        cpu.memory[0x1000] = ByteValue::Constant(13);

        cpu.evaluate(&Statement::new_data(OpCode::ROL, OpData::Accumulator));
        assert_eq!(ByteValue::Constant(0xAA << 1), cpu.a);
        
        cpu.evaluate(&Statement::new_data(OpCode::ROL, OpData::ZeroPage(42)));
        assert_eq!(ByteValue::Constant(0x42 << 1), cpu.memory[42]);
        
        cpu.evaluate(&Statement::new_data(OpCode::ROL, OpData::Absolute(0x1000)));
        assert_eq!(ByteValue::Constant(13 << 1), cpu.memory[0x1000]);
    }

    #[test]
    fn eval_shr()
    {
        let mut cpu = CPU::new();

        cpu.a = ByteValue::Constant(0xAA);
        cpu.memory[42] = ByteValue::Constant(42);
        cpu.memory[0x1000] = ByteValue::Constant(13);

        cpu.evaluate(&Statement::new_data(OpCode::ROR, OpData::Accumulator));
        assert_eq!(ByteValue::Constant(0xAA >> 1), cpu.a);
        
        cpu.evaluate(&Statement::new_data(OpCode::ROR, OpData::ZeroPage(42)));
        assert_eq!(ByteValue::Constant(0x42 >> 1), cpu.memory[42]);
        
        cpu.evaluate(&Statement::new_data(OpCode::ROR, OpData::Absolute(0x1000)));
        assert_eq!(ByteValue::Constant(13 >> 1), cpu.memory[0x1000]);
    }


    #[test]
    fn eval_load_immediate()
    {
        let mut program = Program::new();

        program.statements = vec![
            Statement::new_data(OpCode::LDA, OpData::Immediate(42)),
            Statement::new_data(OpCode::LDX, OpData::Immediate(43)),
            Statement::new_data(OpCode::LDY, OpData::Immediate(44))];

        let mut cpu = CPU::new();

        cpu.run(&program);
        assert_eq!(ByteValue::Constant(42), cpu.a);
        assert_eq!(ByteValue::Constant(43), cpu.x);
        assert_eq!(ByteValue::Constant(44), cpu.y);

    }

    #[test]
    fn eval_lda_zeropage()
    {
        let mut program = Program::new();


        program.statements = vec![Statement::new_data(OpCode::LDA, OpData::ZeroPage(42))];

        let mut cpu = CPU::new();
        cpu.memory[42] = ByteValue::Constant(17);
        cpu.a = ByteValue::Constant(0);

        cpu.run(&program);
        assert_eq!(ByteValue::Unknown, cpu.x);
        assert_eq!(ByteValue::Unknown, cpu.y);
        assert_eq!(ByteValue::Constant(17), cpu.a);
    }
}
