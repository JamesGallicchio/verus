//! The Verus-Lean AST Abstract Syntax Tree
//!
//! Rust-AST --> Rust-HIR --> VIR-AST --> VLIR-AST --> Lean
//!
//! Right now this is just used to control how Serde serializes it

use rustc_middle::ty::layout::MaybeResult;
use serde::{Deserialize, Serialize};
use std::{convert::{TryFrom, TryInto}, sync::Arc};
use air::ast::Quant;
use vir::ast::{
    ArithOp, Binder, Binders, BitwiseOp, Ident, Idents, InequalityOp, IntRange, MultiOp, Path
};

fn binder<T: Clone + TryInto<U>, U: Clone>(x: Binder<T>) -> Result<Binder<U>,T::Error> {
    Ok(Arc::new(
        air::ast::BinderX { name: x.name.clone(), a: x.a.clone().try_into()? }
    ))
}

fn binders<T: Clone + TryInto<U>, U: Clone>(x: Binders<T>) -> Result<Binders<U>, T::Error> {
    let res: Vec<Binder<U>> = x.iter().map(|b| binder(b.clone()))
        .collect::<Result<Vec<_>,_>>()?;
    Ok(Arc::new(res))
}

fn varident(x: &vir::ast::VarIdent) -> Ident {
    x.0.clone()
}

fn varbinder<T: Clone + TryInto<U>, U: Clone>(x: &vir::ast::VarBinder<T>) -> Result<Binder<U>,T::Error> {
    Ok(Arc::new(air::ast::BinderX {
        name: varident(&x.name),
        a: x.a.clone().try_into()?,
    }))
}

fn varbinders<T: Clone + TryInto<U>, U: Clone>(x: &vir::ast::VarBinders<T>) -> Result<Binders<U>,T::Error> {
    let res: Vec<Binder<U>> = x.iter().map(|b| varbinder(b))
        .collect::<Result<Vec<_>,_>>()?;
    Ok(Arc::new(res))
}

#[derive(Debug, Serialize, Deserialize, Hash, Clone)]
pub struct Typ(Arc<TypX>);
#[derive(Debug, Serialize, Deserialize, Hash)]
pub struct Typs(Arc<Vec<Typ>>);
// Because of ImplPaths in TypX::Datatype, TypX should not implement PartialEq, Eq
// See ast_util::types_equal instead
#[derive(Debug, Serialize, Deserialize, Hash)]
pub enum TypX {
    /// Bool, Int, Datatype are translated directly into corresponding SMT types (they are not SMT-boxed)
    Bool,
    Int(IntRange),
    /// UTF-8 character type
    Char,
    /// `FnSpec` type (TODO rename from 'Lambda' to just 'FnSpec')
    /// (t1, ..., tn) -> t0.
    Lambda(Typs, Typ),
    /// Datatype (concrete or abstract) applied to type arguments
    Datatype(Path, Typs),
    /// Array (static length)
    Array(Typ, Typ),
    /// Array (dynamic)
    Slice(Typ),
    /// Type parameter (inherently SMT-boxed, and cannot be unboxed)
    TypParam(Ident),
    /// Const integer type argument (e.g. for array sizes)
    ConstInt { _0: num_bigint::BigInt },
}

impl TryFrom<vir::ast::Typs> for Typs {
    type Error = String;
    fn try_from(v: vir::ast::Typs) -> Result<Self,Self::Error> {
        let res: Result<Vec<_>,_> = v.as_ref().clone().into_iter().map(|x| x.try_into()).collect();
        Ok(Typs(Arc::new(res?)))
    }
}

impl TryFrom<vir::ast::Typ> for Typ {
    type Error = String;
    fn try_from(v: vir::ast::Typ) -> Result<Self,Self::Error> {
        let res: Result<TypX,String> = v.as_ref().try_into();
        Ok(Typ(Arc::new(res?)))
    }
}

impl TryFrom<&vir::ast::TypX> for TypX {
    type Error = String;
    fn try_from(v: &vir::ast::TypX) -> Result<Self,Self::Error> {
        Ok(match v {
            vir::ast::TypX::Bool => TypX::Bool,
            vir::ast::TypX::Int(r) => TypX::Int(*r),
            vir::ast::TypX::Char => TypX::Char,
            vir::ast::TypX::Lambda(t1, t2) => {
                TypX::Lambda(t1.clone().try_into()?, t2.clone().try_into()?)
            }
            vir::ast::TypX::Datatype(a,b,_) => {
                TypX::Datatype(a.clone(),b.clone().try_into()?)
            }
            vir::ast::TypX::Primitive(vir::ast::Primitive::Array, ts) => {
                let ts_: Typs = ts.clone().try_into()?;
                match ts_.0.as_slice() {
                    [t0,t1] => TypX::Array(t0.clone(),t1.clone()),
                    _ => {
                        return Err(format!("Array arguments wrong length: {:?}", ts_))
                    }
                }
            }
            vir::ast::TypX::Primitive(vir::ast::Primitive::Slice, ts) => {
                let ts_: Typs = ts.clone().try_into()?;
                match ts_.0.as_slice() {
                    [t] => TypX::Slice(t.clone()),
                    _ => {
                        return Err(format!("Slice arguments wrong length: {:?}", ts_))
                    }
                }
            }
            _ => return Err(format!("Unsupported type {:?}", v)),
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum Constant {
    /// true or false
    Bool(bool),
    /// integer of arbitrary size
    Int(
        #[serde(serialize_with = "bigint_serial")]
        num_bigint::BigInt
    ),
    /// Hold generated string slices in here
    StrSlice(Arc<String>),
    // Hold unicode values here
    // Char(char),
}

fn bigint_serial<S: serde::Serializer>(a: &num_bigint::BigInt, b: S) -> Result<S::Ok,S::Error> {
    b.serialize_str(&a.to_string())
}

impl TryFrom<&vir::ast::Constant> for Constant {
    type Error = String;
    fn try_from(v: &vir::ast::Constant) -> Result<Self, Self::Error> {
        Ok(match v {
            vir::ast::Constant::Bool(b) => Constant::Bool(*b),
            vir::ast::Constant::Int(i) => Constant::Int(i.clone()),
            vir::ast::Constant::StrSlice(s) => Constant::StrSlice(s.clone()),
            vir::ast::Constant::Char(_) => return Err(format!("Unsupported constant: {:?}", v)),
        })
    }
}

#[derive(Clone,Debug,Serialize,Deserialize)]
pub enum UnaryOp {
    Not,
    BitNot,
    Clip {
        range: IntRange,
        truncate: bool,
    },
    Id,
}

impl TryFrom<&vir::ast::UnaryOp> for UnaryOp {
    type Error = String;
    fn try_from(v: &vir::ast::UnaryOp) -> Result<Self,Self::Error> {
        Ok(match v {
            vir::ast::UnaryOp::Not => UnaryOp::Not,
            vir::ast::UnaryOp::BitNot => UnaryOp::BitNot,
            vir::ast::UnaryOp::Trigger(_) => UnaryOp::Id,
            vir::ast::UnaryOp::Clip { range, truncate } =>
                UnaryOp::Clip { range: range.clone(), truncate: *truncate },
            vir::ast::UnaryOp::CoerceMode { op_mode:_, from_mode:_, to_mode:_, kind:_ } =>
                UnaryOp::Id,
            vir::ast::UnaryOp::MustBeFinalized |
            vir::ast::UnaryOp::StrLen |
            vir::ast::UnaryOp::HeightTrigger |
            vir::ast::UnaryOp::StrIsAscii |
            vir::ast::UnaryOp::CharToInt |
            vir::ast::UnaryOp::InferSpecForLoopIter { print_hint:_ } |
            vir::ast::UnaryOp::CastToInteger =>
                return Err(format!("Unsupported UnaryOp: {:?}", v)),
        })
    }
}

impl TryFrom<&vir::ast::UnaryOpr> for UnaryOp {
    type Error = String;
    fn try_from(v: &vir::ast::UnaryOpr) -> Result<Self,Self::Error> {
        Ok(match v {
            _ => return Err(format!("Unsupported UnaryOpr: {:?}", v)),
        })
    }
}

#[derive(Clone,Debug,Serialize,Deserialize)]
pub enum BinaryOp {
    And,
    Or,
    Xor,
    Implies,
    Eq,
    Ne,
    Inequality(InequalityOp),
    Arith(ArithOp),
    Bitwise(BitwiseOp),
    StrGetChar,
}

impl TryFrom<&vir::ast::BinaryOp> for BinaryOp {
    type Error = String;
    fn try_from(v: &vir::ast::BinaryOp) -> Result<Self,Self::Error> {
        Ok(match v {
            vir::ast::BinaryOp::And => BinaryOp::And,
            vir::ast::BinaryOp::Or => BinaryOp::Or,
            vir::ast::BinaryOp::Xor => BinaryOp::Xor,
            vir::ast::BinaryOp::Implies => BinaryOp::Implies,
            vir::ast::BinaryOp::Eq(_) => BinaryOp::Eq,
            vir::ast::BinaryOp::Ne => BinaryOp::Ne,
            vir::ast::BinaryOp::Inequality(x) => BinaryOp::Inequality(x.clone()),
            vir::ast::BinaryOp::Arith(x, _) => BinaryOp::Arith(x.clone()),
            vir::ast::BinaryOp::Bitwise(x, _) => BinaryOp::Bitwise(x.clone()),
            vir::ast::BinaryOp::StrGetChar => BinaryOp::StrGetChar,
            vir::ast::BinaryOp::HeightCompare { strictly_lt:_, recursive_function_field:_ } =>
                return Err(format!("Unsupported BinaryOp: {:?}", v)),
        })
    }
}

impl TryFrom<&vir::ast::BinaryOpr> for BinaryOp {
    type Error = String;
    fn try_from(v: &vir::ast::BinaryOpr) -> Result<Self,Self::Error> {
        Ok(match v {
            _ => return Err(format!("Unsupported BinaryOpr: {:?}", v)),
        })
    }
}

fn pattern(p: vir::ast::Pattern) -> Result<Ident, <Expr as TryFrom<vir::ast::Expr>>::Error> {
    Ok(match &p.x {
        vir::ast::PatternX::Var { name, mutable } => {
            if *mutable {
                return Err(format!("Unsupported pattern: mutable var: {:?}", p.x))
            } else {
                varident(&name)
            }
        }
        vir::ast::PatternX::Wildcard(_)
        | vir::ast::PatternX::Binding { name:_, mutable:_, sub_pat:_ }
        | vir::ast::PatternX::Tuple(_)
        | vir::ast::PatternX::Constructor(_, _, _)
        | vir::ast::PatternX::Or(_, _)
        | vir::ast::PatternX::Expr(_)
        | vir::ast::PatternX::Range(_, _) => {
            return Err(format!("Unsupported pattern: {:?}", p.x))
        }
    })
}

#[derive(Clone,Debug,Serialize,Deserialize)]
pub struct Expr(Arc<ExprX>);
#[derive(Debug,Serialize,Deserialize)]
pub struct Exprs(Arc<Vec<Expr>>);

#[derive( Debug, Serialize, Deserialize)]
pub enum ExprX {
    /// Constant
    Const(Constant),
    /// Local variable as a right-hand side
    Var(Ident),
    /// Call to a function passing some expression arguments
    App(Expr, Exprs),
    /// Construct datatype value of type Path and variant Ident,
    /// with field initializers Binders<Expr> and an optional ".." update expression.
    /// For tuple-style variants, the fields are named "_0", "_1", etc.
    /// Fields can appear **in any order** even for tuple variants.
    Ctor(Path, Ident, Binders<Expr>, Option<Expr>),
    /// Primitive unary operation
    Unary(UnaryOp, Expr),
    /// Primitive binary operation
    Binary(BinaryOp, Expr, Expr),
    /// Primitive multi-operand operation
    Multi(MultiOp, Exprs),
    /// If-else
    If(Expr, Expr, Expr),
    /// Let binding
    Let(Binder<Expr>, Expr),
    /// Quantifier (forall/exists), binding the variables in Binders, with body Expr
    Quant(Quant, Binders<Typ>, Expr),
    /// Array literal (can also be used for sequence literals in the future)
    ArrayLiteral(Exprs),
    /// Executable function (declared with 'fn' and referred to by name)
    StaticFun(Path),
    /// Choose specification values satisfying a condition, compute body
    Choose { params: Binders<Typ>, cond: Expr, body: Expr },
}


impl TryFrom<vir::ast::Expr> for Expr {
    type Error = String;
    fn try_from(v: Arc<vir::ast::SpannedTyped<vir::ast::ExprX>>) -> Result<Self,Self::Error> {
        Ok(Expr(Arc::new((&v.x).try_into()?)))
    }
}

impl TryFrom<vir::ast::Exprs> for Exprs {
    type Error = String;
    fn try_from(v: vir::ast::Exprs) -> Result<Self, Self::Error> {
        let res: Vec<Expr> = v.iter()
            .map(|e| e.clone().try_into())
            .collect::<Result<Vec<_>,_>>()?;
        Ok(Exprs(Arc::new(res)))
    }
}

impl TryFrom<&vir::ast::ExprX> for ExprX {
    type Error = String;
    fn try_from(v: &vir::ast::ExprX) -> Result<Self, Self::Error> {
        Ok(match v {
        vir::ast::ExprX::Const(c) => ExprX::Const(c.try_into()?),
        vir::ast::ExprX::Var(v) => ExprX::Var(varident(v)),
        vir::ast::ExprX::Call(a,b) => {
            match a {
                vir::ast::CallTarget::Fun(_, f, _, _, _) => {
                    ExprX::App(
                        Expr(Arc::new(ExprX::StaticFun(f.path.clone()))),
                        b.clone().try_into()?
                    )
                }
                vir::ast::CallTarget::FnSpec(f) => {
                    ExprX::App(
                        f.clone().try_into()?,
                        b.clone().try_into()?
                    )
                }
                vir::ast::CallTarget::BuiltinSpecFun(_, _, _) => {
                    return Err(format!("Call to a builtin spec function: {:?}", a))
                }
            }
        }
        vir::ast::ExprX::Ctor(a,b,c,d) => {
            ExprX::Ctor(a.clone().into(),b.clone(),
                binders::<vir::ast::Expr, Expr>(c.clone())?,
                d.as_ref().map(|d| d.clone().try_into()).transpose()?
            )
        }
        vir::ast::ExprX::Unary(a,b) =>
            ExprX::Unary(a.try_into()?,b.clone().try_into()?),
        vir::ast::ExprX::UnaryOpr(a,b) =>
            ExprX::Unary(a.try_into()?,b.clone().try_into()?),
        vir::ast::ExprX::Binary(a,b,c) =>
            ExprX::Binary(a.try_into()?,b.clone().try_into()?,c.clone().try_into()?),
        vir::ast::ExprX::BinaryOpr(a,b,c) =>
            ExprX::Binary(a.try_into()?,b.clone().try_into()?,c.clone().try_into()?),
        vir::ast::ExprX::Multi(a,b) =>
            ExprX::Multi(a.clone(),b.clone().try_into()?),
        vir::ast::ExprX::Quant(a,b,c) =>
            ExprX::Quant(a.quant,varbinders(b)?,c.clone().try_into()?),
        vir::ast::ExprX::ArrayLiteral(a) =>
            ExprX::ArrayLiteral(a.clone().try_into()?),
        vir::ast::ExprX::Choose { params, cond, body } =>
            ExprX::Choose {
                params: varbinders(params)?,
                cond: cond.clone().try_into()?,
                body: body.clone().try_into()? },
        vir::ast::ExprX::If(a, b, Some(c)) =>
            ExprX::If(
                a.clone().try_into()?,
                b.clone().try_into()?,
                c.clone().try_into()?),
        vir::ast::ExprX::Block(ss,e) => {
            let res: &vir::ast::Expr = e.as_ref().ok_or(format!("Unsupported expr: block with no return expression: {:?}", v))?;
            let last: ExprX = (&res.x).try_into()?;
            let acc: Result<ExprX,String> = ss.iter().try_rfold(last, |x,y| {
                match &y.x {
                    vir::ast::StmtX::Decl { pattern: p, mode:_, init } => {
                        let name = pattern(p.clone())?;
                        let init = init.as_ref().ok_or(format!("Unsupported declaration: missing init: {:?}", v))?;
                        let s: Binder<Expr> = Arc::new(air::ast::BinderX {
                            name,
                            a: init.clone().try_into()?
                        });
                        Ok(ExprX::Let(s, Expr(Arc::new(x))))        
                    }
                    vir::ast::StmtX::Expr(_) => todo!(),
                }
            });
            acc?
        }
        vir::ast::ExprX::WithTriggers { triggers:_, body } => {
            (&body.x).try_into()?
        }
        _ => {
            return Err(format!("Unsupported expression: {:?}", v))
        }
        })
    }
}

fn param(p: vir::ast::Param) -> Result<Binder<Typ>,<Typ as TryFrom<vir::ast::Typ>>::Error> {
    Ok(Arc::new(air::ast::BinderX{
        name: varident(&p.x.name.clone()),
        a: p.x.typ.clone().try_into()?,
    }))
}

fn params(p: vir::ast::Params) -> Result<Binders<Typ>,<Typ as TryFrom<vir::ast::Typ>>::Error> {
    Ok(Arc::new(
        if params_are_noparam(p.clone()) {
            Vec::new()
        } else {
            p.as_ref().clone().into_iter().map(|p| param(p))
                .collect::<Result<Vec<_>,_>>()?
        }
    ))
}

fn params_are_noparam(p: vir::ast::Params) -> bool {
    match p.as_slice() {
        [b] if *b.x.name.0 == "no%param" => true,
        _ => false
    }
}

#[derive(Serialize,Deserialize)]
pub struct Defn {
    /// Name of function
    pub name: Path,
    /// Type parameters to generic functions
    /// (for trait methods, the trait parameters come first, then the method parameters)
    pub typ_params: Idents,
    /// Function parameters
    pub params: Binders<Typ>,
    /// Return value (unit return type is treated specially; see FunctionX::has_return in ast_util)
    pub ret: Binder<Typ>,
    /// Body of the function
    pub body: Expr,
    /// Decreases clause to ensure recursive function termination
    /// decrease.len() == 0 means no decreases clause
    /// decrease.len() >= 1 means list of expressions, interpreted in lexicographic order
    pub decrease: Exprs,
    /// If Expr is true for the arguments to the function,
    /// the function is defined according to the function body and the decreases clauses must hold.
    /// If Expr is false, the function is uninterpreted, the body and decreases clauses are ignored.
    pub decrease_when: Option<Expr>,
}

#[derive(Serialize,Deserialize)]
pub struct Theorem {
    /// Name of function
    pub name: Path,
    /// Type parameters to generic functions
    /// (for trait methods, the trait parameters come first, then the method parameters)
    pub typ_params: Idents,
    /// Function parameters
    pub params: Binders<Typ>,
    /// Preconditions (requires for proof/exec functions, recommends for spec functions)
    pub require: Exprs,
    /// Postconditions (proof/exec functions only)
    pub ensure: Exprs,
}

#[derive(Serialize,Deserialize)]
pub enum Decl {
    Defn(Defn),
    Theorem(Theorem),
}

impl TryFrom<vir::ast::Function> for Decl {
    type Error = String;
    fn try_from(v: vir::ast::Function) -> Result<Self,Self::Error> {
        let ps = params(v.x.params.clone())?;
        match v.x.mode {
            vir::ast::Mode::Spec => {
                Ok(Decl::Defn(Defn {
                    name: v.x.name.path.clone(),
                    typ_params: v.x.typ_params.clone(),
                    params: ps,
                    ret: param(v.x.ret.clone())?,
                    body: v.x.body.clone().ok_or(format!("Spec function with no body"))?
                            .try_into()?,
                    decrease: v.x.decrease.clone().try_into()?,
                    decrease_when: v.x.decrease_when.clone().map(|e| e.try_into()).transpose()?,
                }))
            }
            vir::ast::Mode::Proof => {
                Ok(Decl::Theorem(Theorem {
                    name: v.x.name.path.clone(),
                    typ_params: v.x.typ_params.clone(),
                    params: ps,
                    require: v.x.require.clone().try_into()?,
                    ensure: v.x.ensure.clone().try_into()?,
                }))
            }
            vir::ast::Mode::Exec => {
                Err(format!("Exec mode functions are not supported: {:?}", v.x.name))
            }
        }
    }
}
