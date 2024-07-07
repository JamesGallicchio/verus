//! The Verus-Lean AST Abstract Syntax Tree
//!
//! Rust-AST --> Rust-HIR --> VIR-AST --> VLIR-AST --> Lean
//!
//! Right now this is just used to control how Serde serializes it

use serde::{Deserialize, Serialize};
use std::{convert::{TryFrom, TryInto}, sync::Arc};
use air::ast::Quant;
use crate::ast::{
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

fn varident(x: &crate::ast::VarIdent) -> Ident {
    x.0.clone()
}

fn varbinder<T: Clone + TryInto<U>, U: Clone>(x: &crate::ast::VarBinder<T>) -> Result<Binder<U>,T::Error> {
    Ok(Arc::new(air::ast::BinderX {
        name: varident(&x.name),
        a: x.a.clone().try_into()?,
    }))
}

fn varbinders<T: Clone + TryInto<U>, U: Clone>(x: &crate::ast::VarBinders<T>) -> Result<Binders<U>,T::Error> {
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

impl TryFrom<crate::ast::Typs> for Typs {
    type Error = String;
    fn try_from(v: crate::ast::Typs) -> Result<Self,Self::Error> {
        let res: Result<Vec<_>,_> = v.as_ref().clone().into_iter().map(|x| x.try_into()).collect();
        Ok(Typs(Arc::new(res?)))
    }
}

impl TryFrom<crate::ast::Typ> for Typ {
    type Error = String;
    fn try_from(v: crate::ast::Typ) -> Result<Self,Self::Error> {
        let res: Result<TypX,String> = v.as_ref().try_into();
        Ok(Typ(Arc::new(res?)))
    }
}

impl TryFrom<&crate::ast::TypX> for TypX {
    type Error = String;
    fn try_from(v: &crate::ast::TypX) -> Result<Self,Self::Error> {
        Ok(match v {
            crate::ast::TypX::Bool => TypX::Bool,
            crate::ast::TypX::Int(r) => TypX::Int(*r),
            crate::ast::TypX::SpecFn(t1, t2) => {
                TypX::Lambda(t1.clone().try_into()?, t2.clone().try_into()?)
            }
            crate::ast::TypX::Datatype(a,b,_) => {
                TypX::Datatype(a.clone(),b.clone().try_into()?)
            }
            crate::ast::TypX::Primitive(crate::ast::Primitive::Array, ts) => {
                let ts_: Typs = ts.clone().try_into()?;
                match ts_.0.as_slice() {
                    [t0,t1] => TypX::Array(t0.clone(),t1.clone()),
                    _ => {
                        return Err(format!("Array arguments wrong length: {:?}", ts_))
                    }
                }
            }
            crate::ast::TypX::Primitive(crate::ast::Primitive::Slice, ts) => {
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

impl TryFrom<&crate::ast::Constant> for Constant {
    type Error = String;
    fn try_from(v: &crate::ast::Constant) -> Result<Self, Self::Error> {
        Ok(match v {
            crate::ast::Constant::Bool(b) => Constant::Bool(*b),
            crate::ast::Constant::Int(i) => Constant::Int(i.clone()),
            crate::ast::Constant::StrSlice(s) => Constant::StrSlice(s.clone()),
            crate::ast::Constant::Char(_) => return Err(format!("Unsupported constant: {:?}", v)),
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

impl TryFrom<&crate::ast::UnaryOp> for UnaryOp {
    type Error = String;
    fn try_from(v: &crate::ast::UnaryOp) -> Result<Self,Self::Error> {
        Ok(match v {
            crate::ast::UnaryOp::Not => UnaryOp::Not,
            crate::ast::UnaryOp::BitNot => UnaryOp::BitNot,
            crate::ast::UnaryOp::Trigger(_) => UnaryOp::Id,
            crate::ast::UnaryOp::Clip { range, truncate } =>
                UnaryOp::Clip { range: range.clone(), truncate: *truncate },
            crate::ast::UnaryOp::CoerceMode { op_mode:_, from_mode:_, to_mode:_, kind:_ } =>
                UnaryOp::Id,
            crate::ast::UnaryOp::MustBeFinalized |
            crate::ast::UnaryOp::StrLen |
            crate::ast::UnaryOp::HeightTrigger |
            crate::ast::UnaryOp::StrIsAscii |
            crate::ast::UnaryOp::InferSpecForLoopIter { print_hint:_ } |
            crate::ast::UnaryOp::CastToInteger =>
                return Err(format!("Unsupported UnaryOp: {:?}", v)),
        })
    }
}

impl TryFrom<&crate::ast::UnaryOpr> for UnaryOp {
    type Error = String;
    fn try_from(v: &crate::ast::UnaryOpr) -> Result<Self,Self::Error> {
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

impl TryFrom<&crate::ast::BinaryOp> for BinaryOp {
    type Error = String;
    fn try_from(v: &crate::ast::BinaryOp) -> Result<Self,Self::Error> {
        Ok(match v {
            crate::ast::BinaryOp::And => BinaryOp::And,
            crate::ast::BinaryOp::Or => BinaryOp::Or,
            crate::ast::BinaryOp::Xor => BinaryOp::Xor,
            crate::ast::BinaryOp::Implies => BinaryOp::Implies,
            crate::ast::BinaryOp::Eq(_) => BinaryOp::Eq,
            crate::ast::BinaryOp::Ne => BinaryOp::Ne,
            crate::ast::BinaryOp::Inequality(x) => BinaryOp::Inequality(x.clone()),
            crate::ast::BinaryOp::Arith(x, _) => BinaryOp::Arith(x.clone()),
            crate::ast::BinaryOp::Bitwise(x, _) => BinaryOp::Bitwise(x.clone()),
            crate::ast::BinaryOp::StrGetChar => BinaryOp::StrGetChar,
            crate::ast::BinaryOp::HeightCompare { strictly_lt:_, recursive_function_field:_ } =>
                return Err(format!("Unsupported BinaryOp: {:?}", v)),
        })
    }
}

impl TryFrom<&crate::ast::BinaryOpr> for BinaryOp {
    type Error = String;
    fn try_from(v: &crate::ast::BinaryOpr) -> Result<Self,Self::Error> {
        Ok(match v {
            _ => return Err(format!("Unsupported BinaryOpr: {:?}", v)),
        })
    }
}

fn pattern(p: crate::ast::Pattern) -> Result<Ident, <Expr as TryFrom<crate::sst::Exp>>::Error> {
    Ok(match &p.x {
        crate::ast::PatternX::Var { name, mutable } => {
            if *mutable {
                return Err(format!("Unsupported pattern: mutable var: {:?}", p.x))
            } else {
                varident(&name)
            }
        }
        crate::ast::PatternX::Wildcard(_)
        | crate::ast::PatternX::Binding { name:_, mutable:_, sub_pat:_ }
        | crate::ast::PatternX::Tuple(_)
        | crate::ast::PatternX::Constructor(_, _, _)
        | crate::ast::PatternX::Or(_, _)
        | crate::ast::PatternX::Expr(_)
        | crate::ast::PatternX::Range(_, _) => {
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
    Ctor(Path, Ident, Binders<Expr>),
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


impl TryFrom<crate::sst::Exp> for Expr {
    type Error = String;
    fn try_from(v: Arc<crate::ast::SpannedTyped<crate::sst::ExpX>>) -> Result<Self,Self::Error> {
        Ok(Expr(Arc::new((&v.x).try_into()?)))
    }
}

impl TryFrom<crate::sst::Exps> for Exprs {
    type Error = String;
    fn try_from(v: crate::sst::Exps) -> Result<Self, Self::Error> {
        let res: Vec<Expr> = v.iter()
            .map(|e| e.clone().try_into())
            .collect::<Result<Vec<_>,_>>()?;
        Ok(Exprs(Arc::new(res)))
    }
}

impl TryFrom<&crate::sst::ExpX> for ExprX {
    type Error = String;
    fn try_from(v: &crate::sst::ExpX) -> Result<Self, Self::Error> {
        Ok(match v {
            crate::sst::ExpX::Const(c) => ExprX::Const(c.try_into()?),
            crate::sst::ExpX::Var(v) => ExprX::Var(varident(v)),
            crate::sst::ExpX::Call(f,tys,exps) => {
                match f {
                    crate::sst::CallFun::Fun(f, _) => {
                        ExprX::App(
                            Expr(Arc::new(ExprX::StaticFun(f.path.clone()))),
                            exps.clone().try_into()?
                        )
                    }
                    crate::sst::CallFun::Recursive(f) => {
                        ExprX::App(
                            Expr(Arc::new(ExprX::StaticFun(f.path.clone()))),
                            exps.clone().try_into()?
                        )
                    }
                    crate::sst::CallFun::InternalFun(_) => {
                        return Err(format!("Call to an internal function: {:?}", f))
                    }
                }
            }
            crate::sst::ExpX::CallLambda(_, f, args) =>
                ExprX::App(f.clone().try_into()?, args.clone().try_into()?),
            crate::sst::ExpX::Ctor(a,b,c) => {
                ExprX::Ctor(a.clone().into(),b.clone(),
                    binders::<crate::sst::Exp, Expr>(c.clone())?
                )
            }
            crate::sst::ExpX::Unary(a,b) =>
                ExprX::Unary(a.try_into()?,b.clone().try_into()?),
            crate::sst::ExpX::UnaryOpr(a,b) =>
                ExprX::Unary(a.try_into()?,b.clone().try_into()?),
            crate::sst::ExpX::Binary(a,b,c) =>
                ExprX::Binary(a.try_into()?,b.clone().try_into()?,c.clone().try_into()?),
            crate::sst::ExpX::BinaryOpr(a,b,c) =>
                ExprX::Binary(a.try_into()?,b.clone().try_into()?,c.clone().try_into()?),
            crate::sst::ExpX::If(a, b, c) =>
                ExprX::If(
                    a.clone().try_into()?,
                    b.clone().try_into()?,
                    c.clone().try_into()?),
            crate::sst::ExpX::WithTriggers(_, body) => {
                (&body.x).try_into()?
            }
            crate::sst::ExpX::StaticVar(_) |
            crate::sst::ExpX::VarLoc(_) |
            crate::sst::ExpX::VarAt(_, _) |
            crate::sst::ExpX::Loc(_) |
            crate::sst::ExpX::Old(_, _) |
            crate::sst::ExpX::NullaryOpr(_) |
            crate::sst::ExpX::Bind(_, _) |
            crate::sst::ExpX::ExecFnByName(_) |
            crate::sst::ExpX::Interp(_) |
            crate::sst::ExpX::FuelConst(_) =>
            {
                return Err(format!("Unsupported expression: {:?}", v))
            }
        })
    }
}

pub fn param(p: crate::ast::Param) -> Result<Binder<Typ>,<Typ as TryFrom<crate::ast::Typ>>::Error> {
    Ok(Arc::new(air::ast::BinderX{
        name: varident(&p.x.name.clone()),
        a: p.x.typ.clone().try_into()?,
    }))
}

pub fn params(p: crate::ast::Params) -> Result<Binders<Typ>,<Typ as TryFrom<crate::ast::Typ>>::Error> {
    Ok(Arc::new(
        p.as_ref().clone().into_iter().map(|p| param(p))
            .collect::<Result<Vec<_>,_>>()?
    ))
}

pub fn decl(p: crate::sst::LocalDecl) -> Result<Binder<Typ>,<Typ as TryFrom<crate::ast::Typ>>::Error> {
    Ok(Arc::new(air::ast::BinderX{
        name: varident(&p.ident.clone()),
        a: p.typ.clone().try_into()?,
    }))
}

pub fn decls(p: Vec<crate::sst::LocalDecl>) -> Result<Binders<Typ>,<Typ as TryFrom<crate::ast::Typ>>::Error> {
    Ok(Arc::new(
        p.clone().into_iter().map(|p| decl(p))
            .collect::<Result<Vec<_>,_>>()?
    ))
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

//impl TryFrom<crate::sst::FunctionSst> for Decl {
//    type Error = String;
//    fn try_from(v: crate::sst::FunctionSst) -> Result<Self,Self::Error> {
//        let ps = decls(v.local_decls)?;
//        Ok(Decl::Theorem(Theorem {
//            name: v.path.clone(),
//            typ_params: v.typ_params.clone(),
//            params: ps,
//            require: v.require.clone().try_into()?,
//            ensure: v.ensure.clone().try_into()?,
//        }))
//    }
//}
