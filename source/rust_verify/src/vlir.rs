//! The Verus-Lean AST Abstract Syntax Tree
//!
//! Rust-AST --> Rust-HIR --> VIR-AST --> VLIR-AST --> Lean
//!
//! Right now this is just used to control how Serde serializes it

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use vir::ast::{
    BinaryOp, BinaryOpr, CallTarget, Fun, Ident, Idents, IntRange, MultiOp, NullaryOpr, Path, Primitive, Quant, UnaryOp, UnaryOpr
};

fn varident(v: vir::ast::VarIdent) -> Ident {
    v.0
}

#[derive(Clone,Debug,Serialize,Deserialize)]
pub struct Binder<A: Clone>(Arc<BinderX<A>>);
#[derive(Clone,Debug,Serialize,Deserialize)]
pub struct Binders<A: Clone>(Arc<Vec<Binder<A>>>);

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BinderX<A: Clone> {
    pub name: Ident,
    pub a: A,
}

impl<T0: Clone, T: From<T0> + Clone> From<vir::ast::Binders<T0>> for Binders<T> {
    fn from(value: vir::ast::Binders<T0>) -> Self {
        Binders(Arc::new(value.as_ref().clone().into_iter().map(|b|
            b.into()
        ).collect()))
    }
}

impl<T0: Clone, T: From<T0> + Clone> From<vir::ast::Binder<T0>> for Binder<T> {
    fn from(value: vir::ast::Binder<T0>) -> Self {
        Binder(Arc::new(value.as_ref().clone().into()))
    }
}

impl<T0: Clone, T: From<T0> + Clone> From<air::ast::BinderX<T0>> for BinderX<T> {
    fn from(value: air::ast::BinderX<T0>) -> Self {
        BinderX {
            name: value.name,
            a: value.a.into(),
        }
    }
}

impl<T0: Clone, T: From<T0> + Clone> From<vir::ast::VarBinders<T0>> for Binders<T> {
    fn from(value: vir::ast::VarBinders<T0>) -> Self {
        Binders(Arc::new(value.as_ref().clone().into_iter().map(|b|
            b.into()
        ).collect()))
    }
}

impl<T0: Clone, T: From<T0> + Clone> From<vir::ast::VarBinder<T0>> for Binder<T> {
    fn from(value: vir::ast::VarBinder<T0>) -> Self {
        Binder(Arc::new(value.as_ref().clone().into()))
    }
}

impl<T0: Clone, T: From<T0> + Clone> From<vir::ast::VarBinderX<T0>> for BinderX<T> {
    fn from(value: vir::ast::VarBinderX<T0>) -> Self {
        BinderX {
            name: varident(value.name),
            a: value.a.into(),
        }
    }
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
    Int { _0: IntRange },
    /// UTF-8 character type
    Char,
    /// Tuple type (t1, ..., tn).  Note: ast_simplify replaces Tuple with Datatype.
    Tuple { _0: Typs },
    /// `FnSpec` type (TODO rename from 'Lambda' to just 'FnSpec')
    /// (t1, ..., tn) -> t0.
    Lambda(Typs, Typ),
    /// Executable function types (with a requires and ensures)
    AnonymousClosure(Typs, Typ, usize),
    /// Corresponds to Rust's FnDef type
    /// Typs are generic type args
    /// If Fun is a trait function, then the Option<Fun> has the statically resolved
    /// function if it exists. Similar to ImplPaths, this is technically redundant
    /// (because it follows from the types), but it is not easy to compute without
    /// storing it here. We need it because it is useful for determining which
    /// FnDef axioms to introduce.
    FnDef(Fun, Typs, Option<Fun>),
    /// Datatype (concrete or abstract) applied to type arguments
    Datatype(Path, Typs),
    /// StrSlice type. Currently the vstd StrSlice struct is "seen" as this type
    /// despite the fact that it is in fact a datatype
    StrSlice,
    /// Other primitive type (applied to type arguments)
    Primitive(Primitive, Typs),
    /// Type parameter (inherently SMT-boxed, and cannot be unboxed)
    TypParam { _0: Ident },
    /// Projection such as <D as T<S>>::X or <A as T>::X (SMT-boxed, and can sometimes be unboxed)
    Projection {
        // trait_typ_args[0] is Self type
        trait_typ_args: Typs,
        trait_path: Path,
        name: Ident,
    },
    /// Type of type identifiers
    TypeId,
    /// Const integer type argument (e.g. for array sizes)
    ConstInt { _0: num_bigint::BigInt },
    /// AIR type, used internally during translation
    Unsupported {
        _0: vir::ast::Typ
    },
}

impl From<vir::ast::Typs> for Typs {
    fn from(v: vir::ast::Typs) -> Self {
        Typs(Arc::new(v.as_ref().clone().into_iter().map(|x| x.into()).collect()))
    }
}

impl From<vir::ast::Typ> for Typ {
    fn from(value: vir::ast::Typ) -> Self {
        match value.as_ref() {
            vir::ast::TypX::Bool => Typ(Arc::new(TypX::Bool)),
            vir::ast::TypX::Int(r) => Typ(Arc::new(TypX::Int {_0: *r})),
            vir::ast::TypX::Char => Typ(Arc::new(TypX::Char)),
            vir::ast::TypX::Tuple(ts) => Typ(Arc::new(TypX::Tuple {_0: ts.clone().into()})),
            vir::ast::TypX::Lambda(t1, t2) => Typ(Arc::new(TypX::Lambda(t1.clone().into(), t2.clone().into()))),
            vir::ast::TypX::Datatype(a,b,_) => Typ(Arc::new(TypX::Datatype(a.clone(),b.clone().into()))),
            vir::ast::TypX::AnonymousClosure(_, _, _) |
            vir::ast::TypX::FnDef(_, _, _) |
            vir::ast::TypX::StrSlice |
            vir::ast::TypX::Primitive(_, _) |
            vir::ast::TypX::Decorate(_, _) |
            vir::ast::TypX::Boxed(_) |
            vir::ast::TypX::TypParam(_) |
            vir::ast::TypX::Projection { trait_typ_args: _, trait_path: _, name: _ } |
            vir::ast::TypX::TypeId |
            vir::ast::TypX::ConstInt(_) |
            vir::ast::TypX::Air(_) => Typ(Arc::new(TypX::Unsupported {_0: value})),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum Constant {
    /// true or false
    Bool { _0: bool },
    /// integer of arbitrary size
    Int {
        #[serde(serialize_with = "bigint_serial")]
        _0: num_bigint::BigInt
    },
    /// Hold generated string slices in here
    StrSlice { _0: Arc<String> },
    // Hold unicode values here
    Char { _0: char },
}

fn bigint_serial<S: serde::Serializer>(a: &num_bigint::BigInt, b: S) -> Result<S::Ok,S::Error> {
    b.serialize_str(&a.to_string())
}

impl From<vir::ast::Constant> for Constant {
    fn from(v: vir::ast::Constant) -> Self {
        match v {
            vir::ast::Constant::Bool(b) => Constant::Bool { _0: b },
            vir::ast::Constant::Int(i) => Constant::Int {_0: i},
            vir::ast::Constant::StrSlice(s) => Constant::StrSlice {_0: s},
            vir::ast::Constant::Char(c) => Constant::Char {_0: c},
        }
    }
}


#[derive(Clone,Debug,Serialize,Deserialize)]
pub struct Expr(Arc<ExprX>);
#[derive(Clone,Debug,Serialize,Deserialize)]
pub struct Exprs(Arc<Vec<Expr>>);

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ExprX {
    /// Constant
    Const {
        _0: Constant
    },
    /// Local variable as a right-hand side
    Var {
        _0: Ident
    },
    /// Call to a function passing some expression arguments
    Call(CallTarget, Exprs),
    /// Construct datatype value of type Path and variant Ident,
    /// with field initializers Binders<Expr> and an optional ".." update expression.
    /// For tuple-style variants, the fields are named "_0", "_1", etc.
    /// Fields can appear **in any order** even for tuple variants.
    Ctor(Path, Ident, Binders<Expr>, Option<Expr>),
    /// Primitive 0-argument operation
    NullaryOpr {
        _0: NullaryOpr
    },
    /// Primitive unary operation
    Unary(UnaryOp, Expr),
    /// Special unary operator
    UnaryOpr(UnaryOpr, Expr),
    /// Primitive binary operation
    Binary(BinaryOp, Expr, Expr),
    /// Special binary operation
    BinaryOpr(BinaryOpr, Expr, Expr),
    /// Primitive multi-operand operation
    Multi(MultiOp, Exprs),
    /// Quantifier (forall/exists), binding the variables in Binders, with body Expr
    Quant(Quant, Binders<Typ>, Expr),
    /// Specification closure
    Closure(Binders<Typ>, Expr),
    /// Executable closure
    ExecClosure {
        params: Binders<Typ>,
        body: Expr,
        requires: Exprs,
        ensures: Exprs,
        ret: Binder<Typ>,
    },
    /// Array literal (can also be used for sequence literals in the future)
    ArrayLiteral {
        _0: Exprs
    },
    /// Executable function (declared with 'fn' and referred to by name)
    ExecFnByName {
        _0: Fun
    },
    /// Choose specification values satisfying a condition, compute body
    Choose { params: Binders<Typ>, cond: Expr, body: Expr },
    // /// Assign to local variable
    // /// init_not_mut = true ==> a delayed initialization of a non-mutable variable
    // Assign { init_not_mut: bool, lhs: Expr, rhs: Expr, op: Option<BinaryOp> },
    // /// Assert or assume
    // AssertAssume { is_assume: bool, expr: Expr },
    // /// Assert-forall or assert-by statement
    // AssertBy { vars: VarBinders<Typ>, require: Expr, ensure: Expr, proof: Expr },
    // /// `assert_by` with a dedicated prover option (nonlinear_arith, bit_vector)
    // AssertQuery { requires: Exprs, ensures: Exprs, proof: Expr, mode: AssertQueryMode },
    // /// Assertion discharged via computation
    // AssertCompute(Expr, ComputeMode),
    /// If-else
    If(Expr, Expr, Option<Expr>),
    // /// Sequence of statements, optionally including an expression at the end
    // Block(Stmts, Option<Expr>),
    Unsupported {
        _0: vir::ast::ExprX
    },
}


impl From<vir::ast::Expr> for Expr {
    fn from(v: Arc<vir::ast::SpannedTyped<vir::ast::ExprX>>) -> Self {
        Expr(Arc::new(<vir::ast::ExprX as Clone>::clone(&v.x).into()))
    }
}

impl From<vir::ast::Exprs> for Exprs {
    fn from(v: vir::ast::Exprs) -> Self {
        Exprs(Arc::new(<Vec<std::sync::Arc<_>> as Clone>::clone(&v).into_iter().map(|x| x.into()).collect()))
    }
}

impl From<vir::ast::ExprX> for ExprX {
    fn from(v: vir::ast::ExprX) -> ExprX {
        match v {
        vir::ast::ExprX::Const(c) => ExprX::Const { _0: c.into() },
        vir::ast::ExprX::Var(v) => ExprX::Var { _0: varident(v) },
        vir::ast::ExprX::VarLoc(_) => ExprX::Unsupported { _0: v },
        vir::ast::ExprX::VarAt(_,_) => ExprX::Unsupported { _0: v },
        vir::ast::ExprX::ConstVar(_, _) => ExprX::Unsupported { _0: v },
        vir::ast::ExprX::StaticVar(_) => ExprX::Unsupported { _0: v },
        vir::ast::ExprX::Loc(_) => ExprX::Unsupported { _0: v },
        vir::ast::ExprX::Call(a,b) => ExprX::Call(a.clone(),b.into()),
        vir::ast::ExprX::Tuple(_) => ExprX::Unsupported { _0: v },
        vir::ast::ExprX::Ctor(a,b,c,d) =>
            ExprX::Ctor(a.clone(),b.clone(),
                c.into(),
                d.map(|d| d.into())
            ),
        vir::ast::ExprX::NullaryOpr(a) =>
            ExprX::NullaryOpr { _0: a.clone() },
        vir::ast::ExprX::Unary(a,b) =>
            ExprX::Unary(a.clone(),b.into()),
        vir::ast::ExprX::UnaryOpr(a,b) =>
            ExprX::UnaryOpr(a.clone(),b.into()),
        vir::ast::ExprX::Binary(a,b,c) =>
            ExprX::Binary(a.clone(),b.into(),c.into()),
        vir::ast::ExprX::BinaryOpr(a,b,c) =>
            ExprX::BinaryOpr(a.clone(),b.into(),c.into()),
        vir::ast::ExprX::Multi(a,b) =>
            ExprX::Multi(a.clone(),b.into()),
        vir::ast::ExprX::Quant(a,b,c) =>
            ExprX::Quant(a.clone(),b.into(),c.into()),
        vir::ast::ExprX::Closure(a,b) =>
            ExprX::Closure(a.into(),b.into()),
        vir::ast::ExprX::ExecClosure { params, body, requires, ensures, ret, external_spec: _ } =>
            ExprX::ExecClosure {
                params: params.into(),
                body: body.into(),
                requires: requires.into(),
                ensures: ensures.into(),
                ret: ret.into() },
        vir::ast::ExprX::ArrayLiteral(a) =>
            ExprX::ArrayLiteral { _0: a.into() },
        vir::ast::ExprX::ExecFnByName(b) =>
            ExprX::ExecFnByName { _0: b.clone() },
        vir::ast::ExprX::Choose { params, cond, body } =>
            ExprX::Choose {
                params: params.into(),
                cond: cond.into(),
                body: body.into() },
        vir::ast::ExprX::WithTriggers { triggers: _, body } =>
            <vir::ast::ExprX as Clone>::clone(&body.x).into(),
        vir::ast::ExprX::Assign { init_not_mut: _, lhs: _, rhs: _, op: _ } =>
            ExprX::Unsupported { _0: v },
        vir::ast::ExprX::Fuel(a,b,c) =>
            ExprX::Unsupported { _0: vir::ast::ExprX::Fuel(a,b,c) },
        vir::ast::ExprX::RevealString(a) =>
            ExprX::Unsupported { _0: vir::ast::ExprX::RevealString(a) },
        vir::ast::ExprX::Header(_) =>
            ExprX::Unsupported { _0: v },
        vir::ast::ExprX::AssertAssume { is_assume: _, expr: _ } =>
            ExprX::Unsupported { _0: v },
        vir::ast::ExprX::AssertBy { vars: _, require: _, ensure: _, proof: _ } =>
            ExprX::Unsupported { _0: v },
        vir::ast::ExprX::AssertQuery { requires: _, ensures: _, proof: _, mode: _ } =>
            ExprX::Unsupported { _0: v },
        vir::ast::ExprX::AssertCompute(_, _) =>
            ExprX::Unsupported { _0: v },
        vir::ast::ExprX::If(a, b,c) =>
            ExprX::If(a.into(), b.into(), c.map(|c| c.into())),
        vir::ast::ExprX::Match(_, _) =>
            ExprX::Unsupported { _0: v },
        vir::ast::ExprX::Loop { loop_isolation: _, is_for_loop: _, label: _, cond: _, body: _, invs: _, decrease: _ } =>
            ExprX::Unsupported { _0: v },
        vir::ast::ExprX::OpenInvariant(_,_,_,_) =>
            ExprX::Unsupported { _0: v },
        vir::ast::ExprX::Return(_) =>
            ExprX::Unsupported { _0: v },
        vir::ast::ExprX::BreakOrContinue { label: _, is_break: _ } =>
            ExprX::Unsupported { _0: v },
        vir::ast::ExprX::Ghost { alloc_wrapper: _, tracked: _, expr: _ } =>
            ExprX::Unsupported { _0: v },
        vir::ast::ExprX::Block(_,_) =>
            ExprX::Unsupported { _0: v },
        vir::ast::ExprX::AirStmt(_) =>
            ExprX::Unsupported { _0: v },
        }
    }
}

impl From<vir::ast::Param> for Binder<Typ> {
    fn from(p: vir::ast::Param) -> Self {
        Binder( Arc::new( BinderX{
            name: varident(p.as_ref().clone().x.name),
            a: p.as_ref().clone().x.typ.into(),
        }))
    }
}

impl From<vir::ast::Params> for Binders<Typ> {
    fn from(p: vir::ast::Params) -> Self {
        Binders(Arc::new(p.as_ref().clone().into_iter().map(|p| p.into()).collect()))
    }
}

#[derive(Serialize,Deserialize)]
pub struct Function {
    /// Name of function
    pub name: Fun,
    // /// Proxy used to declare the spec of this function
    // /// (e.g., some function marked `external_fn_specification`)
    // pub proxy: Option<Path>,
    // /// Kind (translation to AIR is different for each different kind)
    // pub kind: FunctionKind,
    // /// Access control (public/private)
    // pub visibility: Visibility,
    // /// Owning module
    // pub owning_module: Option<Path>,
    // /// exec functions are compiled, proof/spec are erased
    // /// exec/proof functions can have requires/ensures, spec cannot
    // /// spec functions can be used in requires/ensures, proof/exec cannot
    // pub mode: Mode,
    // /// Default amount of fuel: 0 means opaque, >= 1 means visible
    // /// For recursive functions, fuel determines the number of unfoldings that the SMT solver sees
    // pub fuel: u32,
    /// Type parameters to generic functions
    /// (for trait methods, the trait parameters come first, then the method parameters)
    pub typ_params: Idents,
    // /// Type bounds of generic functions
    // pub typ_bounds: GenericBounds,
    /// Function parameters
    pub params: Binders<Typ>,
    /// Return value (unit return type is treated specially; see FunctionX::has_return in ast_util)
    pub ret: Binder<Typ>,
    /// Preconditions (requires for proof/exec functions, recommends for spec functions)
    pub require: Exprs,
    /// Postconditions (proof/exec functions only)
    pub ensure: Exprs,
    /// Decreases clause to ensure recursive function termination
    /// decrease.len() == 0 means no decreases clause
    /// decrease.len() >= 1 means list of expressions, interpreted in lexicographic order
    pub decrease: Exprs,
    /// If Expr is true for the arguments to the function,
    /// the function is defined according to the function body and the decreases clauses must hold.
    /// If Expr is false, the function is uninterpreted, the body and decreases clauses are ignored.
    pub decrease_when: Option<Expr>,
    // /// Prove termination with a separate proof function
    // pub decrease_by: Option<Fun>,
    // /// For broadcast_forall functions, poly sets this to Some((params, reqs ==> enss))
    // /// where params and reqs ==> enss use coerce_typ_to_poly rather than coerce_typ_to_native
    // pub broadcast_forall: Option<(Params, Expr)>,
    // /// Axioms (similar to broadcast axioms) for the FnDef type corresponding to
    // /// this function, if one is generated for this particular function.
    // /// Similar to 'external_spec' in the ExecClosure node, this is filled
    // /// in during ast_simplify.
    // pub fndef_axioms: Option<Exprs>,
    // /// MaskSpec that specifies what invariants the function is allowed to open
    // pub mask_spec: Option<MaskSpec>,
    // /// Allows the item to be a const declaration or static
    // pub item_kind: ItemKind,
    // /// For public spec functions, publish == None means that the body is private
    // /// even though the function is public, the bool indicates false = opaque, true = visible
    // /// the body is public
    // pub publish: Option<bool>,
    // /// Various attributes
    // pub attrs: FunctionAttrs,
    // /// Body of the function (may be None for foreign functions or for external_body functions)
    // pub body: Option<Expr>,
    // /// Extra dependencies, only used for for the purposes of recursion-well-foundedness
    // /// Useful only for trusted fns.
    // pub extra_dependencies: Vec<Fun>,
}

impl From<vir::ast::Function> for Function {
    fn from(v: vir::ast::Function) -> Function {
        Function {
            name: v.x.name.clone(),
            typ_params: v.x.typ_params.clone(),
            params: v.x.params.clone().into(),
            ret: v.x.ret.clone().into(),
            require: v.x.require.clone().into(),
            ensure: v.x.ensure.clone().into(),
            decrease: v.x.decrease.clone().into(),
            decrease_when: v.x.decrease_when.clone().map(|e| e.into()),
        }
    }
}
