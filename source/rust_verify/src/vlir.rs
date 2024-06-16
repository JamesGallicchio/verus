//! The Verus-Lean AST Abstract Syntax Tree
//!
//! Rust-AST --> Rust-HIR --> VIR-AST --> VLIR-AST --> Lean
//!
//! Right now this is just used to control how Serde serializes it

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use vir::ast::{Idents, MultiOp, Quant, AutospecUsage, BinaryOp, BinaryOpr, CallTarget, Constant, Fun, Ident, Mode, NullaryOpr, Path, Typ, UnaryOp, UnaryOpr, VarAt};

#[derive(Clone,Debug,Serialize,Deserialize)]
struct Binder<T: Clone>(Arc<BinderX<T>>);
#[derive(Clone,Debug,Serialize,Deserialize)]
struct Binders<T: Clone>(Arc<Vec<Binder<T>>>);

#[derive(Clone, Debug, Serialize, Deserialize)]
struct BinderX<T: Clone>(air::ast::BinderX<T>);

impl<T0: Clone, T: From<T0> + Clone> From<air::ast::Binder<T0>> for Binder<T> {
    fn from(x: air::ast::Binder<T0>) -> Binder<T> {
        Binder(Arc::new(<air::ast::BinderX<T0> as Clone>::clone(&x).into()))
    }
}

impl<T0: Clone, T: From<T0> + Clone> From<air::ast::Binders<T0>> for Binders<T> {
    fn from(x: air::ast::Binders<T0>) -> Binders<T> {
        Binders(Arc::new(
            x.iter().map(|x| x.clone().into()).collect()
        ))
    }
}

impl<T0: Clone, T: From<T0> + Clone> From<air::ast::BinderX<T0>> for BinderX<T> {
    fn from(x: air::ast::BinderX<T0>) -> BinderX<T> {
        BinderX(air::ast::BinderX {
            name: x.name.clone(),
            a: x.a.into(),
        })
    }
}

#[derive(Clone,Debug,Serialize,Deserialize)]
struct VarBinder<A: Clone>(Arc<VarBinderX<A>>);
#[derive(Clone,Debug,Serialize,Deserialize)]
struct VarBinders<A: Clone>(Arc<Vec<VarBinder<A>>>);

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VarBinderX<A: Clone> {
    pub name: VarIdent,
    pub a: A,
}

impl<T0: Clone, T: From<T0> + Clone> From<vir::ast::VarBinder<T0>> for VarBinder<T> {
    fn from(x: vir::ast::VarBinder<T0>) -> Self {
        VarBinder(Arc::new(<vir::ast::VarBinderX<T0> as Clone>::clone(&x).into()))
    }
}

impl<T0: Clone, T: From<T0> + Clone> From<vir::ast::VarBinders<T0>> for VarBinders<T> {
    fn from(x: vir::ast::VarBinders<T0>) -> Self {
        VarBinders(Arc::new(
            x.iter().map(|x| <std::sync::Arc<vir::ast::VarBinderX<T0>> as Clone>::clone(&x).into()).collect()
        ))
    }
}

impl<T0: Clone, T: From<T0> + Clone> From<vir::ast::VarBinderX<T0>> for VarBinderX<T> {
    fn from(x: vir::ast::VarBinderX<T0>) -> VarBinderX<T> {
        VarBinderX {
            name: x.name.into(),
            a: x.a.into(),
        }
    }
}


#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum VarIdentDisambiguate {
    // AIR names that don't derive from rustc's names:
    AirLocal,
    // rustc's parameter unique id comes from the function body; no body means no id:
    NoBodyParam,
    // TypParams are normally Idents, but sometimes we mix TypParams into lists of VarIdents:
    TypParamBare,
    TypParamSuffixed,
    TypParamDecorated,
    // Fields are normally Idents, but sometimes we mix field names into lists of VarIdents:
    Field,
    RustcId(usize),
    // We track whether the variable is SST/AIR statement-bound or expression-bound,
    // to help drop unnecessary ids from expression-bound variables
    VirRenumbered { is_stmt: bool, does_shadow: bool, id: u64 },
    // Some expression-bound variables don't need an id
    VirExprNoNumber,
    // We rename parameters to VirParam if the parameters don't conflict with each other
    VirParam,
    // Recursive definitions have an extra copy of the parameters
    VirParamRecursion(usize),
    // Capture-avoiding substitution creates new names:
    VirSubst(u64),
    VirTemp(u64),
    ExpandErrorsDecl(u64),
}

impl From<vir::ast::VarIdentDisambiguate> for VarIdentDisambiguate {
    fn from(value: vir::ast::VarIdentDisambiguate) -> Self {
        match value {
            vir::ast::VarIdentDisambiguate::AirLocal => VarIdentDisambiguate::AirLocal,
            vir::ast::VarIdentDisambiguate::NoBodyParam => VarIdentDisambiguate::NoBodyParam,
            vir::ast::VarIdentDisambiguate::TypParamBare => VarIdentDisambiguate::TypParamBare,
            vir::ast::VarIdentDisambiguate::TypParamSuffixed => VarIdentDisambiguate::TypParamSuffixed,
            vir::ast::VarIdentDisambiguate::TypParamDecorated => VarIdentDisambiguate::TypParamDecorated,
            vir::ast::VarIdentDisambiguate::Field => VarIdentDisambiguate::Field,
            vir::ast::VarIdentDisambiguate::RustcId(id) => VarIdentDisambiguate::RustcId(id),
            vir::ast::VarIdentDisambiguate::VirRenumbered { is_stmt, does_shadow, id } =>
                VarIdentDisambiguate::VirRenumbered {is_stmt, does_shadow, id},
            vir::ast::VarIdentDisambiguate::VirExprNoNumber => VarIdentDisambiguate::VirExprNoNumber,
            vir::ast::VarIdentDisambiguate::VirParam => VarIdentDisambiguate::VirParam,
            vir::ast::VarIdentDisambiguate::VirParamRecursion(id) => VarIdentDisambiguate::VirParamRecursion(id),
            vir::ast::VarIdentDisambiguate::VirSubst(id) => VarIdentDisambiguate::VirSubst(id),
            vir::ast::VarIdentDisambiguate::VirTemp(id) => VarIdentDisambiguate::VirTemp(id),
            vir::ast::VarIdentDisambiguate::ExpandErrorsDecl(id) => VarIdentDisambiguate::ExpandErrorsDecl(id),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VarIdent(pub Ident, pub VarIdentDisambiguate);

impl From<vir::ast::VarIdent> for VarIdent {
    fn from(value: vir::ast::VarIdent) -> Self {
        VarIdent(value.0,value.1.into())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Params(Arc<Vec<Param>>);
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Param(Arc<ParamX>);

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ParamX {
    pub name: VarIdent,
    pub typ: Typ,
    pub mode: Mode,
    /// An &mut parameter
    pub is_mut: bool,
    /// If the parameter uses a Ghost(x) or Tracked(x) pattern to unwrap the value, this is
    /// the mode of the resulting unwrapped x variable (Spec for Ghost(x), Proof for Tracked(x)).
    /// We also save a copy of the original wrapped name for lifetime_generate
    pub unwrapped_info: Option<(Mode, VarIdent)>,
}

impl From<vir::ast::Param> for Param {
    fn from(v: vir::ast::Param) -> Self {
        Param(Arc::new(<vir::ast::ParamX as Clone>::clone(&v.x).into()))
    }
}

impl From<vir::ast::Params> for Params {
    fn from(v: vir::ast::Params) -> Self {
        Params(Arc::new(v.iter().map(|v| <std::sync::Arc<vir::def::Spanned<vir::ast::ParamX>> as Clone>::clone(&v).into()).collect()))
    }
}

impl From<vir::ast::ParamX> for ParamX {
    fn from(value: vir::ast::ParamX) -> Self {
        ParamX {
            name: value.name.into(),
            typ: value.typ.into(),
            mode: value.mode,
            is_mut: value.is_mut,
            unwrapped_info: value.unwrapped_info.map(|(a,b)| (a,b.into())),
        }
    }
}

#[derive(Clone,Debug,Serialize,Deserialize)]
struct Expr(Arc<ExprX>);
#[derive(Clone,Debug,Serialize,Deserialize)]
struct Exprs(Arc<Vec<Expr>>);

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ExprX {
    /// Constant
    Const(Constant),
    /// Local variable as a right-hand side
    Var(VarIdent),
    /// Local variable as a left-hand side
    VarLoc(VarIdent),
    /// Local variable, at a different stage (e.g. a mutable reference in the post-state)
    VarAt(VarIdent, VarAt),
    /// Use of a const variable.  Note: ast_simplify replaces this with Call.
    ConstVar(Fun, AutospecUsage),
    /// Use of a static variable.
    StaticVar(Fun),
    /// Mutable reference (location)
    Loc(Expr),
    /// Call to a function passing some expression arguments
    Call(CallTarget, Exprs),
    /// Note: ast_simplify replaces this with Ctor
    Tuple(Exprs),
    /// Construct datatype value of type Path and variant Ident,
    /// with field initializers Binders<Expr> and an optional ".." update expression.
    /// For tuple-style variants, the fields are named "_0", "_1", etc.
    /// Fields can appear **in any order** even for tuple variants.
    Ctor(Path, Ident, Binders<Expr>, Option<Expr>),
    /// Primitive 0-argument operation
    NullaryOpr(NullaryOpr),
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
    Quant(Quant, VarBinders<Typ>, Expr),
    /// Specification closure
    Closure(VarBinders<Typ>, Expr),
    /// Executable closure
    ExecClosure {
        params: VarBinders<Typ>,
        body: Expr,
        requires: Exprs,
        ensures: Exprs,
        ret: VarBinder<Typ>,
    },
    /// Array literal (can also be used for sequence literals in the future)
    ArrayLiteral(Exprs),
    /// Executable function (declared with 'fn' and referred to by name)
    ExecFnByName(Fun),
    /// Choose specification values satisfying a condition, compute body
    Choose { params: VarBinders<Typ>, cond: Expr, body: Expr },
    /// Assign to local variable
    /// init_not_mut = true ==> a delayed initialization of a non-mutable variable
    Assign { init_not_mut: bool, lhs: Expr, rhs: Expr, op: Option<BinaryOp> },
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
    Unsupported(vir::ast::ExprX),
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
        vir::ast::ExprX::Const(c) =>
            ExprX::Const(c.clone()),
        vir::ast::ExprX::Var(v) =>
            ExprX::Var(v.into()),
        vir::ast::ExprX::VarLoc(v) =>
            ExprX::VarLoc(v.into()),
        vir::ast::ExprX::VarAt(v, a) =>
            ExprX::VarAt(v.into(), a),
        vir::ast::ExprX::ConstVar(f, au) =>
            ExprX::ConstVar(f.clone(), au),
        vir::ast::ExprX::StaticVar(f) =>
            ExprX::StaticVar(f.clone()),
        vir::ast::ExprX::Loc(a) =>
            ExprX::Loc(a.into()),
        vir::ast::ExprX::Call(a,b) =>
            ExprX::Call(a.clone(),b.into()),
        vir::ast::ExprX::Tuple(a) =>
            ExprX::Tuple(a.into()),
        vir::ast::ExprX::Ctor(a,b,c,d) =>
            ExprX::Ctor(a.clone(),b.clone(),
                c.into(),
                d.map(|d| d.into())
            ),
        vir::ast::ExprX::NullaryOpr(a) =>
            ExprX::NullaryOpr(a.clone()),
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
        vir::ast::ExprX::ExecClosure { params, body, requires, ensures, ret, external_spec } =>
            ExprX::ExecClosure {
                params: params.into(),
                body: body.into(),
                requires: requires.into(),
                ensures: ensures.into(),
                ret: ret.into() },
        vir::ast::ExprX::ArrayLiteral(a) =>
            ExprX::ArrayLiteral(a.into()),
        vir::ast::ExprX::ExecFnByName(b) =>
            ExprX::ExecFnByName(b.clone()),
        vir::ast::ExprX::Choose { params, cond, body } =>
            ExprX::Choose {
                params: params.into(),
                cond: cond.into(),
                body: body.into() },
        vir::ast::ExprX::WithTriggers { triggers: _, body } =>
            <vir::ast::ExprX as Clone>::clone(&body.x).into(),
        vir::ast::ExprX::Assign { init_not_mut, lhs, rhs, op } =>
            ExprX::Assign {
                init_not_mut,
                lhs: lhs.into(),
                rhs: rhs.into(),
                op: op.clone() },
        vir::ast::ExprX::Fuel(a,b,c) =>
            ExprX::Unsupported(vir::ast::ExprX::Fuel(a,b,c)),
        vir::ast::ExprX::RevealString(a) =>
            ExprX::Unsupported(vir::ast::ExprX::RevealString(a)),
        vir::ast::ExprX::Header(_) =>
            ExprX::Unsupported(v),
        vir::ast::ExprX::AssertAssume { is_assume: _, expr: _ } =>
            ExprX::Unsupported(v),
        vir::ast::ExprX::AssertBy { vars: _, require: _, ensure: _, proof: _ } =>
            ExprX::Unsupported(v),
        vir::ast::ExprX::AssertQuery { requires: _, ensures: _, proof: _, mode: _ } =>
            ExprX::Unsupported(v),
        vir::ast::ExprX::AssertCompute(_, _) =>
            ExprX::Unsupported(v),
        vir::ast::ExprX::If(a, b,c) =>
            ExprX::If(a.into(), b.into(), c.map(|c| c.into())),
        vir::ast::ExprX::Match(_, _) =>
            ExprX::Unsupported(v),
        vir::ast::ExprX::Loop { loop_isolation: _, is_for_loop: _, label: _, cond: _, body: _, invs: _, decrease: _ } =>
            ExprX::Unsupported(v),
        vir::ast::ExprX::OpenInvariant(_,_,_,_) =>
            ExprX::Unsupported(v),
        vir::ast::ExprX::Return(_) =>
            ExprX::Unsupported(v),
        vir::ast::ExprX::BreakOrContinue { label: _, is_break: _ } =>
            ExprX::Unsupported(v),
        vir::ast::ExprX::Ghost { alloc_wrapper: _, tracked: _, expr: _ } =>
            ExprX::Unsupported(v),
        vir::ast::ExprX::Block(_,_) =>
            ExprX::Unsupported(v),
        vir::ast::ExprX::AirStmt(_) =>
            ExprX::Unsupported(v),
        }
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
    /// exec functions are compiled, proof/spec are erased
    /// exec/proof functions can have requires/ensures, spec cannot
    /// spec functions can be used in requires/ensures, proof/exec cannot
    pub mode: Mode,
    // /// Default amount of fuel: 0 means opaque, >= 1 means visible
    // /// For recursive functions, fuel determines the number of unfoldings that the SMT solver sees
    // pub fuel: u32,
    /// Type parameters to generic functions
    /// (for trait methods, the trait parameters come first, then the method parameters)
    pub typ_params: Idents,
    // /// Type bounds of generic functions
    // pub typ_bounds: GenericBounds,
    /// Function parameters
    pub params: Params,
    /// Return value (unit return type is treated specially; see FunctionX::has_return in ast_util)
    pub ret: Param,
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
            mode: v.x.mode.clone(),
            typ_params: v.x.typ_params.clone(),
            params: (v.x.params.clone()).into(),
            ret: v.x.ret.clone().into(),
            require: v.x.require.clone().into(),
            ensure: v.x.ensure.clone().into(),
            decrease: v.x.decrease.clone().into(),
            decrease_when: v.x.decrease_when.clone().map(|e| e.into()),
        }
    }
}
