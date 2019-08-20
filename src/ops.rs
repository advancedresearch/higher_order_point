//! Operator traits.

/// Operator for mapping input type into another.
pub trait Map<T, U> {
    /// The output type.
    type Output;
    /// Maps input into another.
    fn map<F: 'static + Fn(U) -> T + Clone + Send + Sync>(self, f: F) -> Self::Output;
}

/// Differential operator.
pub trait Diff {
    /// Returns the differential shape.
    fn diff(self, eps: f64) -> Self;
}

/// Dot operator.
pub trait Dot<Rhs = Self> {
    /// The output type.
    type Output;

    /// Returns the dot product.
    fn dot(self, other: Rhs) -> Self::Output;
}

/// Cross operator.
pub trait Cross<Rhs = Self> {
    /// The output type.
    type Output;

    /// Returns the cross product.
    fn cross(self, other: Rhs) -> Self::Output;
}

/// Norm operator.
pub trait Norm {
    /// The output type.
    type Output;

    /// Returns the norm.
    fn norm(self) -> Self::Output;
}

/// AABB operator.
pub trait AABB {
    /// The corner type.
    type Corner;
    /// Returns the minimum and maximum corner.
    fn aabb(&self) -> (Self::Corner, Self::Corner);
}
