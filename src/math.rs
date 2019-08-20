//! Math functions.

use super::*;

/// Adds two functions.
pub fn add<T: 'static + Copy>(a: Func<T, f64>, b: Func<T, f64>) -> Func<T, f64> {
    Arc::new(move |x| a(x) + b(x))
}

/// Subtracts two functions.
pub fn sub<T: 'static + Copy>(a: Func<T, f64>, b: Func<T, f64>) -> Func<T, f64> {
    Arc::new(move |x| a(x) - b(x))
}

/// Adds a new argument to the right.
pub fn lift_right<T, U: 'static, V: 'static>(f: Func<U, V>) -> Func<(U, T), V> {
    Arc::new(move |(a, _)| f(a))
}

/// Adds a new argument to the left.
pub fn lift_left<T, U: 'static, V: 'static>(f: Func<U, V>) -> Func<(T, U), V> {
    Arc::new(move |(_, a)| f(a))
}

/// Returns identity function.
pub fn id<T>() -> Func<T, T> {
    Arc::new(move |a| a)
}

/// Returns step function.
/// This is zero for negative numbers and one for positive numbers.
pub fn step() -> Func<f64, f64> {
    Arc::new(move |a| if a < 0.0 {0.0} else {1.0})
}

/// Returns floor function.
pub fn floor() -> Func<f64, f64> {
    Arc::new(move |a| a.floor())
}

/// Returns zero function.
pub fn zero<T>() -> Func<T, f64> {
    Arc::new(move |_| 0.0)
}

/// Returns one function.
pub fn one<T>() -> Func<T, f64> {
    Arc::new(move |_| 1.0)
}

/// Returns a constant.
pub fn k<T>(v: f64) -> Func<T, f64> {
    Arc::new(move |_| v)
}

/// Zips two functions, such that it alternates between them.
pub fn zip(a: Func<f64, f64>, b: Func<f64, f64>) -> Func<f64, f64> {
    Arc::new(move |t| if t % 2.0 < 1.0 {
        a(t % 1.0 + (t/2.0).floor())
    } else {
        b(t % 1.0 + ((t-1.0)/2.0).floor())
    })
}

/// Returns the `y` component for `x` on a half circle.
pub fn half_circle() -> Func<f64, f64> {
    Arc::new(move |x| (1.0 - x * x).sqrt())
}

/// Maps input type into another.
pub fn map<F, T, U, V>(a: Func<U, V>, f: F) -> Func<T, V>
where F: 'static + Fn(T) -> U + Send + Sync, U: 'static, V: 'static {
    Arc::new(move |x| a(f(x)))
}

/// Creates a linear combination of two shapes.
pub fn line<T: Clone, U: Clone, V: Clone>(a: &T, b: &U, t: &V) ->
<T as Add<<<U as Sub<T>>::Output as Mul<V>>::Output>>::Output
    where U: Sub<T>,
          <U as Sub<T>>::Output: Mul<V>,
          T: Add<<<U as Sub<T>>::Output as Mul<V>>::Output>
{
    let a1 = a.clone();
    let a2 = a.clone();
    let b = b.clone();
    let t = t.clone();
    a1 + (b - a2) * t
}

/// Constructs a cubic bezier.
#[macro_export]
macro_rules! qbez(
    ($a:expr, $b:expr, $c:expr, $t:expr) => {
        line(&line($a, $b, $t), &line($b, $c, $t), $t)
    }
);

/// Constructs a cubic bezier.
#[macro_export]
macro_rules! cbez(
    ($a:expr, $b:expr, $c:expr, $d:expr, $t:expr) => {
        line(&line($a, $b, $t), &line($c, $d, $t), $t)
    }
);

/// Mathematical constant for 360 degrees in radians.
pub const TAU: f64 = 6.283185307179586;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zip() {
        let a: Func<f64, f64> = Arc::new(move |t| t);
        let b: Func<f64, f64> = Arc::new(move |t| -t);
        let c = zip(a.clone(), b.clone());
        assert_eq!(c(0.0), a(0.0));
        assert_eq!(c(0.5), a(0.5));
        assert_eq!(c(1.0), b(0.0));
        assert_eq!(c(1.5), b(0.5));
        assert_eq!(c(2.0), a(1.0));
        assert_eq!(c(2.5), a(1.5));
        assert_eq!(c(3.0), b(1.0));
        assert_eq!(c(3.5), b(1.5));
        assert_eq!(c(4.0), a(2.0));
        assert_eq!(c(4.5), a(2.5));
    }
}
