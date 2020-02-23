//! # Higher Order Point

#![deny(missing_docs)]

extern crate higher_order_core;

use higher_order_core::*;

use std::sync::Arc;
use std::ops::{Add, Div, Mul, Sub};

/// A point function.
pub type PointFunc<T> = Point<Arg<T>>;

pub use math::*;
pub use ops::*;

pub mod math;
pub mod ops;

/// 3D point.
#[derive(Clone)]
pub struct Point<T = ()> where f64: Ho<T> {
    /// Function for x-coordinates.
    pub x: Fun<T, f64>,
    /// Function for y-coordinates.
    pub y: Fun<T, f64>,
    /// Function for z-coordinates.
    pub z: Fun<T, f64>,
}

impl Copy for Point {}

impl PointFunc<[f64; 2]> {
    /// Returns ground plane with zero z-values.
    pub fn ground_plane() -> Self {
        Point {
            x: Arc::new(move |p: [f64; 2]| p[0]),
            y: Arc::new(move |p: [f64; 2]| p[1]),
            z: zero(),
        }
    }
}

impl PointFunc<[f64; 3]> {
    /// Returns plain Euclidean space.
    pub fn space() -> Self {
        Point {
            x: Arc::new(move |p: [f64; 3]| p[0]),
            y: Arc::new(move |p: [f64; 3]| p[1]),
            z: Arc::new(move |p: [f64; 3]| p[2]),
        }
    }
}

impl std::fmt::Debug for Point {
    fn fmt(&self, w: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        write!(w, "Point {{x: {:?}, y: {:?}, z: {:?}}}",
            self.x,
            self.y,
            self.z
        )
    }
}

impl PartialEq for Point {
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x && self.y == other.y && self.z == other.z
    }
}

impl Dot for Point {
    type Output = f64;
    fn dot(self, other: Self) -> f64 {
        self.x * other.x +
        self.y * other.y +
        self.z * other.z
    }
}

impl<T: 'static + Copy> Dot for PointFunc<T> {
    type Output = Func<T, f64>;
    fn dot(self, other: Self) -> Func<T, f64> {
        let ax = self.x;
        let ay = self.y;
        let az = self.z;
        let bx = other.x;
        let by = other.y;
        let bz = other.z;
        Arc::new(move |a| ax(a) * bx(a) + ay(a) * by(a) + az(a) * bz(a))
    }
}

impl Cross for Point {
    type Output = Point;
    fn cross(self, other: Self) -> Self {
        let a = self;
        let b = other;
        Point {
            x: a.y * b.z - a.z * b.y,
            y: a.z * b.x - a.x * b.z,
            z: a.x * b.y - a.y * b.x,
        }
    }
}

impl<T: 'static + Copy> Cross for PointFunc<T> {
    type Output = Self;
    fn cross(self, other: Self) -> Self {
        let ax1 = self.x.clone();
        let ax2 = self.x;
        let ay1 = self.y.clone();
        let ay2 = self.y;
        let az1 = self.z.clone();
        let az2 = self.z;
        let bx1 = other.x.clone();
        let bx2 = other.x;
        let by1 = other.y.clone();
        let by2 = other.y;
        let bz1 = other.z.clone();
        let bz2 = other.z;
        Point {
            x: Arc::new(move |v| ay1(v) * bz1(v) - az1(v) * by1(v)),
            y: Arc::new(move |v| az2(v) * bx1(v) - ax1(v) * bz2(v)),
            z: Arc::new(move |v| ax2(v) * by2(v) - ay2(v) * bx2(v)),
        }
    }
}

impl Norm for Point {
    type Output = f64;
    fn norm(self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }
}

impl<T: 'static + Copy> Norm for PointFunc<T> {
    type Output = Func<T, f64>;
    fn norm(self) -> Func<T, f64> {
        let fx = self.x;
        let fy = self.y;
        let fz = self.z;
        Arc::new(move |v| (fx(v).powi(2) + fy(v).powi(2) + fz(v).powi(2)).sqrt())
    }
}

impl<T: 'static> PointFunc<T> {
    /// Adds another parameter to the right.
    pub fn lift_right<U>(self) -> PointFunc<(T, U)> {
        let fx = self.x;
        let fy = self.y;
        let fz = self.z;
        Point {
            x: Arc::new(move |(a, _)| fx(a)),
            y: Arc::new(move |(a, _)| fy(a)),
            z: Arc::new(move |(a, _)| fz(a)),
        }
    }

    /// Adds another parameter to the left.
    pub fn lift_left<U>(self) -> PointFunc<(U, T)> {
        let fx = self.x;
        let fy = self.y;
        let fz = self.z;
        Point {
            x: Arc::new(move |(_, a)| fx(a)),
            y: Arc::new(move |(_, a)| fy(a)),
            z: Arc::new(move |(_, a)| fz(a)),
        }
    }

    /// Helper method for calling value.
    pub fn call(&self, val: T) -> Point where T: Copy {
        <Point as Call<T>>::call(self, val)
    }
}

impl<T: 'static, U> Map<T, U> for PointFunc<T> {
    type Output = PointFunc<U>;
    /// Maps input into another.
    fn map<F: 'static + Fn(U) -> T + Clone + Send + Sync>(self, f: F) -> PointFunc<U> {
        let fx = self.x;
        let fy = self.y;
        let fz = self.z;
        let f1 = f.clone();
        let f2 = f.clone();
        let f3 = f;
        Point {
            x: Arc::new(move |v| fx(f1(v))),
            y: Arc::new(move |v| fy(f2(v))),
            z: Arc::new(move |v| fz(f3(v))),
        }
    }
}

impl PointFunc<f64> {
    /// Creates a new circle in the xy-plane.
    pub fn circle() -> Self {
        Point {
            x: Arc::new(move |ang: f64| (ang * TAU).cos()),
            y: Arc::new(move |ang: f64| (ang * TAU).sin()),
            z: zero(),
        }
    }

    /// Creates a new circle in xy-plane that uses radians.
    pub fn circle_radians() -> Self {
        Point {
            x: Arc::new(move |ang: f64| ang.cos()),
            y: Arc::new(move |ang: f64| ang.sin()),
            z: zero(),
        }
    }

    /// Creates a new zig-zag function in the xy-plane.
    pub fn zig_zag() -> Self {
        Point {
            x: zip(id(), Arc::new(move |a| a.floor() + 1.0)),
            y: zip(floor(), id()),
            z: zero(),
        }
    }

    /// Creates a new zag-zig function in the xy-plane.
    pub fn zag_zig() -> Self {
        Point {
            x: zip(floor(), id()),
            y: zip(id(), Arc::new(move |a| a.floor() + 1.0)),
            z: zero(),
        }
    }

    /// Points along the x-axis.
    pub fn x() -> Self {
        Point {
            x: Arc::new(move |v| v),
            y: zero(),
            z: zero(),
        }
    }

    /// Points along the y-axis.
    pub fn y() -> Self {
        Point {
            x: zero(),
            y: Arc::new(move |v| v),
            z: zero(),
        }
    }

    /// Points along the z-axis.
    pub fn z() -> Self {
        Point {
            x: zero(),
            y: zero(),
            z: Arc::new(move |v| v),
        }
    }
}

impl Diff for PointFunc<f64> {
    fn diff(self, eps: f64) -> Self {
        let fx = self.x;
        let fy = self.y;
        let fz = self.z;
        Point {
            x: Arc::new(move |t| (fx(t+eps)-fx(t))/eps),
            y: Arc::new(move |t| (fy(t+eps)-fy(t))/eps),
            z: Arc::new(move |t| (fz(t+eps)-fz(t))/eps),
        }
    }
}

impl<T: 'static + Copy> From<PointFunc<(T, T)>> for PointFunc<[T; 2]> {
    fn from(val: PointFunc<(T, T)>) -> Self {
        let fx = val.x;
        let fy = val.y;
        let fz = val.z;
        Point {
            x: Arc::new(move |a| fx((a[0], a[1]))),
            y: Arc::new(move |a| fy((a[0], a[1]))),
            z: Arc::new(move |a| fz((a[0], a[1]))),
        }
    }
}

impl<T: Clone> Ho<Arg<T>> for Point {
    type Fun = PointFunc<T>;
}

impl<T: Copy> Call<T> for Point
    where f64: Ho<Arg<T>> + Call<T>
{
    fn call(f: &Self::Fun, val: T) -> Point {
        Point::<()> {
            x: <f64 as Call<T>>::call(&f.x, val),
            y: <f64 as Call<T>>::call(&f.y, val),
            z: <f64 as Call<T>>::call(&f.z, val),
        }
    }
}

impl From<Point> for [f64; 3] {
    fn from(val: Point) -> [f64; 3] {
        [val.x, val.y, val.z]
    }
}

impl Into<Point> for [f64; 3] {
    fn into(self) -> Point {
        Point {x: self[0], y: self[1], z: self[2]}
    }
}

impl<T> Into<PointFunc<T>> for [f64; 3] {
    fn into(self) -> PointFunc<T> {
        let x = self[0];
        let y = self[1];
        let z = self[2];
        Point {
            x: Arc::new(move |_| x),
            y: Arc::new(move |_| y),
            z: Arc::new(move |_| z),
        }
    }
}

impl Add<f64> for Point {
    type Output = Self;

    fn add(self, val: f64) -> Self {
        Point {
            x: self.x + val,
            y: self.y + val,
            z: self.z + val,
        }
    }
}

impl Add for Point {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Point {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl<T: 'static + Copy> Add for PointFunc<T> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Point {
            x: add(self.x, other.x),
            y: add(self.y, other.y),
            z: add(self.z, other.z),
        }
    }
}

impl<T: 'static> Add<PointFunc<T>> for Point {
    type Output = PointFunc<T>;

    fn add(self, other: PointFunc<T>) -> PointFunc<T> {
        let ax = self.x;
        let ay = self.y;
        let az = self.z;
        let bx = other.x;
        let by = other.y;
        let bz = other.z;
        Point {
            x: Arc::new(move |a: T| ax + bx(a)),
            y: Arc::new(move |a: T| ay + by(a)),
            z: Arc::new(move |a: T| az + bz(a)),
        }
    }
}

impl<T: 'static, U: Into<Point>> Add<U> for PointFunc<T> {
    type Output = PointFunc<T>;
    fn add(self, other: U) -> PointFunc<T> {
        let other = other.into();
        let ax = self.x;
        let ay = self.y;
        let az = self.z;
        let bx = other.x;
        let by = other.y;
        let bz = other.z;
        Point {
            x: Arc::new(move |a: T| ax(a) + bx),
            y: Arc::new(move |a: T| ay(a) + by),
            z: Arc::new(move |a: T| az(a) + bz),
        }
    }
}

impl<T: 'static + Copy> Add<Func<T, Point>> for PointFunc<T> {
    type Output = PointFunc<T>;
    fn add(self, other: Func<T, Point>) -> PointFunc<T> {
        let ax = self.x;
        let ay = self.y;
        let az = self.z;
        let ox = other.clone();
        let oy = other.clone();
        let oz = other;
        Point {
            x: Arc::new(move |t: T| ax(t) + ox(t).x),
            y: Arc::new(move |t: T| ay(t) + oy(t).y),
            z: Arc::new(move |t: T| az(t) + oz(t).z),
        }
    }
}

impl Sub<f64> for Point {
    type Output = Self;
    fn sub(self, val: f64) -> Self {
        Point {
            x: self.x - val,
            y: self.y - val,
            z: self.z - val,
        }
    }
}

impl Sub for Point {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Point {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl<T: 'static + Copy> Sub for PointFunc<T> {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Point {
            x: sub(self.x, other.x),
            y: sub(self.y, other.y),
            z: sub(self.z, other.z),
        }
    }
}

impl<T: 'static> Sub<PointFunc<T>> for Point {
    type Output = PointFunc<T>;
    fn sub(self, other: PointFunc<T>) -> PointFunc<T> {
        let ax = self.x;
        let ay = self.y;
        let az = self.z;
        let bx = other.x;
        let by = other.y;
        let bz = other.z;
        Point {
            x: Arc::new(move |a: T| ax - bx(a)),
            y: Arc::new(move |a: T| ay - by(a)),
            z: Arc::new(move |a: T| az - bz(a)),
        }
    }
}

impl<T: 'static, U: Into<Point>> Sub<U> for PointFunc<T> {
    type Output = Self;
    fn sub(self, other: U) -> Self {
        let other = other.into();
        let ax = self.x;
        let ay = self.y;
        let az = self.z;
        let bx = other.x;
        let by = other.y;
        let bz = other.z;
        Point {
            x: Arc::new(move |a: T| ax(a) - bx),
            y: Arc::new(move |a: T| ay(a) - by),
            z: Arc::new(move |a: T| az(a) - bz),
        }
    }
}

impl Mul<f64> for Point {
    type Output = Self;
    fn mul(self, other: f64) -> Self {
        Point {
            x: self.x * other,
            y: self.y * other,
            z: self.z * other,
        }
    }
}

impl<T: 'static> Mul<f64> for PointFunc<T> {
    type Output = Self;
    fn mul(self, other: f64) -> Self {
        let x = self.x;
        let y = self.y;
        let z = self.z;
        Point {
            x: Arc::new(move |a: T| x(a) * other),
            y: Arc::new(move |a: T| y(a) * other),
            z: Arc::new(move |a: T| z(a) * other),
        }
    }
}

impl<T: 'static> Mul<Func<T, f64>> for Point {
    type Output = PointFunc<T>;
    fn mul(self, other: Func<T, f64>) -> PointFunc<T> {
        let x = self.x;
        let y = self.y;
        let z = self.z;
        let ox = other.clone();
        let oy = other.clone();
        let oz = other;
        Point {
            x: Arc::new(move |t| x * ox(t)),
            y: Arc::new(move |t| y * oy(t)),
            z: Arc::new(move |t| z * oz(t)),
        }
    }
}

impl<T: 'static + Copy> Mul<Func<T, f64>> for PointFunc<T> {
    type Output = Self;
    fn mul(self, other: Func<T, f64>) -> Self {
        let x = self.x;
        let y = self.y;
        let z = self.z;
        let ox = other.clone();
        let oy = other.clone();
        let oz = other;
        Point {
            x: Arc::new(move |a: T| x(a) * ox(a)),
            y: Arc::new(move |a: T| y(a) * oy(a)),
            z: Arc::new(move |a: T| z(a) * oz(a)),
        }
    }
}

impl<T: 'static> Mul<PointFunc<T>> for Point {
    type Output = PointFunc<T>;
    fn mul(self, other: PointFunc<T>) -> PointFunc<T> {
        let x = self.x;
        let y = self.y;
        let z = self.z;
        let ox = other.x;
        let oy = other.y;
        let oz = other.z;
        Point {
            x: Arc::new(move |t| x * ox(t)),
            y: Arc::new(move |t| y * oy(t)),
            z: Arc::new(move |t| z * oz(t)),
        }
    }
}

impl<T: 'static + Copy> Mul for PointFunc<T> {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        let ax = self.x;
        let ay = self.y;
        let az = self.z;
        let bx = other.x;
        let by = other.y;
        let bz = other.z;
        Point {
            x: Arc::new(move |t| ax(t) * bx(t)),
            y: Arc::new(move |t| ay(t) * by(t)),
            z: Arc::new(move |t| az(t) * bz(t)),
        }
    }
}

impl<T: 'static> Div<f64> for PointFunc<T> {
    type Output = Self;
    fn div(self, other: f64) -> Self {
        let fx = self.x;
        let fy = self.y;
        let fz = self.z;
        Point {
            x: Arc::new(move |v| fx(v) / other),
            y: Arc::new(move |v| fy(v) / other),
            z: Arc::new(move |v| fz(v) / other),
        }
    }
}

impl<T: 'static + Copy> Div<Func<T, f64>> for PointFunc<T> {
    type Output = Self;
    fn div(self, other: Func<T, f64>) -> Self {
        let fx = self.x;
        let fy = self.y;
        let fz = self.z;
        let ox = other.clone();
        let oy = other.clone();
        let oz = other;
        Point {
            x: Arc::new(move |v| fx(v) / ox(v)),
            y: Arc::new(move |v| fy(v) / oy(v)),
            z: Arc::new(move |v| fz(v) / oz(v)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bool_arg() {
        let p: Point = Point {x: 2.0, y: 4.0, z: 6.0};
        assert_eq!(p.x, 2.0);

        let p: PointFunc<_> = Point {
            x: Arc::new(move |b| if b {1.0} else {2.0}),
            y: Arc::new(move |b| if b {3.0} else {4.0}),
            z: Arc::new(move |_| 0.0),
        };
        assert_eq!((p.x)(true), 1.0);
        assert_eq!((p.y)(false), 4.0);
        let q1 = p.call(true);
        assert_eq!(q1.x, 1.0);
        assert_eq!(q1.y, 3.0);
        let q2 = p.call(false);
        assert_eq!(q2.x, 2.0);
        assert_eq!(q2.y, 4.0);

        let p: PointFunc<_> = Point {
            x: k(1.0),
            y: k(2.0),
            z: zero(),
        };
        let q = p.call(());
        assert_eq!(q.x, 1.0);
        assert_eq!(q.y, 2.0);
    }

    #[test]
    fn cylinder() {
        let p: PointFunc<_> = Point {
            x: Arc::new(move |(ang, _height): (f64, f64)| ang.cos()),
            y: Arc::new(move |(ang, _height): (f64, f64)| ang.sin()),
            z: zero(),
        };
        let q: PointFunc<_> = Point {
            x: zero(),
            y: zero(),
            z: Arc::new(move |(_, height): (f64, f64)| height),
        };
        let r = p + q;
        let r1 = r.call((0.0, 0.0));
        assert_eq!(r1.x, 1.0);
        assert_eq!(r1.y, 0.0);
        assert_eq!(r1.z, 0.0);
        let r2 = r.call((0.0, 1.0));
        assert_eq!(r2.x, 1.0);
        assert_eq!(r2.y, 0.0);
        assert_eq!(r2.z, 1.0);
    }

    #[test]
    fn lift() {
        let p = Point::circle();
        let q = Point::z();
        let r = p.lift_right::<f64>() + q.lift_left::<f64>();
        let r1 = r.call((0.0, 0.0));
        assert_eq!(r1.x, 1.0);
        assert_eq!(r1.y, 0.0);
        assert_eq!(r1.z, 0.0);
        let r2 = r.call((0.0, 1.0));
        assert_eq!(r2.x, 1.0);
        assert_eq!(r2.y, 0.0);
        assert_eq!(r2.z, 1.0);
    }

    #[test]
    fn mul_scalar() {
        let p = Point::circle();
        let q = p * 0.5;
        assert_eq!((q.x)(0.0), 0.5);
        assert_eq!((q.y)(0.0), 0.0);
        assert!(((q.x)(0.25) - 0.0).abs() < 0.0000000001);
        assert_eq!((q.y)(0.25), 0.5);
    }

    #[test]
    fn disc() {
        let p = Point::circle();
        let q = id();
        let r = p.lift_right::<f64>() * lift_left::<f64, _, _>(q);
        let r1 = r.call((0.0, 1.0));
        assert_eq!(r1.x, 1.0);
        assert_eq!(r1.y, 0.0);
        let r2 = r.call((0.25, 0.75));
        assert!((r2.x - 0.0).abs() < 0.0000000001);
        assert_eq!(r2.y, 0.75);
    }

    #[test]
    fn into() {
        let p = Point::circle();
        let q = p.lift_right::<f64>();
        let r: PointFunc<[f64; 2]> = q.into();
        let r1 = r.call([0.0, 0.0]);
        assert_eq!(r1.x, 1.0);
        assert_eq!(r1.y, 0.0);
    }

    #[test]
    fn line_shape() {
        let p = Point::circle().lift_right::<f64>();
        let q: Point = Point {x: 0.0, y: 0.0, z: 0.0};
        let t = lift_left::<f64, f64, f64>(id());
        let r1 = line(&p, &q, &t);
        let r2 = line(&q, &p, &t);

        let r1a = r1.call((0.0, 0.0));
        assert_eq!(r1a.x, 1.0);
        assert_eq!(r1a.y, 0.0);
        let r1b = r1.call((0.0, 1.0));
        assert_eq!(r1b.x, 0.0);
        assert_eq!(r1b.y, 0.0);

        let r2a = r2.call((0.0, 0.0));
        assert_eq!(r2a.x, 0.0);
        assert_eq!(r2a.y, 0.0);
        let r2b = r2.call((0.0, 1.0));
        assert_eq!(r2b.x, 1.0);
        assert_eq!(r2b.y, 0.0);
    }

    #[test]
    fn qbez_shape() {
        // Create 3 circles that are transported along a line.
        let a = Point::circle() + [0.0; 3];
        let b = Point::circle() + [0.5, 0.0, 0.0];
        let c = Point::circle() + [1.0, 0.0, 0.0];

        let a = a.lift_right::<f64>();
        let b = b.lift_right::<f64>();
        let c = c.lift_right::<f64>();

        let t = lift_left::<f64, f64, f64>(id());

        let r = qbez!(&a, &b, &c, &t);

        let r1 = r.call((0.0, 0.0));
        assert_eq!(r1.x, 1.0);
        assert_eq!(r1.y, 0.0);
        let r2 = r.call((0.0, 0.5));
        assert_eq!(r2.x, 1.5);
        assert_eq!(r2.y, 0.0);
        let r3 = r.call((0.0, 1.0));
        assert_eq!(r3.x, 2.0);
        assert_eq!(r3.y, 0.0);

        let r1 = r.call((0.25, 0.0));
        assert!((r1.x - 0.0).abs() < 0.0000001);
        assert_eq!(r1.y, 1.0);
        let r2 = r.call((0.25, 0.5));
        assert_eq!(r2.x, 0.5);
        assert_eq!(r2.y, 1.0);
        let r3 = r.call((0.25, 1.0));
        assert_eq!(r3.x, 1.0);
        assert_eq!(r3.y, 1.0);
    }

    #[test]
    fn cbez_shape() {
        let a = Point::circle() - [0.0; 3];
        let b = Point::circle() - [0.5, 0.0, 0.0];
        let c = Point::circle() - [1.0, 0.0, 0.0];

        let a = a.lift_right::<f64>();
        let b = b.lift_right::<f64>();
        let c = c.lift_right::<f64>();

        let t = lift_left::<f64, f64, f64>(id());

        let r: PointFunc<(f64, f64)> = cbez!(&a, &b, &b, &c, &t);

        let r1 = r.call((0.0, 0.0));
        assert_eq!(r1.x, 1.0);
        let r2 = r.call((0.0, 0.5));
        assert_eq!(r2.x, 0.5);
        let r3 = r.call((0.0, 1.0));
        assert_eq!(r3.x, 0.0);
    }

    #[test]
    fn into_function() {
        let _: PointFunc<[f64; 2]> = [0.0; 3].into();
    }

    #[test]
    fn dot() {
        let a: PointFunc<f64> = [1.0, 0.0, 0.0].into();
        let b: PointFunc<f64> = [0.5, 0.5, 0.0].into();
        let c = a.dot(b);
        let c0 = c(0.0);
        assert_eq!(c0, 0.5);
    }

    #[test]
    fn cross() {
        let a = Point::circle();
        let b = Point::circle().map(|t| t + 0.25);
        let c = a.cross(b);

        let c1 = c.call(0.0);
        assert_eq!(c1.x, 0.0);
        assert_eq!(c1.y, 0.0);
        assert_eq!(c1.z, 1.0);

        let c2 = c.call(0.25);
        assert_eq!(c2.x, 0.0);
        assert_eq!(c2.y, 0.0);
        assert_eq!(c2.z, 1.0);
    }

    #[test]
    fn diff() {
        let a = Point::circle();
        let da = a.diff(0.00000001);

        let a1 = da.call(0.0);
        assert!((a1.x - 0.0).abs() < 0.000001);
        assert!((a1.y - TAU).abs() < 0.00001);
        assert_eq!(a1.z, 0.0);

        let a2 = da.call(0.25);
        assert!((a2.x + TAU).abs() < 0.000001);
        assert!((a2.y - 0.0).abs() < 0.00001);
        assert_eq!(a2.z, 0.0);
    }

    #[test]
    fn norm() {
        let a = Point::circle();
        let b = a.clone().norm();
        assert_eq!(b(0.0), 1.0);
        assert_eq!(b(0.25), 1.0);

        let _ = a.clone() / b;
        let _ = a / 2.0;
    }

    #[test]
    fn test_step() {
        let a = Point::circle() * step();

        let a1 = a.call(-0.001);
        assert_eq!(a1.x, 0.0);
        assert_eq!(a1.y, 0.0);
        assert_eq!(a1.z, 0.0);

        let a2 = a.call(0.0);
        assert_eq!(a2.x, 1.0);
        assert_eq!(a2.y, 0.0);
        assert_eq!(a2.z, 0.0);
    }

    #[test]
    fn test_floor() {
        let a = Point::circle() * floor();

        assert_eq!(a.call(-2.0).x, -2.0);
        assert_eq!(a.call(-1.0).x, -1.0);
        assert_eq!(a.call(0.0).x, 0.0);
        assert_eq!(a.call(1.0).x, 1.0);
    }

    #[test]
    fn zig_zag() {
        let a = Point::zig_zag();
        let a1 = a.call(0.0);
        assert_eq!(a1.x, 0.0);
        assert_eq!(a1.y, 0.0);
        let a2 = a.call(0.5);
        assert_eq!(a2.x, 0.5);
        assert_eq!(a2.y, 0.0);
        let a3 = a.call(1.0);
        assert_eq!(a3.x, 1.0);
        assert_eq!(a3.y, 0.0);
        let a4 = a.call(1.5);
        assert_eq!(a4.x, 1.0);
        assert_eq!(a4.y, 0.5);
        let a5 = a.call(2.0);
        assert_eq!(a5.x, 1.0);
        assert_eq!(a5.y, 1.0);
        let a6 = a.call(2.5);
        assert_eq!(a6.x, 1.5);
        assert_eq!(a6.y, 1.0);
        let a7 = a.call(3.0);
        assert_eq!(a7.x, 2.0);
        assert_eq!(a7.y, 1.0);
        let a8 = a.call(3.5);
        assert_eq!(a8.x, 2.0);
        assert_eq!(a8.y, 1.5);
        let a9 = a.call(4.0);
        assert_eq!(a9.x, 2.0);
        assert_eq!(a9.y, 2.0);
    }

    #[test]
    fn into_vec() {
        let a = Point {x: 0.0, y: 1.0, z: 2.0};
        let b: [f64; 3] = a.into();
        assert_eq!(b, [0.0, 1.0, 2.0]);
    }
}
