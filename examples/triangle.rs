/*
This example shows how to use higher order maps with triangles.

Here, we define a point function of continuous triangle indices.
Continuous triangle indices creates a smooth map from corner to corner,
such that interpolating between transformated primitives is possible.

Various geometric primitives on the triangle can then be constructed:

- Triangle surface, e.g. `[0.0, 1.0, 2.0]`
- Triangle edge, e.g. `[0.0, 1.0]`
- Triangle corner, e.g. `[[0.0, 1.0], [1.0, 2.0]]`

Rotating the triangle from corner to corner can be done using
a function that maps indices to indices.

A higher order map preserves the structure of geometric primitives.
*/

extern crate higher_order_point;
extern crate higher_order_core;

use higher_order_core::{Func, HMap, HPair};
use higher_order_point::*;

use std::sync::Arc;

fn main() {
    // 2 pi.
    let tau = 6.283185307179586;
    // Define triangle coordinates by continuous indices.
    let p = PointFunc::<f64> {
        x: Arc::new(move |i| (i / 3.0 * tau).cos()),
        y: Arc::new(move |i| (i / 3.0 * tau).sin()),
        z: k(0.0)
    };
    // Define triangle indices.
    let triangle = [0.0, 1.0, 2.0];
    // Define edge indices.
    let edge = [0.0, 1.0];
    // Define a function that rotates triangle indices.
    let rotate: Func<f64, f64> = Arc::new(move |x| (x+1.0)%3.0);

    let xs: [f64; 3] = triangle.hmap(&p.x);
    let ys: [f64; 3] = triangle.hmap(&p.y);
    let zs: [f64; 3] = triangle.hmap(&p.z);
    println!("triangle {:?}\n\txs: {:?}\n\tys: {:?}\n\tzs: {:?}", triangle, xs, ys, zs);

    let triangle_coords: [Point; 3] = triangle.hmap(&p);
    println!("triangle coords {:?}:", triangle);
    for i in 0..3 {
        println!("\t{:?}", triangle_coords[i]);
    }

    let rotated_triangle: [f64; 3] = triangle.hmap(&rotate);

    let rotated_triangle_coords: [Point; 3] = rotated_triangle.hmap(&p);
    println!("rotated triangle coords {:?}:", rotated_triangle);
    for i in 0..3 {
        println!("\t{:?}", rotated_triangle_coords[i]);
    }

    let xs: [f64; 2] = edge.hmap(&p.x);
    let ys: [f64; 2] = edge.hmap(&p.y);
    let zs: [f64; 2] = edge.hmap(&p.z);
    println!("edge {:?}\n\txs: {:?}\n\tys: {:?}\n\tzs: {:?}", edge, xs, ys, zs);

    let edge_coords: [Point; 2] = edge.hmap(&p);
    println!("edge coords {:?}:", edge);
    for i in 0..2 {
        println!("\t{:?}", edge_coords[i]);
    }

    let rotated_edge: [f64; 2] = edge.hmap(&rotate);

    let rotated_edge_coords: [Point; 2] = rotated_edge.hmap(&p);
    println!("rotated edge coords {:?}:", rotated_edge);
    for i in 0..2 {
        println!("\t{:?}", rotated_edge_coords[i]);
    }

    let corner = [[0.0, 1.0], [1.0, 2.0]];
    let xs: [[f64; 2]; 2] = corner.hmap(&p.x);
    let ys: [[f64; 2]; 2] = corner.hmap(&p.y);
    let zs: [[f64; 2]; 2] = corner.hmap(&p.z);
    println!("corner {:?}\n\txs: {:?}\n\tys: {:?}\n\tzs: {:?}", corner, xs, ys, zs);

    let corner_coords: [[Point; 2]; 2] = corner.hmap(&p);
    println!("corner coords {:?}:", corner);
    for i in 0..2 {
        println!("\t[");
        for j in 0..2 {
            println!("\t\t{:?}", corner_coords[i][j]);
        }
        println!("\t],");
    }

    let rotated_corner: [[f64; 2]; 2] = corner.hmap(&rotate);

    let rotated_corner_coords: [[Point; 2]; 2] = rotated_corner.hmap(&p);
    println!("rotated corner coords {:?}:", rotated_corner);
    for i in 0..2 {
        println!("\t[");
        for j in 0..2 {
            println!("\t\t{:?}", rotated_corner_coords[i][j]);
        }
        println!("\t],");
    }

    // Interpolate 50% between continuous indices `a` and `b`.
    let in_between: Func<(f64, f64), f64> = Arc::new(move |(a, mut b)| {
        // Fix interpolation such that it rotates same direction for all points.
        if b < a {b += 3.0};
        (a+(b-a)*0.5)%3.0
    });
    let in_between_edge: [f64; 2] = (edge, rotated_edge).hpair().hmap(&in_between);
    let in_between_edge_coords: [Point; 2] = in_between_edge.hmap(&p);
    println!("in-between rotated edge coords {:?}:", in_between_edge);
    for i in 0..2 {
        println!("\t{:?}", in_between_edge_coords[i]);
    }
    let in_between_corner: [[f64; 2]; 2] = (corner, rotated_corner).hpair().hmap(&in_between);
    let in_between_corner_coords: [[Point; 2]; 2] = in_between_corner.hmap(&p);
    println!("in-between rotated corner coords {:?}:", in_between_corner);
    for i in 0..2 {
        println!("\t[");
        for j in 0..2 {
            println!("\t\t{:?}", in_between_corner_coords[i][j]);
        }
        println!("\t],");
    }
}
