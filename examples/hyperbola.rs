/*
This example shows how to generate a hyperbolic shape
by twisting circles connected by lines.
*/

extern crate piston;
extern crate sdl2_window;
extern crate opengl_graphics;
extern crate graphics;
extern crate higher_order_core;
extern crate higher_order_point as hop;
extern crate camera_controllers;
extern crate vecmath;

use std::sync::Arc;

use piston::*;
use sdl2_window::*;
use opengl_graphics::*;
use graphics::*;
use higher_order_core::*;
use hop::*;
use camera_controllers::*;
use vecmath::Matrix4;

fn main() {
    let opengl = OpenGL::V3_2;
    let settings = WindowSettings::new("Test", [512; 2])
        .exit_on_esc(true);
    let mut window: Sdl2Window = settings.build().unwrap();

    let mut capture = true;
    window.set_capture_cursor(capture);

    let mut time: f64 = 0.0;
    let mut events = Events::new(EventSettings::new());
    let mut gl = GlGraphics::new(opengl);
    let mut first_person_settings = FirstPersonSettings::keyboard_wasd();
    first_person_settings.speed_vertical = 16.0;
    first_person_settings.speed_horizontal = 16.0;
    let mut first_person = FirstPerson::new([0.0, 4.0, 4.0], first_person_settings);

    let get_projection = |w: &Sdl2Window| {
        let draw_size = w.draw_size();
        CameraPerspective {
            fov: 90.0, near_clip: 0.1, far_clip: 1000.0,
            aspect_ratio: (draw_size.width as f32) / (draw_size.height as f32)
        }.projection()
    };

    let model = vecmath::mat4_id();
    let mut projection = get_projection(&window);

    while let Some(e) = events.next(&mut window) {
        if capture {
            first_person.event(&e);
        }

        if let Some(args) = e.render_args() {
            gl.draw(args.viewport(), |c, g| {
                clear([1.0; 4], g);

                let mvp = model_view_projection(
                    model,
                    first_person.camera(args.ext_dt).orthogonal(),
                    projection
                );

                let mut renderer = Renderer::new();

                let n = 4;
                let hyperbola: HyperbolaFunc<(f64, usize, f64)> = Hyperbola {
                    height: Arc::new(move |(h, _, _)| h),
                    phase: Arc::new(move |(_, i, x)| 0.5 * x * (i as f64 * time).sin()),
                };

                for j in 0..n {
                    let y = j as f64 / n as f64;
                    for i in 0..n {
                        let x = i as f64 / (n-1) as f64;
                        let hy = hyperbola.call((y * 2.0 + 2.0, i, x));
                        let diag_surface = hy.surface();
                        let offset = [y * 16.0, 0.0, x * 16.0];
                        let p = hy.bottom() + offset;
                        let q = hy.top() + offset;

                        renderer.sample(&p, 50);
                        renderer.sample(&q, 50);
                        renderer.sample2(&(diag_surface.clone() + offset), [10, 30]);

                        let n = 3;
                        for i in 0..n {
                            let x = (i+1) as f64 / (n+1) as f64;
                            let s = hy.ring(x) + offset;
                            renderer.sample(&s, 30);
                        }
                    }
                }

                renderer.draw(&window, &mvp, &c, g);
            })
        }

        if let Some(args) = e.update_args() {
            time += args.dt;
        }

        if let Some(_) = e.resize_args() {
            projection = get_projection(&window);
        }

        if let Some(Button::Keyboard(Key::C)) = e.press_args() {
            capture = !capture;
            window.set_capture_cursor(capture);
        }
    }
}

pub type HyperbolaFunc<T> = Hyperbola<Arg<T>>;

#[derive(Clone)]
pub struct Hyperbola<T = ()> where f64: Ho<T> {
    height: Fun<T, f64>,
    phase: Fun<T, f64>,
}

pub trait Edge {
    type Output;
    fn top(&self) -> Self::Output;
    fn bottom(&self) -> Self::Output;
}

impl Edge for Hyperbola {
    type Output = PointFunc<f64>;

    fn top(&self) -> PointFunc<f64> {
        let phase = self.phase;
        (Point::circle() + [0.0, 0.0, self.height]).map(move |t| t + phase)
    }

    fn bottom(&self) -> PointFunc<f64> {
        Point::circle()
    }
}

impl<T> Edge for HyperbolaFunc<T>
    where T: 'static + Copy
{
    type Output = PointFunc<(T, f64)>;

    fn top(&self) -> PointFunc<(T, f64)> {
        let hyx = self.clone();
        let hyy = self.clone();
        let hyz = self.clone();
        Point {
            x: Arc::new(move |(a, b)| (hyx.call(a).top().x)(b)),
            y: Arc::new(move |(a, b)| (hyy.call(a).top().y)(b)),
            z: Arc::new(move |(a, b)| (hyz.call(a).top().z)(b)),
        }
    }

    fn bottom(&self) -> PointFunc<(T, f64)> {
        let hyx = self.clone();
        let hyy = self.clone();
        let hyz = self.clone();
        Point {
            x: Arc::new(move |(a, b)| (hyx.call(a).bottom().x)(b)),
            y: Arc::new(move |(a, b)| (hyy.call(a).bottom().y)(b)),
            z: Arc::new(move |(a, b)| (hyz.call(a).bottom().z)(b)),
        }
    }
}

impl<T> HyperbolaFunc<T> {
    pub fn call(&self, val: T) -> Hyperbola where T: 'static + Copy {
        <Hyperbola as Call<T>>::call(self, val)
    }
}

impl<T: Clone> Ho<Arg<T>> for Hyperbola {type Fun = HyperbolaFunc<T>;}
impl<T: Copy> Call<T> for Hyperbola
    where f64: Call<T>
{
    fn call(f: &Self::Fun, val: T) -> Hyperbola {
        Hyperbola::<()> {
            height: <f64 as Call<T>>::call(&f.height, val),
            phase: <f64 as Call<T>>::call(&f.phase, val),
        }
    }
}

pub trait Ring {
    type Output;
    fn ring(&self, t: f64) -> Self::Output;
}

impl Ring for Hyperbola {
    type Output = PointFunc<f64>;
    fn ring(&self, t: f64) -> PointFunc<f64> {
        let bottom = self.bottom();
        let top = self.top();
        hop::line(&bottom, &top, &t)
    }
}

impl<T: 'static + Copy> Ring for HyperbolaFunc<T> {
    type Output = PointFunc<(T, f64)>;
    fn ring(&self, t: f64) -> Self::Output {
        let bottom = self.bottom();
        let top = self.top();
        let r: PointFunc<(T, f64)> = hop::line(&bottom, &top, &t);
        let rx = r.x;
        let ry = r.y;
        let rz = r.z;
        PointFunc::<(T, f64)> {
            x: Arc::new(move |a| rx(a)),
            y: Arc::new(move |a| ry(a)),
            z: Arc::new(move |a| rz(a)),
        }
    }
}

pub trait Surface {
    type Output;
    fn surface(&self) -> Self::Output;
}

impl Surface for Hyperbola {
    type Output = PointFunc<[f64; 2]>;
    fn surface(&self) -> PointFunc<[f64; 2]> {
        let hyx = self.clone();
        let hyy = self.clone();
        let hyz = self.clone();
        Point {
            x: Arc::new(move |p| hyx.ring(p[1]).call(p[0]).x),
            y: Arc::new(move |p| hyy.ring(p[1]).call(p[0]).y),
            z: Arc::new(move |p| hyz.ring(p[1]).call(p[0]).z),
        }
    }
}

impl<T: 'static + Copy> Surface for HyperbolaFunc<T> {
    type Output = PointFunc<(T, [f64; 2])>;
    fn surface(&self) -> Self::Output {
        let hyx = self.clone();
        let hyy = self.clone();
        let hyz = self.clone();
        Point {
            x: Arc::new(move |(a, p)| hyx.call(a).ring(p[1]).call(p[0]).x),
            y: Arc::new(move |(a, p)| hyy.call(a).ring(p[1]).call(p[0]).y),
            z: Arc::new(move |(a, p)| hyz.call(a).ring(p[1]).call(p[0]).z),
        }
    }
}

/// A crappy 3D point renderer.
pub struct Renderer {
    pub points: Vec<Point>,
}

impl Renderer {
    pub fn new() -> Renderer {
        Renderer {
            points: vec![]
        }
    }

    pub fn sample(&mut self, p: &PointFunc<f64>, n: usize) {
        for i in 0..n {
            let x = i as f64 / n as f64;
            self.points.push(p.call(x));
        }
    }

    pub fn sample2(&mut self, p: &PointFunc<[f64; 2]>, n: [usize; 2]) {
        for i in 0..n[0] {
            for j in 0..n[1] {
                self.points.push(p.call([i as f64 / n[0] as f64, j as f64 / n[1] as f64]));
            }
        }
    }

    pub fn draw(&self, window: &impl Window, mvp: &Matrix4<f32>, c: &Context, g: &mut impl Graphics) {
        let rad = 0.01;
        let draw_size = window.draw_size();
        let halfw = draw_size.width / 2.0;
        let tr = c.transform.trans(halfw, draw_size.height / 2.0).scale(halfw, -halfw);
        for p in &self.points {
            let p = [p.x as f32, p.y as f32, p.z as f32, 1.0];
            let p = vecmath::col_mat4_transform(*mvp, p);
            if p[2] < 0.0 {continue};
            let p = vecmath::vec4_scale(p, 1.0/p[3]);
            rectangle(
                [0.0, 0.0, 0.0, 1.0],
                [p[0] as f64 - rad, p[1] as f64 - rad, 2.0 * rad, 2.0 * rad],
                tr, g
            );
        }
    }
}
