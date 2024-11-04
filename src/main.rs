#![feature(strict_overflow_ops, float_next_up_down)]

use macroquad::prelude::*;

type Float = f64;

/// The nearest representable value < 1.0
const ONE_NEXT_DOWN: Float = (1. as Float).next_down();

/// A triangle in CCW orientation with vertexes starting from the bottom right
/// corner in our XY plane.
struct Triangle([(DVec3, usize); 3]);

impl Triangle {
    /// Compute a plane that matches the triangle
    fn plane(&self) -> Plane {
        // Compute the normal of the triangle
        let normal = (self.0[1].0 - self.0[0].0)
            .cross(self.0[2].0 - self.0[0].0);

        // Compute offset of the plane
        let offset = normal.x * self.0[0].0.x + normal.y * self.0[0].0.y +
            normal.z * self.0[0].0.z;

        Plane { normal, offset }
    }
}

/// An infinite plane which can be sampled for `z` values from a given `xy`
struct Plane {
    /// Normal of the plane
    normal: DVec3,

    /// Z-offset of the plane from the origin
    offset: Float,
}

impl Plane {
    /// Get the point on a plane at a given `xy`
    fn get(&self, xy: DVec2) -> DVec3 {
        // Compute Z value for the point
        let z = -(self.normal.x * xy.x + self.normal.y * xy.y - self.offset) /
            self.normal.z;

        DVec3::new(xy.x, xy.y, z)
    }
}

/// A 2-dimensional surface map of a rectangular surface
///
/// The surface map is right-handed Z up
///
/// height (y, yres)
///   +-----+
///   |     |
///   +-----+ width (x, xres)
/// (0,0)
struct Surface {
    /// Raw data, indexed by y * xres + x
    data: Vec<Float>,

    /// Width (x axis)
    width: Float,

    /// Height (y axis)
    height: Float,

    /// Internal X resolution
    xres: usize,

    /// Internal Y resolution
    yres: usize,

    /// Internal mesh used during rendering. Used just to prevent reallocs of
    /// backing buffers
    mesh: Mesh,
}

impl Surface {
    /// Create a new surface
    fn new(width: Float, height: Float, xres: usize, yres: usize) -> Self {
        Self {
            data: vec![0.; xres.strict_mul(yres)],
            mesh: Mesh {
                vertices: Vec::new(),
                indices:  Vec::new(),
                texture:  None,
            },
            width, height, xres, yres,
        }
    }

    /// Gets the point at a given internal coordinate
    ///
    /// Returns the 3d point as well as the index into data that holds the Z
    /// information
    fn get_int(&self, x: usize, y: usize) -> (DVec3, usize) {
        // Fetch the data point
        let idx = y * self.xres + x;
        let z = self.data[idx];

        // Get the location of the internal point
        let xy = self.to_point(x as Float, y as Float);

        (DVec3::new(xy.x, xy.y, z), idx)
    }

    /// Take an internal point and compute the actual point for it
    fn to_point(&self, x: Float, y: Float) -> DVec2 {
        let x = x / self.xres.strict_sub(1) as Float * self.width;
        let y = y / self.yres.strict_sub(1) as Float * self.height;
        DVec2::new(x, y)
    }

    /// Take a point and compute the internal coordinate for the point
    fn to_int_point(&self, pt: DVec2) -> DVec2 {
        // Ensure the requested point is in bounds of the surface
        assert!(pt.x >= 0. && pt.y >= 0. &&
            pt.x <= self.width && pt.y <= self.height);

        // Compute normalized coords [0, 1)
        let pt_norm = DVec2::new(
            (pt.x / self.width).min(ONE_NEXT_DOWN),
            (pt.y / self.height).min(ONE_NEXT_DOWN),
        );

        // Compute zero-indexed internal coordinate based on the resolution of
        // the surface
        let pt_int = DVec2::new(
            pt_norm.x * self.xres.strict_sub(1) as Float,
            pt_norm.y * self.yres.strict_sub(1) as Float,
        );

        pt_int
    }

    /// Get the triangle that contains the point
    ///
    /// Returns the CCW triangle containing `pt` always starting from the
    /// bottom right.
    fn get_triangle(&self, pt: DVec2) -> Triangle {
        // Get the internal representation of the point
        let pt_int = self.to_int_point(pt);

        // Compute extents of the rectangle containing the point
        let left   = pt_int.x.floor() as usize;
        let right  = left + 1;
        let bottom = pt_int.y.floor() as usize;
        let top    = bottom + 1;

        // Find all corners of the rectangle this point belongs to
        let top_left     = self.get_int(left,  top);
        let top_right    = self.get_int(right, top);
        let bottom_left  = self.get_int(left,  bottom);
        let bottom_right = self.get_int(right, bottom);

        // We know that the bottom left and top right will always be used in
        // every triangle. Now we need to determine if the point is closer to
        // the top left or the bottom right.
        let dist_tl = pt.distance(top_left.0.xy());
        let dist_br = pt.distance(bottom_right.0.xy());

        // Determine the triangle the point falls in
        //
        // Since the two triangles _internally_ make a perfect square, if the
        // y coord is below the x coord, then it belongs to the bottom right.
        //
        // The "slope" of the internal triangle is 1.0, thus y below x means
        // it's below the slope, and thus part of the bottom right triangle.
        if pt_int.y.fract() < pt_int.x.fract() {
            // Use bottom right
            Triangle([bottom_left, bottom_right, top_right])
        } else {
            // Use top left
            Triangle([bottom_left, top_right, top_left])
        }
    }

    // Compute a hat for a set of ranges
    //
    // Ranges are (bottom left, top right)
    fn hat(&mut self, ranges: &[(DVec2, DVec2)]) {
        let mut points = Vec::new();

        // Compute all points which could potentially be the highest points
        // under a given range.
        for range in ranges {
            let bottom_left_int = self.to_int_point(range.0);
            let top_right_int   = self.to_int_point(range.1);

            let left   = range.0.x;
            let right  = range.1.x;
            let bottom = range.0.y;
            let top    = range.1.y;

            // Points checked in a range:
            //
            // - The four virtual corners of the range
            // - All virtual intersections between the range edges and the
            //   backing surface grid and triangles
            // - All real vertices inside the range

            // Create points for the four corners of the hat.
            points.push(self.get(DVec2::new(left,  bottom)));
            points.push(self.get(DVec2::new(right, bottom)));
            points.push(self.get(DVec2::new(left,  top)));
            points.push(self.get(DVec2::new(right, top)));

            // Find the four corners of the grid contained inside the hat
            let left_int   = bottom_left_int.x.ceil() as usize;
            let right_int  = top_right_int.x.floor() as usize;
            let bottom_int = bottom_left_int.y.ceil() as usize;
            let top_int    = top_right_int.y.floor() as usize;

            let left_int_fract   = bottom_left_int.x.fract();
            let right_int_fract  = top_right_int.x.fract();
            let bottom_int_fract = bottom_left_int.y.fract();
            let top_int_fract    = top_right_int.y.fract();

            for x in left_int..=right_int {
                // Intersections between the top and bottom edges and the grid
                let point = self.to_point(x as Float, 0.);
                points.push(self.get(DVec2::new(point.x, bottom)));
                points.push(self.get(DVec2::new(point.x, top)));
            }

            for y in bottom_int..=top_int {
                // Intersections between the left and right edges and the grid
                let point = self.to_point(0., y as Float);
                points.push(self.get(DVec2::new(left,  point.y)));
                points.push(self.get(DVec2::new(right, point.y)));
            }

            for x in left_int.saturating_sub(1)..=right_int {
                // Intersections between the top and bottom edges and the
                // triangle hypotenuses
                let pointb = self.to_point(x as Float + bottom_int_fract, 0.);
                let pointt = self.to_point(x as Float + top_int_fract,    0.);

                if pointb.x > left && pointb.x < right {
                    points.push(self.get(DVec2::new(pointb.x, bottom)));
                }

                if pointt.x > left && pointt.x < right {
                    points.push(self.get(DVec2::new(pointt.x, top)));
                }
            }

            for y in bottom_int.saturating_sub(1)..=top_int {
                // Intersections between the left and right edges and the
                // triangle hypotenuses
                let pointl = self.to_point(0., y as Float + left_int_fract);
                let pointr = self.to_point(0., y as Float + right_int_fract);

                if pointl.y > bottom && pointl.y < top {
                    points.push(self.get(DVec2::new(left, pointl.y)));
                }

                if pointr.y > bottom && pointr.y < top {
                    points.push(self.get(DVec2::new(right, pointr.y)));
                }
            }

            // Finally, create points for all internal grid vertexes
            for y in bottom_int..=top_int {
                for x in left_int..=right_int {
                    points.push(self.get(
                        self.to_point(x as Float, y as Float)));
                }
            }
        }

        let mut best_plane = (std::f64::MAX, None);
        for &va in points.iter() {
            for &vb in points.iter() {
                'next_combo: for &vc in points.iter() {
                    // Generate triangle for the three selected vertices
                    let triangle = Triangle([(va, 0), (vb, 0), (vc, 0)]);

                    // Compute plane for triangle
                    let plane = triangle.plane();

                    // Minimize distance to all points
                    let mut cum_distance = 0.;
                    for point in points.iter() {
                        let hat_z = plane.get(point.xy()).z;
                        let delta = hat_z - point.z;
                        if delta < -0.000001 {
                            // Hat is below a point, it's not a valid point!
                            continue 'next_combo;
                        }

                        cum_distance += delta * delta;
                    }

                    if cum_distance < best_plane.0 {
                        best_plane = (cum_distance, Some(plane));
                    }
                }
            }
        }

        let best_plane = best_plane.1.unwrap();

        self.mesh.vertices.clear();
        self.mesh.indices.clear();
        for range in ranges {
            let left   = range.0.x;
            let right  = range.1.x;
            let bottom = range.0.y;
            let top    = range.1.y;

            macro_rules! push_vert {
                ($x:expr, $y:expr) => {
                    let point = best_plane.get(DVec2::new($x, $y));
                    self.mesh.vertices.push(Vertex::new(
                        point.x as f32, point.y as f32, point.z as f32,
                        0., 0., Color::new(1., 0., 0., 0.5)));
                }
            }

            push_vert!(left,  bottom);
            push_vert!(right, top);
            push_vert!(left,  top);

            push_vert!(left,  bottom);
            push_vert!(right, bottom);
            push_vert!(right, top);

            for _ in 0..6 {
                self.mesh.indices.push(self.mesh.indices.len() as u16);
            }
        }

        // For debugging, render all points considered in the hatting
        for point in points {
            let dist = best_plane.get(point.xy()).z - point.z;
            if dist.abs() < 0.000001 {
                draw_sphere(point.as_vec3(), 0.02, None, PINK);
            } else {
                draw_sphere(point.as_vec3(), 0.005, None, ORANGE);
            }
        }

        // Render the hat
        draw_mesh(&self.mesh);
    }

    /// Get the height of the surface at a given point via interpolation of the
    /// triangles
    fn get(&self, pt: DVec2) -> DVec3 {
        self.get_triangle(pt).plane().get(pt)
    }

    /// Set the height of the surface at a given point
    ///
    /// This raises the entire triangle containing the point to the target Z
    /// value
    fn set(&mut self, pt: DVec3) {
        // Get the triangle containing `pt`
        let triangle = self.get_triangle(pt.xy());

        // Adjust all vertexes of the triangle
        for (_, idx) in triangle.0.iter() {
            self.data[*idx] = pt.z;
        }
    }

    /// Issue render commands for the surface
    fn draw(&mut self) {
        // Clear the mesh
        self.mesh.vertices.clear();
        self.mesh.indices.clear();

        // Internal grid layout:
        //
        // +y
        //   +---+---+
        //   |  /|  /|
        //   | / | / |
        //   |/  |/  |
        //   +---+---+
        // (0,0)      +x
        for y in 0..self.yres.strict_sub(1) {
            for x in 0..self.xres.strict_sub(1) {
                // Convenience bindings
                let vertices = &mut self.mesh.vertices;
                let indices  = &mut self.mesh.indices;

                // Save off the number of vertices
                let num_vert = vertices.len();

                for vertex in [
                    (0, 0), (1, 1), (0, 1), // Triangle 1
                    (0, 0), (1, 0), (1, 1), // Triangle 2
                ] {
                    // Compute vertex coords
                    let x = x + vertex.0;
                    let y = y + vertex.1;

                    // Compute height map index
                    let idx = y * self.xres + x;

                    // Get height map data
                    let z = self.data[idx];

                    // Compute scaled X and Y
                    let x = x as Float / self.xres.strict_sub(1) as Float *
                        self.width;
                    let y = y as Float / self.yres.strict_sub(1) as Float *
                        self.height;

                    // Record the vertex. No color needed we'll compute it with
                    // the normal afterwards
                    vertices.push(Vertex::new(
                        x as f32, y as f32, z as f32, 0., 0., BLACK));
                }

                // Compute the normals to update the colors
                for triangle in 0..2 {
                    // Get the coords
                    let p0 = vertices[num_vert + triangle * 3 + 0].position;
                    let p1 = vertices[num_vert + triangle * 3 + 1].position;
                    let p2 = vertices[num_vert + triangle * 3 + 2].position;

                    let normal = (p1 - p0).cross(p2 - p0).normalize();

                    // Rotate the normal a bit for better lighting contrast
                    let normal = Mat3::from_rotation_y(0.2) *
                        Mat3::from_rotation_x(0.2) * normal;

                    let color = Color::new(0., 0., normal.z, 1.);

                    // Update the colors
                    vertices[num_vert + triangle * 3 + 0].color = color.into();
                    vertices[num_vert + triangle * 3 + 1].color = color.into();
                    vertices[num_vert + triangle * 3 + 2].color = color.into();
                }

                // Record the vertices to use to create triangles
                for ii in 0..6 {
                    indices.push((num_vert + ii) as u16);
                }

                // Perform a draw call when we've accumulated enough data
                if vertices.len() >= 4096 {
                    draw_mesh(&self.mesh);
                    self.mesh.vertices.clear();
                    self.mesh.indices.clear();
                }
            }
        }

        // Flush remaining data if there is any
        if self.mesh.vertices.len() > 0 {
            draw_mesh(&self.mesh);
        }
    }
}

/// Construct window configuration
fn window_conf() -> Conf {
    Conf {
        window_title: "Window name".to_owned(),
        fullscreen:   false,
        sample_count: 8, // MSAA
        platform: miniquad::conf::Platform {
            // Set to None for vsync on, Some(0) for off
            swap_interval: None,
            //swap_interval: Some(0),
            ..Default::default()
        },
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    let mut surface = Surface::new(1., 1., 8, 16);

    rand::srand(7175);

    let mut rad: Float = 0.0;

    for _frame in 0u64.. {
        clear_background(DARKGRAY);

        set_camera(&Camera3D {
            position: vec3(0.5, -0.5, 0.8),
            up: vec3(0., 0., 1.),
            target: vec3(0.5, 0.5, 0.),
            ..Default::default()
        });

        if _frame % 60 == 0 {
            for point in surface.data.iter_mut() {
                *point = rand::gen_range(-0.05, 0.05);
            }
        }

        let pt = DVec2::from_angle(rad) / 3. + DVec2::new(rad.sin() / 10. + 0.5, rad.sin() / 10. + 0.5);
        let mut pt = surface.get(pt);
        pt.z = (rad * 4.).sin() / 20.;
        //surface.set(pt);

        surface.draw();

        surface.hat(&[
            (DVec2::new(0.47, 0.45), DVec2::new(0.80, 0.84)),
        ]);

        draw_sphere(surface.get(pt.xy()).as_vec3(), 0.01, None, GREEN);

        rad += 0.005;
        next_frame().await
    }
}

