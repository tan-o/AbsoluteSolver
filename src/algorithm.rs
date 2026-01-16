// ==========================================
// 公共类型定义
// ==========================================
pub type HandLandmarks = Vec<[f32; 3]>;
pub type FaceLandmarks = Vec<[f32; 3]>;

#[derive(Debug, Clone, Copy)]
pub struct Anchor {
    pub x_center: f32,
    pub y_center: f32,
    pub w: f32,
    pub h: f32,
}

// ==========================================
// DetectionBox: 用于 NMS 处理
// ==========================================
#[derive(Debug, Clone, Copy)]
pub struct DetectionBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub score: f32,
}

impl DetectionBox {
    pub fn area(&self) -> f32 {
        (self.x2 - self.x1).max(0.0) * (self.y2 - self.y1).max(0.0)
    }

    // 计算 Intersection over Union
    pub fn iou(&self, other: &DetectionBox) -> f32 {
        let inter_x1 = self.x1.max(other.x1);
        let inter_y1 = self.y1.max(other.y1);
        let inter_x2 = self.x2.min(other.x2);
        let inter_y2 = self.y2.min(other.y2);

        let inter_area = (inter_x2 - inter_x1).max(0.0) * (inter_y2 - inter_y1).max(0.0);
        let union_area = self.area() + other.area() - inter_area;

        if union_area == 0.0 {
            0.0
        } else {
            inter_area / union_area
        }
    }
}

// 简单的 NMS 实现
pub fn non_max_suppression(mut boxes: Vec<DetectionBox>, iou_threshold: f32) -> Vec<DetectionBox> {
    boxes.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

    let mut keepers = Vec::new();
    let mut is_suppressed = vec![false; boxes.len()];

    for i in 0..boxes.len() {
        if is_suppressed[i] {
            continue;
        }
        keepers.push(boxes[i]);

        for j in (i + 1)..boxes.len() {
            if !is_suppressed[j] {
                if boxes[i].iou(&boxes[j]) > iou_threshold {
                    is_suppressed[j] = true;
                }
            }
        }
    }
    keepers
}

// ==========================================
// OneEuroFilter 实现
// ==========================================
pub struct OneEuroFilter {
    min_cutoff: f32,
    beta: f32,
    d_cutoff: f32,
    x_prev: f32,
    dx_prev: f32,
    t_prev: f64,
}

impl OneEuroFilter {
    pub fn new(min_cutoff: f32, beta: f32) -> Self {
        Self {
            min_cutoff,
            beta,
            d_cutoff: 1.0,
            x_prev: 0.0,
            dx_prev: 0.0,
            t_prev: 0.0,
        }
    }

    pub fn filter(&mut self, t: f64, x: f32) -> f32 {
        if self.t_prev == 0.0 {
            self.x_prev = x;
            self.t_prev = t;
            return x;
        }
        let dt = (t - self.t_prev) as f32;
        if dt <= 0.0 {
            return self.x_prev;
        }

        let alpha_d = self.smoothing_factor(dt, self.d_cutoff);
        let dx = (x - self.x_prev) / dt;
        let dx_hat = self.exponential_smoothing(alpha_d, dx, self.dx_prev);

        let cutoff = self.min_cutoff + self.beta * dx_hat.abs();
        let alpha = self.smoothing_factor(dt, cutoff);
        let x_hat = self.exponential_smoothing(alpha, x, self.x_prev);

        self.x_prev = x_hat;
        self.dx_prev = dx_hat;
        self.t_prev = t;
        x_hat
    }

    fn smoothing_factor(&self, dt: f32, cutoff: f32) -> f32 {
        let r = 2.0 * std::f32::consts::PI * cutoff * dt;
        r / (r + 1.0)
    }

    fn exponential_smoothing(&self, alpha: f32, x: f32, x_prev: f32) -> f32 {
        alpha * x + (1.0 - alpha) * x_prev
    }
}

pub struct LandmarkSmoother {
    filters: Vec<[OneEuroFilter; 3]>,
    start_time: std::time::Instant,
}

impl LandmarkSmoother {
    pub fn new(num_landmarks: usize) -> Self {
        let mut filters = Vec::with_capacity(num_landmarks);
        for _ in 0..num_landmarks {
            // 参数调优: min_cutoff 越小越稳，beta 越大越跟手
            // 0.01, 100.0 是一个比较平衡的值
            filters.push([
                OneEuroFilter::new(0.01, 100.0),
                OneEuroFilter::new(0.01, 100.0),
                OneEuroFilter::new(0.01, 100.0),
            ]);
        }
        Self {
            filters,
            start_time: std::time::Instant::now(),
        }
    }

    pub fn smooth(&mut self, landmarks: &Vec<[f32; 3]>) -> Vec<[f32; 3]> {
        let t = self.start_time.elapsed().as_secs_f64();
        let mut smoothed = Vec::with_capacity(landmarks.len());
        for (i, point) in landmarks.iter().enumerate() {
            if i >= self.filters.len() {
                break;
            }
            let f = &mut self.filters[i];
            smoothed.push([
                f[0].filter(t, point[0]),
                f[1].filter(t, point[1]),
                f[2].filter(t, point[2]),
            ]);
        }
        smoothed
    }

    pub fn reset(&mut self) {
        self.start_time = std::time::Instant::now();
        for f in &mut self.filters {
            for axis in f {
                axis.t_prev = 0.0;
            }
        }
    }
}

// // ==========================================
// // Anchor 生成函数
// // ==========================================
// pub fn generate_hand_anchors(input_size: i32) -> Vec<Anchor> {
//     let mut anchors = Vec::new();
//     let strides = [8, 16, 32, 32];
//     for (_layer_id, &stride) in strides.iter().enumerate() {
//         let grid_rows = (input_size + stride - 1) / stride;
//         let grid_cols = (input_size + stride - 1) / stride;
//         for y in 0..grid_rows {
//             for x in 0..grid_cols {
//                 for _ in 0..2 {
//                     let x_center = (x as f32 + 0.5) / grid_cols as f32;
//                     let y_center = (y as f32 + 0.5) / grid_rows as f32;
//                     anchors.push(Anchor {
//                         x_center,
//                         y_center,
//                         w: 1.0,
//                         h: 1.0,
//                     });
//                 }
//             }
//         }
//     }
//     if anchors.len() > 2944 {
//         anchors.truncate(2944);
//     }
//     while anchors.len() < 2944 {
//         anchors.push(Anchor {
//             x_center: 0.5,
//             y_center: 0.5,
//             w: 1.0,
//             h: 1.0,
//         });
//     }
//     anchors
// }

pub fn generate_face_anchors(input_size: i32) -> Vec<Anchor> {
    let mut anchors = Vec::new();
    let strides = [16, 32];
    for &stride in strides.iter() {
        let grid_rows = (input_size + stride - 1) / stride;
        let grid_cols = (input_size + stride - 1) / stride;
        let anchors_num = if stride == 16 { 2 } else { 6 };
        for y in 0..grid_rows {
            for x in 0..grid_cols {
                for _ in 0..anchors_num {
                    let x_center = (x as f32 + 0.5) / grid_cols as f32;
                    let y_center = (y as f32 + 0.5) / grid_rows as f32;
                    anchors.push(Anchor {
                        x_center,
                        y_center,
                        w: 1.0,
                        h: 1.0,
                    });
                }
            }
        }
    }
    anchors
}
