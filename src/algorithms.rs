// src/algorithms.rs

use anyhow::{Context, Result};
use opencv::{
    core::{self, Mat, Point, Point2f, Scalar, Size, Vec3b, Vec4b}, // [修复] 移除了 RotatedRect
    imgcodecs,
    imgproc,
    prelude::*,
};

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
            filters.push([
                OneEuroFilter::new(0.05, 200.0),
                OneEuroFilter::new(0.05, 200.0),
                OneEuroFilter::new(0.05, 200.0),
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

// ==========================================
// 图像预处理
// ==========================================
pub struct PreprocessResult {
    pub img: Mat,
    pub scale: f32,
    pub pad_x: i32,
    pub pad_y: i32,
}

pub fn letterbox(src: &Mat, target_width: i32, target_height: i32) -> Result<PreprocessResult> {
    let src_w = src.cols();
    let src_h = src.rows();
    let scale = (target_width as f32 / src_w as f32).min(target_height as f32 / src_h as f32);
    let new_w = (src_w as f32 * scale).round() as i32;
    let new_h = (src_h as f32 * scale).round() as i32;

    let mut resized = Mat::default();
    imgproc::resize(
        src,
        &mut resized,
        Size::new(new_w, new_h),
        0.0,
        0.0,
        imgproc::INTER_LINEAR,
    )?;

    let dw = target_width - new_w;
    let dh = target_height - new_h;
    let top = dh / 2;
    let bottom = dh - top;
    let left = dw / 2;
    let right = dw - left;

    let mut dst = Mat::default();
    core::copy_make_border(
        &resized,
        &mut dst,
        top,
        bottom,
        left,
        right,
        core::BorderTypes::BORDER_CONSTANT as i32,
        Scalar::new(0.0, 0.0, 0.0, 0.0),
    )?;

    Ok(PreprocessResult {
        img: dst,
        scale,
        pad_x: left,
        pad_y: top,
    })
}

pub fn auto_correct_exposure(src: &Mat) -> Result<Mat> {
    let mut small_src = Mat::default();
    let small_size = Size::new(160, 120);
    imgproc::resize(
        src,
        &mut small_src,
        small_size,
        0.0,
        0.0,
        imgproc::INTER_LINEAR,
    )?;

    let mut gray = Mat::default();
    if small_src.channels() == 3 {
        imgproc::cvt_color(
            &small_src,
            &mut gray,
            imgproc::COLOR_BGR2GRAY,
            0,
            core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;
    } else {
        small_src.copy_to(&mut gray)?;
    }

    let mean_scalar = core::mean(&gray, &core::no_array())?;
    let mean_brightness = mean_scalar[0] as f32;

    if mean_brightness > 95.0 && mean_brightness < 165.0 {
        return Ok(src.clone());
    }

    let target_brightness: f32 = 130.0;
    let safe_mean = mean_brightness.max(1.0);
    let gamma = ((target_brightness / 255.0).ln() / (safe_mean / 255.0).ln()).clamp(0.4, 3.0);

    let mut lut_data = Vec::with_capacity(256);
    for i in 0..256 {
        let normalized = i as f32 / 255.0;
        let corrected = normalized.powf(gamma) * 255.0;
        lut_data.push(corrected.clamp(0.0, 255.0) as u8);
    }

    let lut_mat = Mat::from_slice(&lut_data)?;
    let lut_final = lut_mat.reshape(1, 256)?;
    let mut dst = Mat::default();
    core::lut(src, &lut_final, &mut dst)?;
    Ok(dst)
}

pub fn load_and_prepare_mask(path: &str) -> Result<Mat> {
    let raw_mask = imgcodecs::imread(path, imgcodecs::IMREAD_UNCHANGED)
        .with_context(|| format!("无法加载图片: {}", path))?;

    let mut final_mask = Mat::default();
    if raw_mask.channels() == 3 {
        imgproc::cvt_color(
            &raw_mask,
            &mut final_mask,
            imgproc::COLOR_BGR2BGRA,
            0,
            core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;
    } else if raw_mask.channels() == 4 {
        final_mask = raw_mask;
    } else {
        anyhow::bail!("不支持的通道数: {}", raw_mask.channels());
    }

    Ok(final_mask)
}

pub fn overlay_image(background: &mut Mat, foreground: &Mat, top_left: Point) -> Result<()> {
    let bg_w = background.cols();
    let bg_h = background.rows();
    let fg_w = foreground.cols();
    let fg_h = foreground.rows();

    let x = top_left.x;
    let y = top_left.y;

    if x + fg_w <= 0 || x >= bg_w || y + fg_h <= 0 || y >= bg_h {
        return Ok(());
    }

    let start_x = x.max(0);
    let start_y = y.max(0);
    let end_x = (x + fg_w).min(bg_w);
    let end_y = (y + fg_h).min(bg_h);
    let w = end_x - start_x;
    let h = end_y - start_y;

    if w <= 0 || h <= 0 {
        return Ok(());
    }

    let fg_off_x = start_x - x;
    let fg_off_y = start_y - y;

    for row in 0..h {
        for col in 0..w {
            let bg_y = start_y + row;
            let bg_x = start_x + col;
            let fg_y = fg_off_y + row;
            let fg_x = fg_off_x + col;

            let fg_pixel: Vec4b = *foreground.at_2d(fg_y, fg_x)?;
            let alpha = fg_pixel[3] as f32 / 255.0;

            if alpha < 0.01 {
                continue;
            }

            let bg_pixel_ptr = background.at_2d_mut::<Vec3b>(bg_y, bg_x)?;

            if alpha > 0.99 {
                bg_pixel_ptr[0] = fg_pixel[0];
                bg_pixel_ptr[1] = fg_pixel[1];
                bg_pixel_ptr[2] = fg_pixel[2];
            } else {
                let inv_alpha = 1.0 - alpha;
                bg_pixel_ptr[0] =
                    (fg_pixel[0] as f32 * alpha + bg_pixel_ptr[0] as f32 * inv_alpha) as u8;
                bg_pixel_ptr[1] =
                    (fg_pixel[1] as f32 * alpha + bg_pixel_ptr[1] as f32 * inv_alpha) as u8;
                bg_pixel_ptr[2] =
                    (fg_pixel[2] as f32 * alpha + bg_pixel_ptr[2] as f32 * inv_alpha) as u8;
            }
        }
    }
    Ok(())
}

// ==========================================
// 几何计算与图片变换算法 (优化版)
// ==========================================

/// 计算三角形 (p1, p2, p3) 的外接圆半径
pub fn calculate_circumradius(p1: [f32; 3], p2: [f32; 3], p3: [f32; 3]) -> f32 {
    let dist = |a: [f32; 3], b: [f32; 3]| -> f32 {
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
    };

    let a = dist(p2, p3);
    let b = dist(p1, p3);
    let c = dist(p1, p2);

    let s = (a + b + c) / 2.0;
    let area = (s * (s - a) * (s - b) * (s - c)).sqrt();

    if area < 1e-5 {
        return 0.0;
    }

    (a * b * c) / (4.0 * area)
}

/// 计算手部旋转角度 (基于 0 -> 9 向量)
pub fn calculate_hand_angle(wrist: Point, middle_mcp: Point) -> f64 {
    let dx = middle_mcp.x - wrist.x;
    let dy = middle_mcp.y - wrist.y;
    // 计算基础角度
    (dy as f64).atan2(dx as f64).to_degrees()
}

/// 旋转并绘制图片 (高画质版)
/// center: 图片在背景图上的中心坐标
/// target_size: 图片的目标显示大小 (宽/高)
/// angle: 旋转角度 (度)
pub fn draw_rotated_image(
    bg: &mut Mat,
    img: &Mat,
    center: Point,
    target_size: i32,
    angle: f64,
) -> Result<()> {
    if target_size <= 0 || img.empty() {
        return Ok(());
    }

    // 计算包含旋转后图像所需的最小矩形尺寸
    let diag = (target_size as f32 * std::f32::consts::SQRT_2).ceil() as i32;
    let canvas_size = diag + 2;

    // 一步到位的变换矩阵 (缩放 + 旋转 + 平移)
    let src_center = Point2f::new(img.cols() as f32 / 2.0, img.rows() as f32 / 2.0);
    let scale_factor = target_size as f64 / img.cols() as f64;

    // 获取基础旋转矩阵
    let mut rot_mat = imgproc::get_rotation_matrix_2d(src_center, angle, scale_factor)?;

    // 调整矩阵的平移部分，使其移动到新画布的中心
    let new_center_x = canvas_size as f64 / 2.0;
    let new_center_y = canvas_size as f64 / 2.0;

    // [修复] 使用 at_2d 读取矩阵元素
    // let tx = rot_mat.at_2d::<f64>(0, 2)?;
    // let ty = rot_mat.at_2d::<f64>(1, 2)?;

    // [修复] 使用 at_2d_mut 修改矩阵元素
    *rot_mat.at_2d_mut::<f64>(0, 2)? += new_center_x - src_center.x as f64; // tx + ...
    *rot_mat.at_2d_mut::<f64>(1, 2)? += new_center_y - src_center.y as f64; // ty + ...

    // 使用 INTER_CUBIC (三次插值) 进行变换
    let mut rotated_patch = Mat::default();
    imgproc::warp_affine(
        img,
        &mut rotated_patch,
        &rot_mat,
        Size::new(canvas_size, canvas_size),
        imgproc::INTER_CUBIC, // 高质量插值
        core::BORDER_CONSTANT,
        Scalar::default(),
    )?;

    // 计算贴图位置
    let top_left = Point::new(center.x - canvas_size / 2, center.y - canvas_size / 2);

    // 叠加
    overlay_image(bg, &rotated_patch, top_left)?;

    Ok(())
}
