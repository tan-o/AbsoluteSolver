use anyhow::{Context, Result};
use ndarray::Array4;
use opencv::{
    core::{AlgorithmHint, Rect, Size, Vec3b},
    imgproc,
    prelude::*,
};
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::Tensor,
};

use crate::algorithm::{generate_face_anchors, Anchor, FaceLandmarks, LandmarkSmoother};
// 引入通用的预处理模块
use crate::config::AlgorithmParams;
use crate::preprocess;

pub struct FacePipeline {
    detector_sess: Session,
    landmark_sess: Session,
    detector_input_size: i32,
    landmark_input_size: i32,
    avg_score: f32,
    stability: f32,
    threshold: f32,
    anchors: Vec<Anchor>,
    smoother: LandmarkSmoother,
}

impl FacePipeline {
    pub fn new(config: AlgorithmParams) -> Result<Self> {
        let detector_path = "head/FaceDetector/FaceDetector.onnx";
        let landmark_path = "head/FaceLandmarkDetector/FaceLandmarkDetector.onnx";

        // 这里建议也加上 DirectML 支持 (如果你的 preprocess 是为了 GPU 准备的)
        // 但为了保持代码简洁，这里先保持默认配置，你可以按需添加 ExecutionProvider
        let detector_sess = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(detector_path)
            .context("无法加载 FaceDetector")?;

        let landmark_sess = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(landmark_path)
            .context("无法加载 FaceLandmarkDetector")?;

        let detector_input_size = 256;
        let landmark_input_size = 192;
        let anchors = generate_face_anchors(detector_input_size);

        Ok(Self {
            detector_sess,
            landmark_sess,
            detector_input_size,
            landmark_input_size,
            avg_score: 0.0,
            stability: config.stability.clamp(0.0, 1.0),
            threshold: config.threshold.clamp(0.0, 1.0),
            anchors,
            smoother: LandmarkSmoother::new(468),
        })
    }

    pub fn process(&mut self, full_frame: &Mat) -> Result<Option<(FaceLandmarks, Rect)>> {
        let input_size = self.detector_input_size;

        // ==========================================
        // Step 1: 预处理 (Letterbox)
        // ==========================================
        // 直接调用通用模块，自动处理缩放和黑边，防止拉伸
        let pre_res = preprocess::letterbox(full_frame, input_size, input_size)?;
        let letterboxed_img = pre_res.img;

        let mut rgb_frame = Mat::default();
        imgproc::cvt_color(
            &letterboxed_img,
            &mut rgb_frame,
            imgproc::COLOR_BGR2RGB,
            0,
            AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;

        let mut input_array =
            Array4::<f32>::zeros((1, 3, input_size as usize, input_size as usize));
        for y in 0..input_size {
            for x in 0..input_size {
                let pixel = rgb_frame.at_2d::<Vec3b>(y, x)?;
                // 归一化 [-1, 1]
                input_array[[0, 0, y as usize, x as usize]] = (pixel[0] as f32 - 127.5) / 127.5;
                input_array[[0, 1, y as usize, x as usize]] = (pixel[1] as f32 - 127.5) / 127.5;
                input_array[[0, 2, y as usize, x as usize]] = (pixel[2] as f32 - 127.5) / 127.5;
            }
        }

        // ==========================================
        // Step 2: 推理
        // ==========================================
        let outputs = self
            .detector_sess
            .run(ort::inputs![Tensor::from_array(input_array)?])?;

        let scores_1 = outputs
            .get("box_scores_1")
            .unwrap()
            .try_extract_tensor::<f32>()?
            .1;
        let scores_2 = outputs
            .get("box_scores_2")
            .unwrap()
            .try_extract_tensor::<f32>()?
            .1;
        let coords_1 = outputs
            .get("box_coords_1")
            .unwrap()
            .try_extract_tensor::<f32>()?
            .1;
        let coords_2 = outputs
            .get("box_coords_2")
            .unwrap()
            .try_extract_tensor::<f32>()?
            .1;

        let mut max_score = -1000.0f32;
        let mut max_idx = 0;
        let mut best_layer = 0;

        for (i, &score) in scores_1.iter().enumerate() {
            if score > max_score {
                max_score = score;
                max_idx = i;
                best_layer = 1;
            }
        }
        for (i, &score) in scores_2.iter().enumerate() {
            if score > max_score {
                max_score = score;
                max_idx = i;
                best_layer = 2;
            }
        }

        let current_prob = 1.0 / (1.0 + (-max_score).exp());

        // 平滑分数
        let alpha_fall = 1.0 - (0.95 * self.stability);
        let alpha_rise = 1.0 - (0.8 * self.stability);
        if current_prob > self.avg_score {
            self.avg_score = self.avg_score * (1.0 - alpha_rise) + current_prob * alpha_rise;
        } else {
            self.avg_score = self.avg_score * (1.0 - alpha_fall) + current_prob * alpha_fall;
        }

        if self.avg_score < self.threshold {
            self.smoother.reset();
            return Ok(None);
        }

        let (raw_coords, anchor) = if best_layer == 1 {
            let start = max_idx * 16;
            let c = &coords_1;
            (
                [c[start], c[start + 1], c[start + 2], c[start + 3]],
                self.anchors[max_idx],
            )
        } else {
            let start = max_idx * 16;
            let c = &coords_2;
            (
                [c[start], c[start + 1], c[start + 2], c[start + 3]],
                self.anchors[512 + max_idx],
            )
        };

        // 计算 Letterbox 坐标系下的归一化值
        let cx_norm: f32;
        let cy_norm: f32;
        let w_norm: f32;
        let h_norm: f32;

        if raw_coords[2].abs() > 5.0 {
            w_norm = raw_coords[2] / input_size as f32;
            h_norm = raw_coords[3] / input_size as f32;
            let dx = raw_coords[0] / input_size as f32;
            let dy = raw_coords[1] / input_size as f32;
            cx_norm = anchor.x_center + dx;
            cy_norm = anchor.y_center + dy;
        } else {
            let dw = raw_coords[2].clamp(-5.0, 5.0);
            let dh = raw_coords[3].clamp(-5.0, 5.0);
            cx_norm = anchor.x_center + raw_coords[0] * anchor.w;
            cy_norm = anchor.y_center + raw_coords[1] * anchor.h;
            w_norm = dw.exp() * anchor.w;
            h_norm = dh.exp() * anchor.h;
        }

        // ==========================================
        // Step 3: 坐标反算 (核心)
        // ==========================================
        // 1. 转为 Letterbox 图像上的像素坐标
        let cx_lb = cx_norm * input_size as f32;
        let cy_lb = cy_norm * input_size as f32;
        let w_lb = w_norm * input_size as f32;
        let h_lb = h_norm * input_size as f32;

        // 2. 利用 preprocess 返回的参数映射回原图 (去黑边 -> 反缩放)
        let cx_global = (cx_lb - pre_res.pad_x as f32) / pre_res.scale;
        let cy_global = (cy_lb - pre_res.pad_y as f32) / pre_res.scale;
        let w_global = w_lb / pre_res.scale;
        let h_global = h_lb / pre_res.scale;

        // 3. 计算 ROI (强制正方形)
        let img_w = full_frame.cols() as f32;
        let img_h = full_frame.rows() as f32;

        // 1.5倍扩充，稍微多看一点背景
        let scale_roi = 1.0;
        let box_size_px = w_global.max(h_global) * scale_roi;

        let x1 = (cx_global - box_size_px / 2.0) as i32;
        let y1 = (cy_global - box_size_px / 2.0) as i32;
        let roi_w = box_size_px as i32;
        let roi_h = box_size_px as i32;

        let safe_x1 = x1.clamp(0, img_w as i32);
        let safe_y1 = y1.clamp(0, img_h as i32);
        let safe_x2 = (x1 + roi_w).clamp(0, img_w as i32);
        let safe_y2 = (y1 + roi_h).clamp(0, img_h as i32);

        let final_w = safe_x2 - safe_x1;
        let final_h = safe_y2 - safe_y1;

        if final_w <= 0 || final_h <= 0 {
            self.smoother.reset();
            return Ok(None);
        }
        let roi_rect = Rect::new(safe_x1, safe_y1, final_w, final_h);

        // ==========================================
        // Step 4: 关键点检测 (Landmark)
        // ==========================================
        let face_roi = Mat::roi(full_frame, roi_rect)?;
        let mut landmark_input = Mat::default();
        imgproc::resize(
            &face_roi,
            &mut landmark_input,
            Size::new(self.landmark_input_size, self.landmark_input_size),
            0.0,
            0.0,
            imgproc::INTER_LINEAR,
        )?;

        let mut lm_rgb = Mat::default();
        imgproc::cvt_color(
            &landmark_input,
            &mut lm_rgb,
            imgproc::COLOR_BGR2RGB,
            0,
            AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;
        let mut lm_array = Array4::<f32>::zeros((
            1,
            3,
            self.landmark_input_size as usize,
            self.landmark_input_size as usize,
        ));
        for y in 0..self.landmark_input_size {
            for x in 0..self.landmark_input_size {
                let pixel = lm_rgb.at_2d::<Vec3b>(y, x)?;
                lm_array[[0, 0, y as usize, x as usize]] = pixel[0] as f32 / 255.0;
                lm_array[[0, 1, y as usize, x as usize]] = pixel[1] as f32 / 255.0;
                lm_array[[0, 2, y as usize, x as usize]] = pixel[2] as f32 / 255.0;
            }
        }

        let lm_outputs = self
            .landmark_sess
            .run(ort::inputs![Tensor::from_array(lm_array)?])?;
        let lm_tensor = lm_outputs
            .get("landmarks")
            .or_else(|| lm_outputs.get("Identity"))
            .context("找不到 Face Landmarks 输出")?
            .try_extract_tensor::<f32>()?
            .1;

        let mut landmarks = Vec::new();
        for i in 0..468 {
            let lx = lm_tensor[i * 3];
            let ly = lm_tensor[i * 3 + 1];
            // 简单映射：因为 ROI 已经是切出来的正方形，直接乘宽高加偏移
            let global_x = roi_rect.x as f32 + lx * roi_rect.width as f32;
            let global_y = roi_rect.y as f32 + ly * roi_rect.height as f32;
            landmarks.push([global_x, global_y, lm_tensor[i * 3 + 2]]);
        }

        let smoothed_landmarks = self.smoother.smooth(&landmarks);

        Ok(Some((smoothed_landmarks, roi_rect)))
    }
}
