use anyhow::{Context, Result};
use ndarray::{Array4, ArrayView3};
use opencv::{
    core::{AlgorithmHint, Rect, Size, Vec3b},
    imgproc,
    prelude::*,
};
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::Tensor,
};

use crate::algorithm::{non_max_suppression, DetectionBox, HandLandmarks, LandmarkSmoother};
// 引入通用的预处理模块
use crate::config::AlgorithmParams;
use crate::preprocess;

pub struct HandPipeline {
    // Session
    detector_sess: Session,
    landmark_sess: Session,

    // Configs
    yolo_input_width: i32,
    yolo_input_height: i32,
    landmark_input_size: i32,

    // State
    avg_score: f32,
    stability: f32,
    threshold: f32,
    smoother: LandmarkSmoother,
}

impl HandPipeline {
    pub fn new(config: AlgorithmParams) -> Result<Self> {
        let detector_path = "hand/HandDetector/yolo11-hand-keypoint.onnx";
        let detector_sess = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(8)?
            .with_inter_threads(2)?
            .commit_from_file(detector_path)
            .context("无法加载 YOLO Hand Detector")?;

        let landmark_path = "hand/HandLandmarkDetector/hand_landmark_sparse_Nx3x224x224.onnx";
        let landmark_sess = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(8)?
            .with_inter_threads(2)?
            .commit_from_file(landmark_path)
            .context("无法加载 Hand Landmark Detector")?;

        Ok(Self {
            detector_sess,
            landmark_sess,
            yolo_input_height: 480,
            yolo_input_width: 640,
            landmark_input_size: 224,

            avg_score: 0.0,
            stability: config.stability.clamp(0.0, 1.0),
            threshold: config.threshold.clamp(0.0, 1.0),
            smoother: LandmarkSmoother::new(21),
        })
    }

    pub fn process(&mut self, full_frame: &Mat) -> Result<Option<(HandLandmarks, Rect)>> {
        // ==========================================
        // Step 1: YOLO 检测 (带 Letterbox)
        // ==========================================

        // 调用通用预处理
        let pre_res =
            preprocess::letterbox(full_frame, self.yolo_input_width, self.yolo_input_height)?;
        let letterboxed_img = pre_res.img;

        let mut rgb_frame = Mat::default();
        imgproc::cvt_color(
            &letterboxed_img,
            &mut rgb_frame,
            imgproc::COLOR_BGR2RGB,
            0,
            AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;

        let mut input_array = Array4::<f32>::zeros((
            1,
            3,
            self.yolo_input_height as usize,
            self.yolo_input_width as usize,
        ));

        for y in 0..self.yolo_input_height {
            for x in 0..self.yolo_input_width {
                let pixel = rgb_frame.at_2d::<Vec3b>(y, x)?;
                input_array[[0, 0, y as usize, x as usize]] = pixel[0] as f32 / 255.0;
                input_array[[0, 1, y as usize, x as usize]] = pixel[1] as f32 / 255.0;
                input_array[[0, 2, y as usize, x as usize]] = pixel[2] as f32 / 255.0;
            }
        }

        let outputs = self
            .detector_sess
            .run(ort::inputs![Tensor::from_array(input_array)?])?;

        let (output_shape, raw_data) = outputs
            .get("output0")
            .context("找不到 YOLO output0")?
            .try_extract_tensor::<f32>()?;

        let output_view = ArrayView3::from_shape(
            (
                output_shape[0] as usize,
                output_shape[1] as usize,
                output_shape[2] as usize,
            ),
            raw_data,
        )
        .context("无法构建 Tensor 视图")?;

        let num_anchors = output_shape[2] as usize;
        let mut boxes = Vec::new();

        // 提取候选框 并 立即映射回全局坐标
        for i in 0..num_anchors {
            let score = output_view[[0, 4, i]];

            if score > 0.4 {
                // 读取 Letterbox 坐标
                let cx_lb = output_view[[0, 0, i]];
                let cy_lb = output_view[[0, 1, i]];
                let w_lb = output_view[[0, 2, i]];
                let h_lb = output_view[[0, 3, i]];

                // 【关键】映射回 Global 坐标
                let cx_global = (cx_lb - pre_res.pad_x as f32) / pre_res.scale;
                let cy_global = (cy_lb - pre_res.pad_y as f32) / pre_res.scale;
                let w_global = w_lb / pre_res.scale;
                let h_global = h_lb / pre_res.scale;

                let x1 = cx_global - w_global / 2.0;
                let y1 = cy_global - h_global / 2.0;
                let x2 = cx_global + w_global / 2.0;
                let y2 = cy_global + h_global / 2.0;

                boxes.push(DetectionBox {
                    x1,
                    y1,
                    x2,
                    y2,
                    score,
                });
            }
        }

        let best_boxes = non_max_suppression(boxes, 0.5);

        if best_boxes.is_empty() {
            self.smoother.reset();
            return Ok(None);
        }

        let best_box = best_boxes[0];

        // 平滑分数
        let alpha_fall = 1.0 - (0.95 * self.stability);
        let alpha_rise = 1.0 - (0.8 * self.stability);
        if best_box.score > self.avg_score {
            self.avg_score = self.avg_score * (1.0 - alpha_rise) + best_box.score * alpha_rise;
        } else {
            self.avg_score = self.avg_score * (1.0 - alpha_fall) + best_box.score * alpha_fall;
        }

        if self.avg_score < self.threshold {
            self.smoother.reset();
            return Ok(None);
        }

        // ==========================================
        // Step 2: 准备 ROI (基于全局坐标)
        // ==========================================

        let img_w = full_frame.cols() as f32;
        let img_h = full_frame.rows() as f32;

        let pad_scale = 1.0;
        let box_w = best_box.x2 - best_box.x1;
        let box_h = best_box.y2 - best_box.y1;
        let cx = best_box.x1 + box_w / 2.0;
        let cy = best_box.y1 + box_h / 2.0;

        let side_len = box_w.max(box_h) * pad_scale;

        let x1 = (cx - side_len / 2.0) as i32;
        let y1 = (cy - side_len / 2.0) as i32;
        let roi_w = side_len as i32;
        let roi_h = side_len as i32;

        let safe_x1 = x1.clamp(0, img_w as i32);
        let safe_y1 = y1.clamp(0, img_h as i32);
        let safe_x2 = (x1 + roi_w).clamp(0, img_w as i32);
        let safe_y2 = (y1 + roi_h).clamp(0, img_h as i32);

        let final_w = safe_x2 - safe_x1;
        let final_h = safe_y2 - safe_y1;

        if final_w <= 10 || final_h <= 10 {
            return Ok(None);
        }

        let roi_rect = Rect::new(safe_x1, safe_y1, final_w, final_h);

        // ==========================================
        // Step 3: Landmark 回归 (224x224)
        // ==========================================

        let hand_roi = Mat::roi(full_frame, roi_rect)?;
        let mut landmark_input = Mat::default();

        imgproc::resize(
            &hand_roi,
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

        let (_lm_shape, lm_data) = lm_outputs
            .get("xyz_x21")
            .or_else(|| lm_outputs.get("Identity"))
            .context("找不到 Landmark 输出 (xyz_x21)")?
            .try_extract_tensor::<f32>()?;

        // ==========================================
        // Step 4: 坐标映射 (224 -> ROI -> Global)
        // ==========================================
        let mut landmarks = Vec::with_capacity(21);

        for i in 0..21 {
            let lx = lm_data[i * 3];
            let ly = lm_data[i * 3 + 1];
            let lz = lm_data[i * 3 + 2];

            // 归一化 (假设 Sparse 模型输出的是 0~224 的像素值)
            let normalized_x = lx / self.landmark_input_size as f32;
            let normalized_y = ly / self.landmark_input_size as f32;

            let global_x = roi_rect.x as f32 + normalized_x * roi_rect.width as f32;
            let global_y = roi_rect.y as f32 + normalized_y * roi_rect.height as f32;

            landmarks.push([global_x, global_y, lz]);
        }

        let smoothed_landmarks = self.smoother.smooth(&landmarks);

        Ok(Some((smoothed_landmarks, roi_rect)))
    }
}
