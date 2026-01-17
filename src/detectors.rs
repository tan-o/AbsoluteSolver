// ==========================================
// 特征检测模块 - 整合手部和人脸检测
// ==========================================

use anyhow::{Context, Result};
use ndarray::Array4;
use opencv::{
    core::{AlgorithmHint, Rect, Size},
    imgproc,
    prelude::*,
};
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::Tensor,
};

use crate::algorithms::{
    generate_face_anchors, letterbox, non_max_suppression, Anchor, DetectionBox, FaceLandmarks,
    HandLandmarks, LandmarkSmoother,
};
use crate::config::AlgorithmParams;

// ==========================================
// 手部检测管道
// ==========================================
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
        let pre_res = letterbox(full_frame, self.yolo_input_width, self.yolo_input_height)?;
        let letterboxed_img = pre_res.img;

        let mut rgb_frame = Mat::default();
        imgproc::cvt_color(
            &letterboxed_img,
            &mut rgb_frame,
            imgproc::COLOR_BGR2RGB,
            0,
            AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;

        // 【优化】将 RGB Mat 转换为归一化的 f32 数组，使用更高效的按行遍历
        let mut input_array = Array4::<f32>::zeros((
            1,
            3,
            self.yolo_input_height as usize,
            self.yolo_input_width as usize,
        ));

        // 使用 Mat 数据指针而不是逐像素访问（快 10 倍以上）
        let bytes = rgb_frame.data_bytes()?;
        let width = self.yolo_input_width as usize;
        let height = self.yolo_input_height as usize;
        let total_pixels = width * height;

        // RGB 格式，所以 3 字节为一个像素
        for idx in 0..total_pixels {
            let r = bytes[idx * 3] as f32 / 255.0;
            let g = bytes[idx * 3 + 1] as f32 / 255.0;
            let b = bytes[idx * 3 + 2] as f32 / 255.0;

            let y = idx / width;
            let x = idx % width;
            input_array[[0, 0, y, x]] = r;
            input_array[[0, 1, y, x]] = g;
            input_array[[0, 2, y, x]] = b;
        }

        let outputs = self
            .detector_sess
            .run(ort::inputs![Tensor::from_array(input_array)?])?;

        let (output_shape, raw_data) = outputs
            .get("output0")
            .context("找不到 YOLO output0")?
            .try_extract_tensor::<f32>()?;

        let output_view = ndarray::ArrayView3::from_shape(
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

        // 【优化】使用数据指针而不是逐像素访问
        let lm_bytes = lm_rgb.data_bytes()?;
        let lm_total_pixels =
            (self.landmark_input_size as usize) * (self.landmark_input_size as usize);
        let lm_size = self.landmark_input_size as usize;

        for idx in 0..lm_total_pixels {
            let y = idx / lm_size;
            let x = idx % lm_size;
            lm_array[[0, 0, y, x]] = lm_bytes[idx * 3] as f32 / 255.0;
            lm_array[[0, 1, y, x]] = lm_bytes[idx * 3 + 1] as f32 / 255.0;
            lm_array[[0, 2, y, x]] = lm_bytes[idx * 3 + 2] as f32 / 255.0;
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

// ==========================================
// 人脸检测管道
// ==========================================
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

        let detector_sess = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(8)?
            .with_inter_threads(2)?
            .commit_from_file(detector_path)
            .context("无法加载 FaceDetector")?;

        let landmark_sess = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(8)?
            .with_inter_threads(2)?
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
        let pre_res = letterbox(full_frame, input_size, input_size)?;
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

        // 【优化】使用数据指针而不是逐像素访问（快 10 倍以上）
        let bytes = rgb_frame.data_bytes()?;
        let total_pixels = (input_size as usize) * (input_size as usize);
        let size = input_size as usize;

        for idx in 0..total_pixels {
            // RGB 格式，3 字节一个像素
            let r = (bytes[idx * 3] as f32 - 127.5) / 127.5;
            let g = (bytes[idx * 3 + 1] as f32 - 127.5) / 127.5;
            let b = (bytes[idx * 3 + 2] as f32 - 127.5) / 127.5;

            let y = idx / size;
            let x = idx % size;
            input_array[[0, 0, y, x]] = r;
            input_array[[0, 1, y, x]] = g;
            input_array[[0, 2, y, x]] = b;
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

        // 【优化】使用数据指针而不是逐像素访问
        let lm_bytes = lm_rgb.data_bytes()?;
        let lm_total_pixels =
            (self.landmark_input_size as usize) * (self.landmark_input_size as usize);
        let lm_size = self.landmark_input_size as usize;

        for idx in 0..lm_total_pixels {
            let y = idx / lm_size;
            let x = idx % lm_size;
            lm_array[[0, 0, y, x]] = lm_bytes[idx * 3] as f32 / 255.0;
            lm_array[[0, 1, y, x]] = lm_bytes[idx * 3 + 1] as f32 / 255.0;
            lm_array[[0, 2, y, x]] = lm_bytes[idx * 3 + 2] as f32 / 255.0;
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
