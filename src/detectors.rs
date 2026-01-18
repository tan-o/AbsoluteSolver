// src/detectors.rs

use crate::algorithms::{
    generate_face_anchors, letterbox, non_max_suppression, Anchor, DetectionBox, FaceLandmarks,
    HandLandmarks, LandmarkSmoother,
};
use crate::config::{AlgorithmParams, InferenceConfig};
use anyhow::{Context, Result};
use ndarray::Array4;
use opencv::{
    core::{AlgorithmHint, Rect, Size}, // 确保引入了 AlgorithmHint
    imgproc,
    prelude::*,
};
use std::fs;

// 【新增】引入 Windows DXGI 库
use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory1, IDXGIFactory1};

use ort::{
    execution_providers::{CUDAExecutionProvider, DirectMLExecutionProvider},
    session::{builder::GraphOptimizationLevel, Session}, // 移除了私有的 SessionBuilder
    value::Tensor,
};

// ==========================================
// 1. 智能设备寻址模块 (纯 Rust DXGI 版)
// ==========================================

/// 获取核显 (iGPU) 的 DirectML 设备 ID
/// 原理：使用 Windows DXGI API 遍历显卡，顺序与 DirectML 严格一致
fn get_igpu_id() -> i32 {
    println!(">> [Auto-Detect] 正在扫描 DXGI 适配器以寻找核显 (iGPU)...");

    // 1. 加载关键词
    let keywords = match fs::read_to_string("igpu_list.txt") {
        Ok(content) => {
            println!(">> [Info] 已加载外部核显关键词文件: igpu_list.txt");
            content
                .lines()
                .map(|s| s.trim().to_lowercase())
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
        }
        Err(_) => {
            vec![
                "intel".to_string(),
                "uhd".to_string(),
                "iris".to_string(),
                "xe".to_string(),
                "radeon graphics".to_string(),
            ]
        }
    };

    // 2. 创建 DXGI 工厂
    // unsafe 是必须的，因为我们在调用 Windows C++ API
    let factory: Result<IDXGIFactory1, _> = unsafe { CreateDXGIFactory1() };

    match factory {
        Ok(factory) => {
            let mut i = 0;
            // 遍历所有显卡适配器
            while let Ok(adapter) = unsafe { factory.EnumAdapters1(i) } {
                // 获取显卡描述信息
                if let Ok(desc) = unsafe { adapter.GetDesc1() } {
                    // desc.Description 是 [u16; 128]，需要转成 String
                    let name = String::from_utf16_lossy(&desc.Description);
                    // 去除末尾的空字符
                    let name = name.trim_matches(char::from(0)).trim();

                    println!("   [DXGI ID: {}] {}", i, name);

                    let name_lower = name.to_lowercase();
                    // 3. 匹配关键词
                    for keyword in &keywords {
                        if name_lower.contains(keyword) {
                            println!(
                                ">> [Match] 成功匹配核显: '{}' (包含关键词 '{}')",
                                name, keyword
                            );
                            println!(">> [Result] 将使用 DirectML 设备 ID: {}", i);
                            return i as i32;
                        }
                    }
                }
                i += 1;
            }
            println!(">> [Warn] 未在列表中找到符合关键词的核显，将回退到默认 ID 0");
        }
        Err(e) => {
            println!(">> [Error] 无法创建 DXGI 工厂: {:?}", e);
        }
    }

    0 // 默认返回 0
}

// ==========================================
// 2. 核心：创建 Session
// ==========================================
fn create_session(path: &str, config: &InferenceConfig) -> Result<(Session, String)> {
    let make_builder = || {
        Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(config.cpu_threads)
            .context("Failed to create SessionBuilder")
    };

    let mode = config.device.to_lowercase();

    // -----------------------------------------------------------
    // 策略 A: CUDA
    // -----------------------------------------------------------
    if mode == "cuda" || mode == "auto" {
        println!(">> [Init] 尝试加载 CUDA (模式: {}) ...", mode);
        let ep = CUDAExecutionProvider::default().build().error_on_failure();

        let try_load = || -> Result<Session> {
            let builder = make_builder()?.with_execution_providers([ep])?;
            Ok(builder.commit_from_file(path)?)
        };

        if let Ok(session) = try_load() {
            println!(">> [Success] ✅ CUDA 加载成功");
            return Ok((session, "CUDA".to_string()));
        } else if mode == "cuda" {
            eprintln!(">> [Warn] ⚠️ CUDA 加载失败，将尝试回退...");
        }
    }

    // -----------------------------------------------------------
    // 策略 B: iGPU (智能 DXGI 寻址)
    // -----------------------------------------------------------
    if mode == "igpu" {
        let igpu_id = get_igpu_id(); // 这里现在调用的是上面的纯 Rust 版本
        let ep = DirectMLExecutionProvider::default()
            .with_device_id(igpu_id)
            .build()
            .error_on_failure();

        let try_load = || -> Result<Session> {
            let builder = make_builder()?.with_execution_providers([ep])?;
            Ok(builder.commit_from_file(path)?)
        };

        if let Ok(session) = try_load() {
            let dev_name = format!("iGPU-{}", igpu_id);
            println!(">> [Success] ✅ {} 加载成功", dev_name);
            return Ok((session, dev_name));
        } else {
            eprintln!(">> [Warn] ⚠️ 指定的核显加载失败，准备回退 CPU");
        }
    }

    // -----------------------------------------------------------
    // 策略 C: GPU (默认 DirectML)
    // -----------------------------------------------------------
    if mode == "gpu" || mode == "auto" {
        println!(">> [Init] 尝试加载 DirectML (默认设备) ...");
        let ep = DirectMLExecutionProvider::default()
            .build()
            .error_on_failure();

        let try_load = || -> Result<Session> {
            let builder = make_builder()?.with_execution_providers([ep])?;
            Ok(builder.commit_from_file(path)?)
        };

        if let Ok(session) = try_load() {
            println!(">> [Success] ✅ DirectML (GPU) 加载成功");
            return Ok((session, "DirectML (GPU)".to_string()));
        }
    }

    // -----------------------------------------------------------
    // 策略 D: CPU
    // -----------------------------------------------------------
    println!(">> [Info] 使用 CPU 加载模型: {}", path);
    let session = make_builder()?
        .commit_from_file(path)
        .context(format!("CPU 模式也无法加载模型: {}", path))?;

    Ok((session, "CPU".to_string()))
}
// ==========================================
// HandPipeline (手部检测)
// ==========================================
pub struct HandPipeline {
    detector_sess: Session,
    landmark_sess: Session,
    yolo_input_width: i32,
    yolo_input_height: i32,
    landmark_input_size: i32,
    avg_score: f32,
    stability: f32,
    threshold: f32,
    smoother: LandmarkSmoother,
    pub device_name: String,
}

impl HandPipeline {
    pub fn new(algo_config: AlgorithmParams, infer_config: &InferenceConfig) -> Result<Self> {
        let detector_path = "hand/HandDetector/yolo11-hand-keypoint.onnx";
        let landmark_path = "hand/HandLandmarkDetector/hand_landmark_sparse_Nx3x224x224.onnx";

        let (detector_sess, dev_name) = create_session(detector_path, infer_config)?;
        let (landmark_sess, _) = create_session(landmark_path, infer_config)?;

        Ok(Self {
            detector_sess,
            landmark_sess,
            yolo_input_width: 640,
            yolo_input_height: 480,
            landmark_input_size: 224,
            avg_score: 0.0,
            stability: algo_config.stability.clamp(0.0, 1.0),
            threshold: algo_config.threshold.clamp(0.0, 1.0),
            smoother: LandmarkSmoother::new(21),
            device_name: dev_name,
        })
    }

    pub fn process(&mut self, full_frame: &Mat) -> Result<Option<(HandLandmarks, Rect)>> {
        // Step 1: Preprocess (Letterbox)
        let pre_res = letterbox(full_frame, self.yolo_input_width, self.yolo_input_height)?;

        let mut rgb_frame = Mat::default();
        // 【修复】API 变更：OpenCV 0.90+ cvt_color 最后一个参数必须是 AlgorithmHint
        imgproc::cvt_color(
            &pre_res.img,
            &mut rgb_frame,
            imgproc::COLOR_BGR2RGB,
            0,
            AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;

        // Step 2: Prepare Input Tensor
        let width = self.yolo_input_width as usize;
        let height = self.yolo_input_height as usize;
        let total_pixels = width * height;
        let mut input_vec = vec![0f32; 1 * 3 * total_pixels];
        let (r_plane, rest) = input_vec.split_at_mut(total_pixels);
        let (g_plane, b_plane) = rest.split_at_mut(total_pixels);
        let bytes = rgb_frame.data_bytes()?;

        // 【说明】这个循环虽然看起来“原始”，但它是 HWC -> CHW 转换最高效的方式
        // 任何基于矩阵的 reshape/transpose 都会涉及额外的内存分配或多次遍历
        for (i, chunk) in bytes.chunks_exact(3).enumerate() {
            r_plane[i] = chunk[0] as f32 / 255.0;
            g_plane[i] = chunk[1] as f32 / 255.0;
            b_plane[i] = chunk[2] as f32 / 255.0;
        }

        let input_tensor =
            Tensor::from_array(Array4::from_shape_vec((1, 3, height, width), input_vec)?)?;

        // Step 3: Inference
        let outputs = self.detector_sess.run(ort::inputs![input_tensor])?;

        let (output_shape, raw_data) = outputs
            .get("output0")
            .context("Missing YOLO output0")?
            .try_extract_tensor::<f32>()?;

        // Step 4: Parse YOLO Output
        let output_view = ndarray::ArrayView3::from_shape(
            (
                output_shape[0] as usize,
                output_shape[1] as usize,
                output_shape[2] as usize,
            ),
            raw_data,
        )?;

        let num_anchors = output_shape[2] as usize;
        let mut boxes = Vec::new();

        // 这里的循环可以用迭代器简化，但性能差异微乎其微
        for i in 0..num_anchors {
            let score = output_view[[0, 4, i]];
            if score > 0.4 {
                let cx = output_view[[0, 0, i]];
                let cy = output_view[[0, 1, i]];
                let w = output_view[[0, 2, i]];
                let h = output_view[[0, 3, i]];

                // Coordinate Mapping
                let cx_global = (cx - pre_res.pad_x as f32) / pre_res.scale;
                let cy_global = (cy - pre_res.pad_y as f32) / pre_res.scale;
                let w_global = w / pre_res.scale;
                let h_global = h / pre_res.scale;

                boxes.push(DetectionBox {
                    x1: cx_global - w_global / 2.0,
                    y1: cy_global - h_global / 2.0,
                    x2: cx_global + w_global / 2.0,
                    y2: cy_global + h_global / 2.0,
                    score,
                });
            }
        }

        let best_boxes = non_max_suppression(boxes, 0.5);
        if best_boxes.is_empty() {
            self.smoother.reset();
            return Ok(None);
        }

        // Score Smoothing
        let best_box = best_boxes[0];
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

        // Step 5: Crop & Run Landmark
        let pad_scale = 1.0;
        let box_w = best_box.x2 - best_box.x1;
        let box_h = best_box.y2 - best_box.y1;
        let cx = best_box.x1 + box_w / 2.0;
        let cy = best_box.y1 + box_h / 2.0;
        let side_len = box_w.max(box_h) * pad_scale;

        let img_w = full_frame.cols() as f32;
        let img_h = full_frame.rows() as f32;

        let x1 = (cx - side_len / 2.0) as i32;
        let y1 = (cy - side_len / 2.0) as i32;
        let roi_s = side_len as i32;

        let safe_x1 = x1.clamp(0, img_w as i32);
        let safe_y1 = y1.clamp(0, img_h as i32);
        let safe_x2 = (x1 + roi_s).clamp(0, img_w as i32);
        let safe_y2 = (y1 + roi_s).clamp(0, img_h as i32);

        if (safe_x2 - safe_x1) <= 10 || (safe_y2 - safe_y1) <= 10 {
            return Ok(None);
        }

        let roi_rect = Rect::new(safe_x1, safe_y1, safe_x2 - safe_x1, safe_y2 - safe_y1);
        let hand_roi = Mat::roi(full_frame, roi_rect)?;

        let mut lm_input = Mat::default();
        imgproc::resize(
            &hand_roi,
            &mut lm_input,
            Size::new(self.landmark_input_size, self.landmark_input_size),
            0.0,
            0.0,
            imgproc::INTER_LINEAR,
        )?;

        let mut lm_rgb = Mat::default();
        // 【修复】API 变更
        imgproc::cvt_color(
            &lm_input,
            &mut lm_rgb,
            imgproc::COLOR_BGR2RGB,
            0,
            AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;

        let lm_size = self.landmark_input_size as usize;
        let lm_total = lm_size * lm_size;
        let mut lm_vec = vec![0f32; 1 * 3 * lm_total];
        let (lm_r, lm_rest) = lm_vec.split_at_mut(lm_total);
        let (lm_g, lm_b) = lm_rest.split_at_mut(lm_total);
        let lm_bytes = lm_rgb.data_bytes()?;

        for (i, chunk) in lm_bytes.chunks_exact(3).enumerate() {
            lm_r[i] = chunk[0] as f32 / 255.0;
            lm_g[i] = chunk[1] as f32 / 255.0;
            lm_b[i] = chunk[2] as f32 / 255.0;
        }

        let lm_outputs =
            self.landmark_sess
                .run(ort::inputs![Tensor::from_array(Array4::from_shape_vec(
                    (1, 3, lm_size, lm_size),
                    lm_vec
                )?)?])?;

        let lm_data = lm_outputs
            .get("xyz_x21")
            .or_else(|| lm_outputs.get("Identity"))
            .context("Missing Landmark output")?
            .try_extract_tensor::<f32>()?
            .1;

        let mut landmarks = Vec::with_capacity(21);
        for i in 0..21 {
            let lx = lm_data[i * 3];
            let ly = lm_data[i * 3 + 1];
            let lz = lm_data[i * 3 + 2];
            let gx =
                roi_rect.x as f32 + (lx / self.landmark_input_size as f32) * roi_rect.width as f32;
            let gy =
                roi_rect.y as f32 + (ly / self.landmark_input_size as f32) * roi_rect.height as f32;
            landmarks.push([gx, gy, lz]);
        }

        Ok(Some((self.smoother.smooth(&landmarks), roi_rect)))
    }
}

// ==========================================
// FacePipeline (人脸检测)
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
    pub fn new(algo_config: AlgorithmParams, infer_config: &InferenceConfig) -> Result<Self> {
        let detector_path = "head/FaceDetector/FaceDetector.onnx";
        let landmark_path = "head/FaceLandmarkDetector/FaceLandmarkDetector.onnx";

        let (detector_sess, _) = create_session(detector_path, infer_config)?;
        let (landmark_sess, _) = create_session(landmark_path, infer_config)?;

        let detector_input_size = 256;
        let anchors = generate_face_anchors(detector_input_size);

        Ok(Self {
            detector_sess,
            landmark_sess,
            detector_input_size,
            landmark_input_size: 192,
            avg_score: 0.0,
            stability: algo_config.stability.clamp(0.0, 1.0),
            threshold: algo_config.threshold.clamp(0.0, 1.0),
            anchors,
            smoother: LandmarkSmoother::new(468),
        })
    }

    pub fn process(&mut self, full_frame: &Mat) -> Result<Option<(FaceLandmarks, Rect)>> {
        let input_size = self.detector_input_size;

        // Step 1: Preprocess (Letterbox)
        let pre_res = letterbox(full_frame, input_size, input_size)?;

        let mut rgb_frame = Mat::default();
        // 【修复】API 变更
        imgproc::cvt_color(
            &pre_res.img,
            &mut rgb_frame,
            imgproc::COLOR_BGR2RGB,
            0,
            AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;

        let size_usize = input_size as usize;
        let total_pixels = size_usize * size_usize;
        let mut input_vec = vec![0f32; 1 * 3 * total_pixels];
        let (r, rest) = input_vec.split_at_mut(total_pixels);
        let (g, b) = rest.split_at_mut(total_pixels);
        let bytes = rgb_frame.data_bytes()?;

        // Normalization: (v - 127.5) / 127.5
        for (i, chunk) in bytes.chunks_exact(3).enumerate() {
            r[i] = (chunk[0] as f32 - 127.5) / 127.5;
            g[i] = (chunk[1] as f32 - 127.5) / 127.5;
            b[i] = (chunk[2] as f32 - 127.5) / 127.5;
        }

        let input_tensor = Tensor::from_array(Array4::from_shape_vec(
            (1, 3, size_usize, size_usize),
            input_vec,
        )?)?;

        // Step 2: Inference
        let outputs = self.detector_sess.run(ort::inputs![input_tensor])?;

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

        let current_prob = 1.0 / (1.0 + (-max_score).exp()); // Sigmoid

        // Smoothing
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

        // Decode Box
        let (raw_coords, anchor) = if best_layer == 1 {
            let start = max_idx * 16;
            (
                [
                    coords_1[start],
                    coords_1[start + 1],
                    coords_1[start + 2],
                    coords_1[start + 3],
                ],
                self.anchors[max_idx],
            )
        } else {
            let start = max_idx * 16;
            (
                [
                    coords_2[start],
                    coords_2[start + 1],
                    coords_2[start + 2],
                    coords_2[start + 3],
                ],
                self.anchors[512 + max_idx],
            )
        };

        let cx_norm;
        let cy_norm;
        let w_norm;
        let h_norm;
        if raw_coords[2].abs() > 5.0 {
            // Raw format
            w_norm = raw_coords[2] / input_size as f32;
            h_norm = raw_coords[3] / input_size as f32;
            cx_norm = anchor.x_center + (raw_coords[0] / input_size as f32);
            cy_norm = anchor.y_center + (raw_coords[1] / input_size as f32);
        } else {
            // Encoded format
            cx_norm = anchor.x_center + raw_coords[0] * anchor.w;
            cy_norm = anchor.y_center + raw_coords[1] * anchor.h;
            w_norm = raw_coords[2].clamp(-5.0, 5.0).exp() * anchor.w;
            h_norm = raw_coords[3].clamp(-5.0, 5.0).exp() * anchor.h;
        }

        // Global coords
        let cx_global = (cx_norm * input_size as f32 - pre_res.pad_x as f32) / pre_res.scale;
        let cy_global = (cy_norm * input_size as f32 - pre_res.pad_y as f32) / pre_res.scale;
        let w_global = (w_norm * input_size as f32) / pre_res.scale;
        let h_global = (h_norm * input_size as f32) / pre_res.scale;

        let scale_roi = 1.5;
        let box_size = w_global.max(h_global) * scale_roi;

        let img_w = full_frame.cols() as f32;
        let img_h = full_frame.rows() as f32;

        let x1 = (cx_global - box_size / 2.0) as i32;
        let y1 = (cy_global - box_size / 2.0) as i32;
        let roi_s = box_size as i32;

        let safe_x1 = x1.clamp(0, img_w as i32);
        let safe_y1 = y1.clamp(0, img_h as i32);
        let safe_x2 = (x1 + roi_s).clamp(0, img_w as i32);
        let safe_y2 = (y1 + roi_s).clamp(0, img_h as i32);

        if (safe_x2 - safe_x1) <= 0 || (safe_y2 - safe_y1) <= 0 {
            return Ok(None);
        }
        let roi_rect = Rect::new(safe_x1, safe_y1, safe_x2 - safe_x1, safe_y2 - safe_y1);

        // Step 4: Landmarks
        let face_roi = Mat::roi(full_frame, roi_rect)?;
        let mut lm_input = Mat::default();
        imgproc::resize(
            &face_roi,
            &mut lm_input,
            Size::new(self.landmark_input_size, self.landmark_input_size),
            0.0,
            0.0,
            imgproc::INTER_LINEAR,
        )?;

        let mut lm_rgb = Mat::default();
        // 【修复】API 变更
        imgproc::cvt_color(
            &lm_input,
            &mut lm_rgb,
            imgproc::COLOR_BGR2RGB,
            0,
            AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;

        let lm_size = self.landmark_input_size as usize;
        let lm_total = lm_size * lm_size;
        let mut lm_vec = vec![0f32; 1 * 3 * lm_total];
        let (lm_r, rest) = lm_vec.split_at_mut(lm_total);
        let (lm_g, lm_b) = rest.split_at_mut(lm_total);
        let bytes = lm_rgb.data_bytes()?;

        for (i, chunk) in bytes.chunks_exact(3).enumerate() {
            lm_r[i] = chunk[0] as f32 / 255.0;
            lm_g[i] = chunk[1] as f32 / 255.0;
            lm_b[i] = chunk[2] as f32 / 255.0;
        }

        let lm_outputs =
            self.landmark_sess
                .run(ort::inputs![Tensor::from_array(Array4::from_shape_vec(
                    (1, 3, lm_size, lm_size),
                    lm_vec
                )?)?])?;

        let lm_data = lm_outputs
            .get("landmarks")
            .or_else(|| lm_outputs.get("Identity"))
            .unwrap()
            .try_extract_tensor::<f32>()?
            .1;

        let mut landmarks = Vec::with_capacity(468);
        for i in 0..468 {
            let lx = lm_data[i * 3];
            let ly = lm_data[i * 3 + 1];
            let lz = lm_data[i * 3 + 2];
            let gx = roi_rect.x as f32 + lx * roi_rect.width as f32;
            let gy = roi_rect.y as f32 + ly * roi_rect.height as f32;
            landmarks.push([gx, gy, lz]);
        }

        Ok(Some((self.smoother.smooth(&landmarks), roi_rect)))
    }
}
