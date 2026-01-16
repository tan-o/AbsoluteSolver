use anyhow::{Context, Result};
use opencv::{
    core::{Mat, Point, Rect, Scalar, Size},
    highgui, imgproc,
    prelude::*,
    videoio,
};
use std::fs::File;
use std::time::{Duration, Instant};

mod algorithm;
mod config;
mod hand;
mod head;
mod preprocess;

use config::Config;
use hand::HandPipeline;
use head::FacePipeline;

struct SolverApp {
    config: Config,
    camera: videoio::VideoCapture,
    hand_pipeline: HandPipeline,
    face_pipeline: FacePipeline,
    last_known_face: Option<(Rect, Instant)>,
    // 存储原始的高分辨率遮罩图
    raw_face_mask: Mat,
}

impl SolverApp {
    fn new(config_path: &str) -> Result<Self> {
        println!(">> [Init] 系统初始化中...");
        let file = File::open(config_path).context("无法打开配置文件")?;
        let config: Config = serde_yaml::from_reader(file).context("配置格式错误")?;
        Self::setup_window(&config)?;

        let mut camera = videoio::VideoCapture::new(config.camera.index, videoio::CAP_ANY)?;
        if !videoio::VideoCapture::is_opened(&camera)? {
            anyhow::bail!("无法打开摄像头 ID: {}", config.camera.index);
        }
        camera.set(videoio::CAP_PROP_FRAME_WIDTH, config.camera.width as f64)?;
        camera.set(videoio::CAP_PROP_FRAME_HEIGHT, config.camera.height as f64)?;
        camera.set(videoio::CAP_PROP_FPS, 60.0)?;

        println!(
            ">> [Init] 摄像头分辨率: {}x{}",
            config.camera.width, config.camera.height
        );
        let hand_pipeline = HandPipeline::new(config.algorithm.hand.clone())?;
        let face_pipeline = FacePipeline::new(config.algorithm.face.clone())?;

        // ==========================================
        // 加载原始遮罩图 (不缩放)
        // ==========================================
        println!(">> [Init] 加载资源: {}", config.assets.avatar);
        let raw_face_mask = preprocess::load_and_prepare_mask(&config.assets.avatar)?;

        Ok(SolverApp {
            config,
            camera,
            hand_pipeline,
            face_pipeline,
            last_known_face: None,
            raw_face_mask,
        })
    }

    fn setup_window(config: &Config) -> Result<()> {
        let win_w = (config.camera.width as f64 * config.window.scale) as i32;
        let win_h = (config.camera.height as f64 * config.window.scale) as i32;
        highgui::named_window(&config.window.title, highgui::WINDOW_NORMAL)?;
        highgui::resize_window(&config.window.title, win_w, win_h)?;
        Ok(())
    }

    fn run(&mut self) -> Result<()> {
        println!(">> [Run] 开始主循环...");
        let mut frame = Mat::default();
        let mut display_frame = Mat::default();
        let mut flipped_frame = Mat::default();
        // 用于存放每帧动态缩放后的遮罩
        let mut current_frame_mask = Mat::default();

        let win_w = (self.config.camera.width as f64 * self.config.window.scale) as i32;
        let win_h = (self.config.camera.height as f64 * self.config.window.scale) as i32;
        let gui_size = Size::new(win_w, win_h);

        let mut active_cooldown = 0;
        let cooldown_frames = self.config.performance.active_fps * 2;
        const FACE_MEMORY_MS: u128 = 500;
        let active_fps = self.config.performance.active_fps;
        let idle_fps = self.config.performance.idle_fps;

        let overlap_threshold = self.config.algorithm.hand.overlap.unwrap_or(0.5);
        // 获取 YAML 中的缩放比例
        let avatar_scale_factor = self.config.assets.scale;

        loop {
            let start_time = Instant::now();
            self.camera.read(&mut frame)?;
            if frame.size()?.width == 0 {
                continue;
            }

            // 1. 预处理
            let processed_frame = preprocess::auto_correct_exposure(&frame)?;

            // 2. AI 推理
            let hand_result = self.hand_pipeline.process(&processed_frame)?;
            let face_result = self.face_pipeline.process(&processed_frame)?;

            // 3. 逻辑整合
            let mut final_hand: Option<(Vec<[f32; 3]>, Rect)> = None;
            let mut final_face: Option<(Vec<[f32; 3]>, Rect)> = None;

            if let Some((landmarks, rect)) = face_result {
                final_face = Some((landmarks, rect));
                self.last_known_face = Some((rect, Instant::now()));
            }

            if let Some((landmarks, hand_rect)) = hand_result {
                let mut is_false_positive = false;
                if let Some((face_rect, last_seen)) = &self.last_known_face {
                    if last_seen.elapsed().as_millis() < FACE_MEMORY_MS {
                        let intersection = hand_rect & *face_rect;
                        let intersect_area = intersection.area();
                        let face_area = face_rect.area() as f32;

                        let coverage = if face_area > 0.0 {
                            intersect_area as f32 / face_area
                        } else {
                            0.0
                        };

                        let h_cx = hand_rect.x + hand_rect.width / 2;
                        let h_cy = hand_rect.y + hand_rect.height / 2;
                        let f_cx = face_rect.x + face_rect.width / 2;
                        let f_cy = face_rect.y + face_rect.height / 2;
                        let dist = (((h_cx - f_cx).pow(2) + (h_cy - f_cy).pow(2)) as f32).sqrt();

                        if coverage > overlap_threshold || dist < (face_rect.width as f32 * 0.5) {
                            is_false_positive = true;
                        }
                    }
                }
                if !is_false_positive {
                    final_hand = Some((landmarks, hand_rect));
                }
            }

            // 4. 绘制准备
            imgproc::resize(
                &processed_frame,
                &mut display_frame,
                gui_size,
                0.0,
                0.0,
                imgproc::INTER_LINEAR,
            )?;
            opencv::core::flip(&display_frame, &mut flipped_frame, 1)?;

            let win_scale = self.config.window.scale as f32;
            let mut detected_hand = false;
            let mut detected_face = false;

            // 绘制手
            if let Some((landmarks, rect)) = final_hand {
                detected_hand = true;
                active_cooldown = cooldown_frames;

                let x = (rect.x as f32 * win_scale) as i32;
                let y = (rect.y as f32 * win_scale) as i32;
                let w = (rect.width as f32 * win_scale) as i32;
                let h = (rect.height as f32 * win_scale) as i32;
                let mirror_x = win_w - (x + w);

                imgproc::rectangle(
                    &mut flipped_frame,
                    Rect::new(mirror_x, y, w, h),
                    Scalar::new(0.0, 255.0, 0.0, 0.0),
                    2,
                    imgproc::LINE_8,
                    0,
                )?;

                for p in landmarks {
                    let mut px = p[0] * win_scale;
                    let py = p[1] * win_scale;
                    px = win_w as f32 - px;
                    imgproc::circle(
                        &mut flipped_frame,
                        Point::new(px as i32, py as i32),
                        4,
                        Scalar::new(0.0, 0.0, 255.0, 0.0),
                        -1,
                        imgproc::LINE_8,
                        0,
                    )?;
                }
            }

            // 绘制脸 + 动态遮罩
            if let Some((landmarks, rect)) = final_face {
                detected_face = true;
                if !detected_hand {
                    active_cooldown = cooldown_frames;
                }

                if let (Some(nose), Some(leye), Some(reye)) =
                    (landmarks.get(1), landmarks.get(33), landmarks.get(263))
                {
                    // 1. 计算三个关键点的几何中心 (Global坐标)
                    let cx_global = (nose[0] + leye[0] + reye[0]) / 3.0;
                    let cy_global = (nose[1] + leye[1] + reye[1]) / 3.0;

                    // 2. 映射到 镜像后的 GUI 坐标
                    let cx_gui = win_w as f32 - (cx_global * win_scale);
                    let cy_gui = cy_global * win_scale;

                    // ==========================================
                    // 动态尺寸计算核心逻辑
                    // ==========================================
                    // 1. 获取检测框在 GUI 窗口中的实际像素大小
                    //    rect.width 是 Global(640) 下的尺寸，乘以 window.scale 才是当前显示的尺寸
                    let face_w_gui = rect.width as f32 * win_scale;
                    let face_h_gui = rect.height as f32 * win_scale;

                    // 2. 乘以 yaml 配置的缩放比例 (avatar_scale_factor)
                    let target_w = (face_w_gui * avatar_scale_factor) as i32;
                    let target_h = (face_h_gui * avatar_scale_factor) as i32;

                    // 3. 实时缩放图片 (仅当尺寸有效时)
                    if target_w > 0 && target_h > 0 {
                        imgproc::resize(
                            &self.raw_face_mask,
                            &mut current_frame_mask,
                            Size::new(target_w, target_h),
                            0.0,
                            0.0,
                            imgproc::INTER_LINEAR,
                        )?;

                        // 4. 计算左上角坐标 (中心点 - 图片的一半)
                        let top_left =
                            Point::new(cx_gui as i32 - target_w / 2, cy_gui as i32 - target_h / 2);

                        // 5. 叠加
                        preprocess::overlay_image(
                            &mut flipped_frame,
                            &current_frame_mask,
                            top_left,
                        )?;
                    }
                }

                // 绘制调试用的人脸框 (如果你不想看到蓝框，可以注释掉下面这一段)
                // let x = (rect.x as f32 * win_scale) as i32;
                // let y = (rect.y as f32 * win_scale) as i32;
                // let w = (rect.width as f32 * win_scale) as i32;
                // let h = (rect.height as f32 * win_scale) as i32;
                // let mirror_x = win_w - (x + w);

                // imgproc::rectangle(
                //     &mut flipped_frame,
                //     Rect::new(mirror_x, y, w, h),
                //     Scalar::new(255.0, 0.0, 0.0, 0.0),
                //     1,
                //     imgproc::LINE_8,
                //     0,
                // )?;
            }

            // UI
            let target_fps = if active_cooldown > 0 {
                active_cooldown -= 1;
                active_fps
            } else {
                idle_fps
            };
            let status_text = format!(
                "FPS: {} | HAND: {} | FACE: {}",
                target_fps, detected_hand, detected_face
            );
            let color = if detected_hand || detected_face {
                Scalar::new(0.0, 255.0, 0.0, 0.0)
            } else {
                Scalar::new(0.0, 255.0, 255.0, 0.0)
            };

            imgproc::put_text(
                &mut flipped_frame,
                &status_text,
                Point::new(10, 30),
                imgproc::FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                imgproc::LINE_8,
                false,
            )?;

            highgui::imshow(&self.config.window.title, &flipped_frame)?;

            let elapsed = start_time.elapsed();
            let frame_ms = 1000 / target_fps as u64;
            let target_dur = Duration::from_millis(frame_ms);

            if target_dur > elapsed {
                let wait = (target_dur - elapsed).as_millis() as i32;
                if highgui::wait_key(wait.max(1))? == 27 {
                    break;
                }
            } else {
                if highgui::wait_key(1)? == 27 {
                    break;
                }
            }
        }
        Ok(())
    }
}

fn main() -> Result<()> {
    let mut app = SolverApp::new("config.yaml")?;
    app.run()?;
    Ok(())
}
