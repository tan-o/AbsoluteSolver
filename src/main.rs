// src/main.rs
#![windows_subsystem = "windows"] // 保持隐藏控制台窗口

use anyhow::{Context, Result};
use opencv::{
    core::{Mat, Point, Rect, Scalar, Size},
    highgui, imgproc,
    prelude::*,
    videoio,
};
use std::fs::File;
use std::time::{Duration, Instant};

mod algorithms;
mod config;
mod detectors;
mod interaction;
mod shortcuts;

use config::Config;
use detectors::{FacePipeline, HandPipeline};
use interaction::{HandEvent, HandGestureController, HeadMouseController, HeadPoseSolver};
use shortcuts::{AppAction, InputManager};

struct SolverApp {
    config: Config,
    camera: videoio::VideoCapture,
    hand_pipeline: HandPipeline,
    face_pipeline: FacePipeline,
    mouse_controller: HeadMouseController,
    input_manager: InputManager,
    pose_solver: HeadPoseSolver,
    gesture_controller: HandGestureController,

    last_valid_angles: Option<(f64, f64)>,
    last_known_face_rect: Option<(Rect, Instant)>,
    last_valid_landmarks: Option<Vec<[f32; 3]>>,
    raw_face_mask: Mat,
}

impl SolverApp {
    fn new(config_path: &str) -> Result<Self> {
        // ========================================================
        // 动态调试窗口配置
        // ========================================================
        let file = File::open(config_path).context("无法打开配置文件")?;
        let config: Config = serde_yaml::from_reader(file).context("配置格式错误")?;

        if config.debug.show_debug_window {
            #[cfg(target_os = "windows")]
            unsafe {
                extern "system" {
                    fn AllocConsole() -> i32;
                }
                AllocConsole();
                println!(">> [Debug] 调试窗口已开启");
            }
        }
        println!(">> [Init] 系统初始化中...");

        Self::setup_window(&config)?;

        #[cfg(target_os = "windows")]
        let api_preference = videoio::CAP_MSMF;
        #[cfg(not(target_os = "windows"))]
        let api_preference = videoio::CAP_ANY;

        let mut camera = videoio::VideoCapture::new(config.camera.index, api_preference)?;

        if !videoio::VideoCapture::is_opened(&camera)? {
            anyhow::bail!("无法打开摄像头 ID: {}", config.camera.index);
        }
        camera.set(videoio::CAP_PROP_FRAME_WIDTH, config.camera.width as f64)?;
        camera.set(videoio::CAP_PROP_FRAME_HEIGHT, config.camera.height as f64)?;
        camera.set(videoio::CAP_PROP_FPS, 30.0)?;

        let hand_pipeline = HandPipeline::new(config.algorithm.hand.clone(), &config.inference)?;
        let face_pipeline = FacePipeline::new(config.algorithm.face.clone(), &config.inference)?;
        let mouse_controller = HeadMouseController::new(config.mouse.clone())?;
        let input_manager = InputManager::new(&config.shortcuts)?;
        let pose_solver = HeadPoseSolver::new(config.camera.width, config.camera.height)?;
        let gesture_controller = HandGestureController::new(config.gesture.clone());
        let raw_face_mask = algorithms::load_and_prepare_mask(&config.assets.avatar)?;

        Ok(SolverApp {
            config,
            camera,
            hand_pipeline,
            face_pipeline,
            mouse_controller,
            input_manager,
            pose_solver,
            gesture_controller,
            last_valid_angles: None,
            last_known_face_rect: None,
            last_valid_landmarks: None,
            raw_face_mask,
        })
    }

    fn setup_window(config: &Config) -> Result<()> {
        let win_w = (config.camera.width as f64 * config.window.scale) as i32;
        let win_h = (config.camera.height as f64 * config.window.scale) as i32;
        highgui::named_window(&config.window.title, highgui::WINDOW_NORMAL)?;
        highgui::resize_window(&config.window.title, win_w, win_h)?;

        if config.debug.window_always_on_top {
            Self::set_window_always_on_top(&config.window.title)?;
        }
        Ok(())
    }

    #[cfg(target_os = "windows")]
    fn set_window_always_on_top(window_title: &str) -> Result<()> {
        use std::ffi::OsStr;
        use std::os::windows::ffi::OsStrExt;

        let wide: Vec<u16> = OsStr::new(window_title)
            .encode_wide()
            .chain(std::iter::once(0))
            .collect();

        unsafe {
            extern "system" {
                fn FindWindowW(lpclass: *const u16, lpname: *const u16) -> *mut std::ffi::c_void;
                fn SetWindowPos(
                    hwnd: *mut std::ffi::c_void,
                    hwnd_insert_after: *mut std::ffi::c_void,
                    x: i32,
                    y: i32,
                    cx: i32,
                    cy: i32,
                    uflags: u32,
                ) -> i32;
            }
            const HWND_TOPMOST: *mut std::ffi::c_void = -1isize as *mut std::ffi::c_void;
            const SWP_NOMOVE: u32 = 0x0002;
            const SWP_NOSIZE: u32 = 0x0001;

            let hwnd = FindWindowW(std::ptr::null(), wide.as_ptr());
            if !hwnd.is_null() {
                SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE);
            }
        }
        Ok(())
    }

    #[cfg(not(target_os = "windows"))]
    fn set_window_always_on_top(_window_title: &str) -> Result<()> {
        Ok(())
    }

    fn run(&mut self) -> Result<()> {
        println!(">> [Run] 系统启动成功。快捷键: Ctrl+Alt+O (总开关), Ctrl+Alt+R (重置中心)");

        let mut frame = Mat::default();
        let mut display_frame = Mat::default();
        let mut flipped_frame = Mat::default();
        let mut current_frame_mask = Mat::default();

        let win_w = (self.config.camera.width as f64 * self.config.window.scale) as i32;
        let win_h = (self.config.camera.height as f64 * self.config.window.scale) as i32;
        let gui_size = Size::new(win_w, win_h);

        let active_fps = self.config.performance.active_fps as u64;
        let idle_fps = self.config.performance.idle_fps as u64;
        let mut active_cooldown = 0;
        let overlap_threshold = self.config.algorithm.hand.overlap.unwrap_or(0.5);

        let mut frame_count = 0u32;
        let mut fps_timer = Instant::now();
        let mut real_fps = 0.0f32;

        loop {
            let start_time = Instant::now();

            // 1. 读取摄像头
            match self.camera.read(&mut frame) {
                Ok(true) => {}
                Ok(false) => {
                    eprintln!(">> [Warn] 丢帧，尝试重连...");
                    if let Err(_) = self.reconnect_camera() {
                        std::thread::sleep(Duration::from_millis(500));
                        continue;
                    }
                }
                Err(e) => {
                    eprintln!(">> [Error] 摄像头错误: {}", e);
                    if let Err(_) = self.reconnect_camera() {
                        std::thread::sleep(Duration::from_millis(500));
                        continue;
                    }
                }
            }

            if frame.size()?.width == 0 {
                continue;
            }

            let processed_frame = algorithms::auto_correct_exposure(&frame)?;

            // 2. 推理检测
            let mut hand_result = if self.config.algorithm.hand.enabled.unwrap_or(true) {
                self.hand_pipeline.process(&processed_frame)?
            } else {
                None
            };

            let face_result = if self.config.algorithm.face.enabled.unwrap_or(true) {
                self.face_pipeline.process(&processed_frame)?
            } else {
                None
            };

            if hand_result.is_some() || face_result.is_some() {
                active_cooldown = 60;
            }

            // ==========================================
            // 核心逻辑
            // ==========================================

            // A. 手脸重叠剔除
            if let Some((_, face_rect)) = face_result {
                if let Some((_, ref mut hand_rect)) = hand_result {
                    let intersection = *hand_rect & face_rect;
                    if hand_rect.area() > 0
                        && (intersection.area() as f32 / hand_rect.area() as f32)
                            > overlap_threshold
                    {
                        hand_result = None;
                    }
                }
            }

            // ==========================================
            // B. 更新手势状态 (UI 和 点击控制都需要)
            // ==========================================
            let current_hand_event = if let Some((ref hand_landmarks, _)) = hand_result {
                self.gesture_controller.process(hand_landmarks)
            } else {
                // 如果手丢了，获取 reset_state 返回的事件 (PinchEnd)
                self.gesture_controller.reset_state()
            };

            // C. 核心追踪逻辑 (新功能: 支持头/手切换)
            let current_target_point = match self.config.mouse.track_mode {
                config::MouseTrackMode::Hand => hand_result
                    .as_ref()
                    .and_then(|(l, _)| self.gesture_controller.solve_palm_center(l)),
                config::MouseTrackMode::Head => face_result
                    .as_ref()
                    .and_then(|(l, _)| self.pose_solver.solve_centroid(l)),
            };

            if let Some((cx, cy)) = current_target_point {
                self.last_valid_angles = Some((cx, cy));
                if self.mouse_controller.enabled {
                    let _ = self.mouse_controller.update(cx, cy);
                }
            }

            // D. UI 数据缓存更新
            if let Some((ref landmarks, rect)) = face_result {
                self.last_valid_landmarks = Some(landmarks.clone());
                self.last_known_face_rect = Some((rect, Instant::now()));
            }

            // E. 鼠标交互
            if self.mouse_controller.enabled {
                match current_hand_event {
                    HandEvent::PinchStart => {
                        let _ = self.mouse_controller.click_down();
                    }
                    HandEvent::PinchEnd => {
                        let _ = self.mouse_controller.click_up();
                    }
                    HandEvent::RotateCW => {
                        let _ = self.mouse_controller.scroll(-2);
                    }
                    HandEvent::RotateCCW => {
                        let _ = self.mouse_controller.scroll(2);
                    }
                    _ => {}
                }
            }

            // 4. 快捷键
            match self.input_manager.check_action() {
                AppAction::ToggleMouse => self.mouse_controller.toggle(),
                AppAction::ResetAnchor => {
                    let reset_coord = match self.config.mouse.track_mode {
                        config::MouseTrackMode::Hand => hand_result
                            .as_ref()
                            .and_then(|(l, _)| self.gesture_controller.solve_palm_center(l)),
                        config::MouseTrackMode::Head => {
                            let l_opt = if let Some((ref l, _)) = face_result {
                                Some(l)
                            } else {
                                self.last_valid_landmarks.as_ref()
                            };
                            l_opt.and_then(|l| self.pose_solver.solve_centroid(l))
                        }
                    };
                    if let Some((cx, cy)) = reset_coord {
                        self.mouse_controller.reset_anchor(cx, cy);
                    }
                }
                _ => {}
            }

            // ==========================================
            // 渲染绘制
            // ==========================================
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

            // 1. 绘制手 (【关键修复】恢复了捏合变色和文字显示的逻辑)
            if let Some((landmarks, rect)) = hand_result {
                let x = (rect.x as f32 * win_scale) as i32;
                let y = (rect.y as f32 * win_scale) as i32;
                let mirror_x = win_w - (x + (rect.width as f32 * win_scale) as i32);

                // 【恢复】调用 is_pinched，消除警告，并恢复视觉反馈
                let is_pinched = self.gesture_controller.is_pinched();
                let color = if self.mouse_controller.enabled && is_pinched {
                    Scalar::new(0.0, 0.0, 255.0, 0.0) // 激活且捏合：红
                } else if self.mouse_controller.enabled {
                    Scalar::new(0.0, 255.0, 0.0, 0.0) // 激活未捏合：绿
                } else {
                    Scalar::new(100.0, 100.0, 100.0, 0.0) // 未激活：灰
                };

                imgproc::rectangle(
                    &mut flipped_frame,
                    Rect::new(
                        mirror_x,
                        y,
                        (rect.width as f32 * win_scale) as i32,
                        (rect.height as f32 * win_scale) as i32,
                    ),
                    color,
                    2,
                    8,
                    0,
                )?;
                for p in landmarks {
                    imgproc::circle(
                        &mut flipped_frame,
                        Point::new(
                            (win_w as f32 - p[0] * win_scale) as i32,
                            (p[1] * win_scale) as i32,
                        ),
                        2,
                        color,
                        -1,
                        8,
                        0,
                    )?;
                }

                // 【恢复】捏合时显示 "CLICK"
                if is_pinched {
                    imgproc::put_text(
                        &mut flipped_frame,
                        "CLICK",
                        Point::new(mirror_x, y - 5),
                        imgproc::FONT_HERSHEY_SIMPLEX,
                        0.8,
                        color,
                        2,
                        8,
                        false,
                    )?;
                }
            }

            // 2. 绘制脸部与面具
            if let Some((rect, last_time)) = self.last_known_face_rect {
                if last_time.elapsed().as_millis() < 500 {
                    if let Some(ref landmarks) = self.last_valid_landmarks {
                        for &idx in HeadPoseSolver::get_rigid_landmark_indices() {
                            if let Some(p) = landmarks.get(idx) {
                                imgproc::circle(
                                    &mut flipped_frame,
                                    Point::new(
                                        (win_w as f32 - p[0] * win_scale) as i32,
                                        (p[1] * win_scale) as i32,
                                    ),
                                    2,
                                    Scalar::new(0.0, 255.0, 255.0, 0.0),
                                    -1,
                                    8,
                                    0,
                                )?;
                            }
                        }
                        if self.config.assets.scale > 0.0 {
                            if let (Some(nose), Some(leye), Some(reye)) =
                                (landmarks.get(1), landmarks.get(33), landmarks.get(263))
                            {
                                let face_w =
                                    rect.width as f32 * win_scale * self.config.assets.scale;
                                let face_h =
                                    rect.height as f32 * win_scale * self.config.assets.scale;
                                imgproc::resize(
                                    &self.raw_face_mask,
                                    &mut current_frame_mask,
                                    Size::new(face_w as i32, face_h as i32),
                                    0.0,
                                    0.0,
                                    imgproc::INTER_LINEAR,
                                )?;
                                let top_left = Point::new(
                                    (win_w as f32
                                        - ((nose[0] + leye[0] + reye[0]) / 3.0) * win_scale)
                                        as i32
                                        - face_w as i32 / 2,
                                    ((nose[1] + leye[1] + reye[1]) / 3.0 * win_scale) as i32
                                        - face_h as i32 / 2,
                                );
                                let _ = algorithms::overlay_image(
                                    &mut flipped_frame,
                                    &current_frame_mask,
                                    top_left,
                                );
                            }
                        }
                    }
                }
            }

            // 3. UI 文本
            let status = format!(
                "FPS: {:.1} | Mode: {} | Sys: {}",
                real_fps,
                if self.config.mouse.track_mode == config::MouseTrackMode::Hand {
                    "HAND"
                } else {
                    "HEAD"
                },
                if self.mouse_controller.enabled {
                    "ON"
                } else {
                    "OFF"
                }
            );
            imgproc::put_text(
                &mut flipped_frame,
                &status,
                Point::new(10, 30),
                imgproc::FONT_HERSHEY_SIMPLEX,
                0.6,
                Scalar::new(0.0, 255.0, 0.0, 0.0),
                2,
                8,
                false,
            )?;

            // 【保留】设备信息显示 (避免 device_name 警告)
            let device_text = format!("Device: {}", self.hand_pipeline.device_name);
            let mut baseline = 0;
            let text_size = imgproc::get_text_size(
                &device_text,
                imgproc::FONT_HERSHEY_SIMPLEX,
                0.6,
                2,
                &mut baseline,
            )?;
            imgproc::put_text(
                &mut flipped_frame,
                &device_text,
                Point::new(win_w - text_size.width - 10, win_h - 10),
                imgproc::FONT_HERSHEY_SIMPLEX,
                0.6,
                Scalar::new(0.0, 255.0, 0.0, 0.0),
                2,
                8,
                false,
            )?;

            highgui::imshow(&self.config.window.title, &flipped_frame)?;

            let key = highgui::wait_key(1)?;
            if key == 27
                || highgui::get_window_property(
                    &self.config.window.title,
                    highgui::WND_PROP_AUTOSIZE,
                )
                .unwrap_or(-1.0)
                    == -1.0
            {
                break;
            }

            if active_cooldown > 0 {
                active_cooldown -= 1;
            }
            frame_count += 1;
            let elapsed_fps = fps_timer.elapsed();
            if elapsed_fps.as_millis() >= 1000 {
                real_fps = frame_count as f32 * 1000.0 / elapsed_fps.as_millis() as f32;
                frame_count = 0;
                fps_timer = Instant::now();
            }

            let frame_duration = Duration::from_millis(
                1000 / (if active_cooldown > 0 {
                    active_fps
                } else {
                    idle_fps
                }),
            );
            if frame_duration > start_time.elapsed() {
                std::thread::sleep(frame_duration - start_time.elapsed());
            }
        }
        Ok(())
    }

    fn reconnect_camera(&mut self) -> Result<()> {
        eprintln!(">> [Info] 正在断开并重新连接摄像头...");
        let _ = self.camera.release();
        std::thread::sleep(Duration::from_millis(1000));

        let max_retries = 3;
        for attempt in 1..=max_retries {
            eprintln!(">> [Info] 尝试重连... ({}/{})", attempt, max_retries);
            match videoio::VideoCapture::new(self.config.camera.index, videoio::CAP_ANY) {
                Ok(mut new_camera) => {
                    if let Ok(true) = videoio::VideoCapture::is_opened(&new_camera) {
                        let _ = new_camera.set(
                            videoio::CAP_PROP_FRAME_WIDTH,
                            self.config.camera.width as f64,
                        );
                        let _ = new_camera.set(
                            videoio::CAP_PROP_FRAME_HEIGHT,
                            self.config.camera.height as f64,
                        );
                        let _ = new_camera.set(videoio::CAP_PROP_FPS, 60.0);
                        self.camera = new_camera;
                        eprintln!(">> [Info] 摄像头重连成功！");
                        return Ok(());
                    }
                }
                Err(e) => eprintln!(">> [Warn] 重连失败: {}", e),
            }
            std::thread::sleep(Duration::from_millis(500));
        }
        anyhow::bail!("无法重新连接摄像头")
    }
}

fn main() -> Result<()> {
    // 捕获 Panic 并暂停，防止闪退
    let result = (|| -> Result<()> {
        let mut app = SolverApp::new("config.yaml")?;
        app.run()?;
        Ok(())
    })();

    if let Err(ref e) = result {
        #[cfg(target_os = "windows")]
        unsafe {
            extern "system" {
                fn AllocConsole() -> i32;
            }
            AllocConsole();
        }
        eprintln!("\n\n>> [FATAL ERROR] 程序崩溃: {:?}", e);
        eprintln!(">> 按回车退出...");
        let _ = std::io::stdin().read_line(&mut String::new());
    }
    result
}
