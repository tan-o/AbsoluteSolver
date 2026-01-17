#![windows_subsystem = "windows"] // 【新增】隐藏Windows控制台窗口

use anyhow::{Context, Result};
use opencv::{
    core::{Mat, Point, Rect, Scalar, Size},
    highgui, imgproc,
    prelude::*,
    videoio,
};
use std::fs::File;
use std::time::{Duration, Instant}; // 【恢复】Duration 用于 FPS 控制

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

        let hand_pipeline = HandPipeline::new(config.algorithm.hand.clone())?;
        let face_pipeline = FacePipeline::new(config.algorithm.face.clone())?;
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

        // 【新增】如果配置了置顶，设置窗口置顶
        if config.debug.window_always_on_top {
            Self::set_window_always_on_top(&config.window.title)?;
        }

        Ok(())
    }

    #[cfg(target_os = "windows")]
    fn set_window_always_on_top(window_title: &str) -> Result<()> {
        use std::ffi::OsStr;
        use std::os::windows::ffi::OsStrExt;

        // 转换标题为 Wide 字符串
        let wide: Vec<u16> = OsStr::new(window_title)
            .encode_wide()
            .chain(std::iter::once(0))
            .collect();

        unsafe {
            // 调用 Windows API SetWindowPos 将窗口设为置顶
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
        // 非 Windows 平台不实现置顶功能
        eprintln!(">> [Warn] 窗口置顶功能仅在 Windows 平台支持");
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

        // 【恢复】FPS 控制变量
        let active_fps = self.config.performance.active_fps as u64;
        let idle_fps = self.config.performance.idle_fps as u64;
        let mut active_cooldown = 0; // 交互冷却帧数
        let overlap_threshold = self.config.algorithm.hand.overlap.unwrap_or(0.5);

        // 【新增】实时FPS计算变量
        let mut frame_count = 0u32;
        let mut fps_timer = Instant::now();
        let mut real_fps = 0.0f32;

        loop {
            // 【恢复】计时开始
            let start_time = Instant::now();

            // 【新增】视频捕获错误处理和重连机制
            match self.camera.read(&mut frame) {
                Ok(true) => {
                    // 成功读取帧
                }
                Ok(false) => {
                    // read() 返回 false 但未报错，通常表示流结束或暂时无法读取
                    eprintln!(">> [Warn] 无法读取视频帧，尝试重新连接摄像头...");
                    match self.reconnect_camera() {
                        Ok(_) => {
                            eprintln!(">> [Info] 摄像头重新连接成功");
                            continue;
                        }
                        Err(e) => {
                            eprintln!(">> [Error] 摄像头重连失败: {}", e);
                            std::thread::sleep(Duration::from_millis(500));
                            continue;
                        }
                    }
                }
                Err(e) => {
                    // read() 返回错误（如 MSMF 的 Error: -2147418113）
                    eprintln!(
                        ">> [Error] 视频捕获出错（通常是摄像头断开或驱动问题）: {}",
                        e
                    );
                    match self.reconnect_camera() {
                        Ok(_) => {
                            eprintln!(">> [Info] 摄像头重新连接成功");
                            continue;
                        }
                        Err(e) => {
                            eprintln!(">> [Error] 摄像头重连失败: {}", e);
                            std::thread::sleep(Duration::from_millis(500));
                            continue;
                        }
                    }
                }
            }

            // 检查读取的帧是否有效
            if frame.size()?.width == 0 {
                eprintln!(">> [Warn] 读取的帧无效（大小为0），跳过本帧");
                continue;
            }

            let processed_frame = algorithms::auto_correct_exposure(&frame)?;

            // 【优化】根据配置决定是否进行检测（节省CPU）
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

            // ==========================================
            // 逻辑处理
            // ==========================================

            // 1. 如果检测到人或者手，激活高 FPS 模式，否则进入休眠省电模式
            if hand_result.is_some() || face_result.is_some() {
                active_cooldown = 60; // 保持 1秒 高性能
            }

            // 2. 总开关控制 (鼠标 & 手势)
            if self.mouse_controller.enabled {
                // 【恢复】手脸重叠剔除 (防止擦脸时误触)
                if let Some((_, face_rect)) = face_result {
                    if let Some((_, ref mut hand_rect)) = hand_result {
                        let intersection = *hand_rect & face_rect;
                        let intersect_area = intersection.area();
                        let hand_area = hand_rect.area();
                        if hand_area > 0 {
                            let ratio = intersect_area as f32 / hand_area as f32;
                            if ratio > overlap_threshold {
                                // 重叠过大，丢弃手部数据
                                hand_result = None;
                            }
                        }
                    }
                }

                // 3. 手势处理
                if let Some((ref hand_landmarks, _)) = hand_result {
                    match self.gesture_controller.process(hand_landmarks) {
                        HandEvent::PinchStart => {
                            let _ = self.mouse_controller.click_down();
                        }
                        HandEvent::PinchEnd => {
                            let _ = self.mouse_controller.click_up();
                        }
                        HandEvent::RotateCW => {
                            let _ = self.mouse_controller.scroll(-1);
                        }
                        HandEvent::RotateCCW => {
                            let _ = self.mouse_controller.scroll(1);
                        }
                        HandEvent::None => {}
                    }
                }

                // 4. 头部鼠标处理
                if let Some((ref landmarks, rect)) = face_result {
                    self.last_valid_landmarks = Some(landmarks.clone());
                    self.last_known_face_rect = Some((rect, Instant::now()));

                    if let Some((cx, cy)) = self.pose_solver.solve_centroid(landmarks) {
                        self.last_valid_angles = Some((cx, cy));
                        if let Err(e) = self.mouse_controller.update(cx, cy) {
                            eprintln!(">> [Error] {}", e);
                        }
                    }
                }
            }

            // 5. 快捷键
            match self.input_manager.check_action() {
                AppAction::ToggleMouse => self.mouse_controller.toggle(),
                AppAction::ResetAnchor => {
                    // 优先使用当前帧，否则使用缓存
                    let l_opt = if let Some((ref l, _)) = face_result {
                        Some(l)
                    } else {
                        self.last_valid_landmarks.as_ref()
                    };
                    if let Some(l) = l_opt {
                        if let Some((cx, cy)) = self.pose_solver.solve_centroid(l) {
                            self.mouse_controller.reset_anchor(cx, cy);
                        }
                    }
                }
                AppAction::None => {}
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

            // 绘制手
            if let Some((landmarks, rect)) = hand_result {
                // 【修复】使用 _landmarks 消除警告，或者这里我们确实要画出来
                let x = (rect.x as f32 * win_scale) as i32;
                let y = (rect.y as f32 * win_scale) as i32;
                let mirror_x = win_w - (x + (rect.width as f32 * win_scale) as i32);

                let is_active =
                    self.mouse_controller.enabled && self.gesture_controller.is_pinched();
                let color = if is_active {
                    Scalar::new(0.0, 0.0, 255.0, 0.0)
                } else if self.mouse_controller.enabled {
                    Scalar::new(0.0, 255.0, 0.0, 0.0)
                } else {
                    Scalar::new(100.0, 100.0, 100.0, 0.0)
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

                // 【修复】把骨架画出来，这样 landmarks 就被使用了，警告消除
                for p in landmarks {
                    let px = win_w as f32 - (p[0] * win_scale);
                    let py = p[1] * win_scale;
                    imgproc::circle(
                        &mut flipped_frame,
                        Point::new(px as i32, py as i32),
                        2,
                        color,
                        -1,
                        8,
                        0,
                    )?;
                }

                if is_active {
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

            // 绘制脸部遮罩
            if let Some((rect, _)) = self.last_known_face_rect {
                if self.last_known_face_rect.unwrap().1.elapsed().as_millis() < 500 {
                    if let Some(ref landmarks) = self.last_valid_landmarks {
                        if let (Some(nose), Some(leye), Some(reye)) =
                            (landmarks.get(1), landmarks.get(33), landmarks.get(263))
                        {
                            let cx_global = (nose[0] + leye[0] + reye[0]) / 3.0;
                            let cy_global = (nose[1] + leye[1] + reye[1]) / 3.0;
                            let cx_gui = win_w as f32 - (cx_global * win_scale);
                            let cy_gui = cy_global * win_scale;
                            let face_w = rect.width as f32 * win_scale * self.config.assets.scale;
                            let face_h = rect.height as f32 * win_scale * self.config.assets.scale;

                            if face_w > 0.0 && face_h > 0.0 {
                                imgproc::resize(
                                    &self.raw_face_mask,
                                    &mut current_frame_mask,
                                    Size::new(face_w as i32, face_h as i32),
                                    0.0,
                                    0.0,
                                    imgproc::INTER_LINEAR,
                                )?;
                                let top_left = Point::new(
                                    cx_gui as i32 - face_w as i32 / 2,
                                    cy_gui as i32 - face_h as i32 / 2,
                                );
                                algorithms::overlay_image(
                                    &mut flipped_frame,
                                    &current_frame_mask,
                                    top_left,
                                )?;
                            }
                        }
                    }
                }
            }

            // UI 信息
            let current_fps = if active_cooldown > 0 {
                active_fps
            } else {
                idle_fps
            };

            // 【修改】显示实时FPS而不是目标FPS
            let status = format!(
                "FPS: {:.1} | Sys: {}",
                real_fps,
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

            // 【新增】根据配置决定是否显示调试窗口
            // 【修复】总是显示摄像头窗口，show_debug_window仅控制调试信息显示
            highgui::imshow(&self.config.window.title, &flipped_frame)?;

            if highgui::wait_key(1)? == 27 {
                break;
            }

            // 【恢复】帧率控制逻辑
            if active_cooldown > 0 {
                active_cooldown -= 1;
            }

            // 【新增】实时FPS计算
            frame_count += 1;
            let elapsed_fps = fps_timer.elapsed();
            if elapsed_fps.as_millis() >= 1000 {
                real_fps = frame_count as f32 * 1000.0 / elapsed_fps.as_millis() as f32;
                frame_count = 0;
                fps_timer = Instant::now();
            }

            let elapsed = start_time.elapsed();
            let frame_duration = Duration::from_millis(1000 / current_fps);
            if frame_duration > elapsed {
                // 如果处理太快，睡一会，降低 CPU 占用
                std::thread::sleep(frame_duration - elapsed);
            }
        }
        Ok(())
    }

    /// 重新连接摄像头
    fn reconnect_camera(&mut self) -> Result<()> {
        eprintln!(">> [Info] 正在断开并重新连接摄像头...");

        // 关闭旧的摄像头连接
        let _ = self.camera.release();

        // 稍作延迟，让驱动有时间释放资源
        std::thread::sleep(Duration::from_millis(1000));

        // 尝试重新打开摄像头
        let max_retries = 3;
        for attempt in 1..=max_retries {
            eprintln!(
                ">> [Info] 尝试重新连接摄像头... (第 {}/{} 次)",
                attempt, max_retries
            );

            match videoio::VideoCapture::new(self.config.camera.index, videoio::CAP_ANY) {
                Ok(mut new_camera) => {
                    if let Ok(is_opened) = videoio::VideoCapture::is_opened(&new_camera) {
                        if is_opened {
                            // 设置摄像头属性
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
                            eprintln!(">> [Info] 摄像头成功重新连接！");
                            return Ok(());
                        }
                    }
                }
                Err(e) => {
                    eprintln!(">> [Warn] 第 {} 次重连尝试失败: {}", attempt, e);
                }
            }

            if attempt < max_retries {
                std::thread::sleep(Duration::from_millis(500));
            }
        }

        anyhow::bail!("经过多次尝试，无法重新连接摄像头")
    }
}

fn main() -> Result<()> {
    let mut app = SolverApp::new("config.yaml")?;
    app.run()?;
    Ok(())
}
