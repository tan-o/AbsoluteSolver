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
// 引入必要的线程同步库
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::Arc;

mod algorithms;
mod config;
mod detectors;
mod interaction;
mod overlay;
mod shortcuts;

use config::Config;
use detectors::{FacePipeline, HandPipeline};

// 合并了所有 interaction 模块的引用
use interaction::{
    HandEvent, HandGestureController, HeadMouseController, HeadPoseSolver, SharedCursorCoords,
};

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
    mouse_state: Arc<AtomicI32>,
    // 持有共享坐标引用
    // shared_coords: Arc<SharedCursorCoords>,
}

// 【新增】引入 TrayIconEvent 用于监听左键点击
use tray_icon::{
    menu::{Menu, MenuEvent, MenuItem},
    Icon,
    MouseButton, // 识别鼠标按键类型
    TrayIconBuilder,
    TrayIconEvent,
};

// 【新增】引入 Windows 消息处理所需的底层 API
use windows::Win32::UI::WindowsAndMessaging::{
    DispatchMessageW, PeekMessageW, TranslateMessage, MSG, PM_REMOVE,
};

// ...

// 辅助函数：利用 OpenCV 读取图片并转换为托盘图标需要的格式
fn load_tray_icon(path: &str) -> Result<Icon> {
    // 1. 读取图片
    let mat = opencv::imgcodecs::imread(path, opencv::imgcodecs::IMREAD_UNCHANGED)?;

    // 2. 调整大小 (托盘图标一般 32x32 或 64x64)
    let mut resized = Mat::default();
    opencv::imgproc::resize(
        &mat,
        &mut resized,
        opencv::core::Size::new(64, 64),
        0.0,
        0.0,
        opencv::imgproc::INTER_AREA,
    )?;

    // 3. 转换颜色空间 BGR/BGRA -> RGBA
    let mut rgba = Mat::default();
    if resized.channels() == 3 {
        opencv::imgproc::cvt_color(
            &resized,
            &mut rgba,
            opencv::imgproc::COLOR_BGR2RGBA,
            0,
            opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;
    } else if resized.channels() == 4 {
        opencv::imgproc::cvt_color(
            &resized,
            &mut rgba,
            opencv::imgproc::COLOR_BGRA2RGBA,
            0,
            opencv::core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;
    } else {
        // 默认创建一个红色的方块图标防止报错
        let width = 64;
        let height = 64;
        let icon_rgba = vec![255u8; (width * height * 4) as usize];
        return Icon::from_rgba(icon_rgba, width, height).context("无法创建默认图标");
    }

    // 4. 提取数据
    let width = rgba.cols() as u32;
    let height = rgba.rows() as u32;
    let data_bytes = rgba.data_bytes()?;

    // 必须 clone 数据，因为 opencv 的 mat 数据生命周期问题
    let icon_data = data_bytes.to_vec();

    Icon::from_rgba(icon_data, width, height).context("无法构建托盘图标")
}

// -------------------------------------------------------------
// 【修改后】轻量级模式：带消息循环的托盘图标
// -------------------------------------------------------------
fn run_overlay_mode(config: Config) -> Result<()> {
    println!(">> [Mode] 纯 Overlay 模式已启动 (视觉识别已禁用)");
    println!(">> [Info] 交互说明:");
    println!("   - 左键单击图标: 显示/隐藏 Overlay");
    println!("   - 右键单击图标: 打开菜单 (退出)");

    let shared_coords = Arc::new(SharedCursorCoords::new(0, 0));

    let initial_state = if config.assets.enable_overlay {
        overlay::STATE_NORMAL
    } else {
        overlay::STATE_HIDDEN
    };
    let mouse_state = Arc::new(AtomicI32::new(initial_state));

    // =========================================================
    // 1. 初始化系统托盘菜单
    // =========================================================
    let tray_menu = Menu::new();
    let quit_item = MenuItem::new("Exit", true, None);
    // 右键菜单只保留退出，因为左键已经可以控制显隐了，这样更简洁
    tray_menu.append(&quit_item).unwrap();

    let icon = load_tray_icon(&config.assets.cursor_normal)
        .or_else(|_| load_tray_icon("pictures/icon.png"))
        .unwrap_or_else(|_| {
            let w = 64;
            let h = 64;
            let raw = vec![255u8; (w * h * 4) as usize];
            Icon::from_rgba(raw, w, h).unwrap()
        });

    // 构建托盘图标
    // 注意：with_menu 默认绑定的是【右键】
    let _tray_icon = TrayIconBuilder::new()
        .with_menu(Box::new(tray_menu))
        .with_tooltip("Absolute Solver")
        .with_icon(icon)
        .build()
        .context("无法创建系统托盘图标")?;

    // =========================================================
    // 2. 启动 Overlay 线程
    // =========================================================
    if config.assets.enable_overlay && !config.assets.cursor_normal.is_empty() {
        overlay::spawn_mouse_overlay(
            &config.assets,
            mouse_state.clone(),
            shared_coords.clone(),
            false,
        )?;
    }

    let mut input_manager = InputManager::new(&config.shortcuts)?;

    // =========================================================
    // 3. 主循环 (加入 Windows 消息泵)
    // =========================================================
    loop {
        // [关键修复] 显式处理 Windows 消息
        // 如果没有这段代码，右键菜单就不会弹出，因为系统消息被阻塞了
        unsafe {
            let mut msg = MSG::default();
            // PM_REMOVE: 读取并从队列中移除消息
            while PeekMessageW(&mut msg, None, 0, 0, PM_REMOVE).as_bool() {
                let _ = TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
        }

        // A. 检查托盘菜单点击事件 (右键菜单里的选项)
        if let Ok(event) = MenuEvent::receiver().try_recv() {
            if event.id == quit_item.id() {
                println!(">> [Exit] 正在退出...");
                break;
            }
        }

        // B. 检查托盘图标本身的点击事件 (左键/双击)
        if let Ok(event) = TrayIconEvent::receiver().try_recv() {
            match event {
                // 左键点击 -> 切换显隐
                TrayIconEvent::Click {
                    button: MouseButton::Left,
                    ..
                } => {
                    let current = mouse_state.load(Ordering::Relaxed);
                    if current == overlay::STATE_HIDDEN {
                        mouse_state.store(overlay::STATE_NORMAL, Ordering::Relaxed);
                        println!(">> [Tray] 显示 Overlay");
                    } else {
                        mouse_state.store(overlay::STATE_HIDDEN, Ordering::Relaxed);
                        println!(">> [Tray] 隐藏 Overlay");
                    }
                }
                // 双击 -> 这里可以加别的逻辑，比如重置位置
                TrayIconEvent::DoubleClick {
                    button: MouseButton::Left,
                    ..
                } => {
                    println!(">> [Tray] 双击 (保留功能)");
                }
                _ => {}
            }
        }

        // C. 检查键盘快捷键
        match input_manager.check_action() {
            AppAction::ToggleMouse => {
                let current = mouse_state.load(Ordering::Relaxed);
                if current == overlay::STATE_HIDDEN {
                    mouse_state.store(overlay::STATE_NORMAL, Ordering::Relaxed);
                } else {
                    mouse_state.store(overlay::STATE_HIDDEN, Ordering::Relaxed);
                }
            }
            // 可以在这里添加 AppAction::Quit 的处理
            _ => {}
        }

        // 适当休眠，避免死循环占用 100% CPU
        // 50ms 对于 UI 响应来说足够流畅
        std::thread::sleep(Duration::from_millis(50));
    }

    Ok(())
}

impl SolverApp {
    fn new(config: Config) -> Result<Self> {
        // ========================================================
        // 动态调试窗口配置
        // ========================================================
        // let file = File::open(config_path).context("无法打开配置文件")?;
        // let config: Config = serde_yaml::from_reader(file).context("配置格式错误")?;

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
        println!(">> [Init] 系统初始化中 (识别模式)...");

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

        // 初始化共享坐标
        let shared_coords = Arc::new(SharedCursorCoords::new(0, 0));

        // 初始化鼠标控制器 (传入 shared_coords)
        let mouse_controller =
            HeadMouseController::new(config.mouse.clone(), shared_coords.clone())?;

        let input_manager = InputManager::new(&config.shortcuts)?;
        let pose_solver = HeadPoseSolver::new(config.camera.width, config.camera.height)?;
        let gesture_controller = HandGestureController::new(config.gesture.clone());
        let raw_face_mask = algorithms::load_and_prepare_mask(&config.assets.avatar)?;

        // ========================================================
        // 启动鼠标 Overlay
        // ========================================================
        // 初始化状态为隐藏
        let mouse_state = Arc::new(AtomicI32::new(overlay::STATE_HIDDEN));

        // 【修改点】启动 Overlay (增加 config.assets.enable_overlay 判断)
        // 只有当 enable_overlay 为 true 且 图片路径不为空时，才创建窗口
        if config.assets.enable_overlay && !config.assets.cursor_normal.is_empty() {
            overlay::spawn_mouse_overlay(
                &config.assets,
                mouse_state.clone(),
                shared_coords.clone(),
                config.mouse.virtual_cursor_mode,
            )?;
            println!(">> [Overlay] 鼠标跟随窗口已启动");
        } else {
            println!(">> [Overlay] 鼠标跟随窗口已禁用");
        }

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
            mouse_state,
            // shared_coords,
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
                    // 【修复】area() 返回 i32，不能和 0.0 (f64) 比较，修改为 0
                    if hand_rect.area() > 0
                        && (intersection.area() as f32 / hand_rect.area() as f32)
                            > overlap_threshold
                    {
                        hand_result = None;
                    }
                }
            }

            // ==========================================
            // B. 更新手势状态
            // ==========================================
            let current_hand_event = if let Some((ref hand_landmarks, _)) = hand_result {
                self.gesture_controller.process(hand_landmarks)
            } else {
                self.gesture_controller.reset_state()
            };

            // C. 核心追踪逻辑
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

            // ==========================================
            // 决定 Overlay 状态
            // ==========================================
            let mut next_overlay_state = overlay::STATE_HIDDEN;

            if self.mouse_controller.enabled {
                next_overlay_state = overlay::STATE_NORMAL;

                if self.gesture_controller.is_pinched() {
                    next_overlay_state = overlay::STATE_CLICK_HOLD;
                } else {
                    match self.gesture_controller.rotation_state {
                        1 => next_overlay_state = overlay::STATE_SCROLL_CW,
                        -1 => next_overlay_state = overlay::STATE_SCROLL_CCW,
                        _ => {}
                    }
                }
            }

            self.mouse_state
                .store(next_overlay_state, Ordering::Relaxed);

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

            // 1. 绘制手
            if let Some((landmarks, rect)) = hand_result {
                let x = (rect.x as f32 * win_scale) as i32;
                let y = (rect.y as f32 * win_scale) as i32;
                let mirror_x = win_w - (x + (rect.width as f32 * win_scale) as i32);

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
                        // 【修改点】逻辑分流：如果 scale > 0 显示面具，否则显示特征点
                        if self.config.assets.scale > 0.0 {
                            // === 分支 A: 绘制面具 (Avatar) ===
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
                        } else {
                            // === 分支 B: 绘制特征点 (仅当 scale <= 0 时) ===
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
    // 1. 在最外层先加载配置
    let file = File::open("config.yaml").context("无法打开配置文件")?;
    let config: Config = serde_yaml::from_reader(file).context("配置格式错误")?;

    // 2. 根据配置决定进入哪种模式
    if !config.algorithm.enable_recognition {
        // === 分支 A: 纯 Overlay 模式 ===
        // 捕获 Panic 防止闪退
        let result = (|| -> Result<()> {
            run_overlay_mode(config)?;
            Ok(())
        })();

        if let Err(ref e) = result {
            eprintln!("\n>> [Error] Overlay 模式运行出错: {:?}", e);
            let _ = std::io::stdin().read_line(&mut String::new());
        }
        return result;
    }

    // === 分支 B: 完整识别模式 ===
    let result = (|| -> Result<()> {
        // 注意：这里调用的是修改后的 new(config)
        let mut app = SolverApp::new(config)?;
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
