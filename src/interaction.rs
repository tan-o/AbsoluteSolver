// ==========================================
// 交互模块 - 整合鼠标控制、手势检测、头部姿态
// ==========================================

use crate::config::{GestureConfig, MouseConfig};
use anyhow::Result;
use enigo::{Axis, Button, Coordinate, Direction, Enigo, Mouse, Settings};
use rdev::display_size;
use std::time::Instant;

// ==========================================
// 手势类型定义
// ==========================================
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HandEvent {
    None,
    PinchStart,
    PinchEnd,
    RotateCW,
    RotateCCW,
}

// ==========================================
// 手势检测控制器
// ==========================================
// ==========================================
// 手势检测控制器
// ==========================================
pub struct HandGestureController {
    is_pinched: bool,
    pinch_dist_smooth: f32,
    last_rotation_time: Instant,
    base_angle: Option<f32>, // 基准角度
    config: GestureConfig,
}

impl HandGestureController {
    pub fn new(config: GestureConfig) -> Self {
        Self {
            is_pinched: false,
            pinch_dist_smooth: 1.0,
            last_rotation_time: Instant::now(),
            base_angle: None,
            config,
        }
    }

    /// 【新增】强制重置状态，供外部（main.rs）在检测不到手时调用
    pub fn reset_state(&mut self) {
        if self.base_angle.is_some() {
            // 只有当有状态时才打印，避免刷屏
            // println!(">> [Gesture] Hand lost, resetting rotation base.");
        }
        self.base_angle = None;
        self.is_pinched = false;
        self.pinch_dist_smooth = 1.0; // 重置捏合平滑值，防止下次出现时误触
    }

    pub fn is_pinched(&self) -> bool {
        self.is_pinched
    }

    pub fn process(&mut self, landmarks: &Vec<[f32; 3]>) -> HandEvent {
        // 1. 安全检查：如果点数不够，视为手丢失，重置状态
        if landmarks.len() < 21 {
            self.reset_state();
            return HandEvent::None;
        }

        // 辅助函数
        let p = |i: usize| -> (f32, f32) { (landmarks[i][0], landmarks[i][1]) };

        // 2. 计算手掌尺度
        let (wx, wy) = p(0);
        let roots = [5, 9, 13, 17];
        let mut sum_dist = 0.0f32;
        for &r in &roots {
            let (rx, ry) = p(r);
            sum_dist += ((rx - wx).powi(2) + (ry - wy).powi(2)).sqrt();
        }
        let palm_scale = sum_dist / 4.0;

        if palm_scale < 0.001 {
            self.reset_state();
            return HandEvent::None;
        }

        // ============================================================
        // 捏合检测 (Pinch)
        // ============================================================
        let (tx, ty) = p(4);
        let (i_x, i_y) = p(8);
        let (m_x, m_y) = p(12);

        let dist_index = ((tx - i_x).powi(2) + (ty - i_y).powi(2)).sqrt();
        let dist_middle = ((tx - m_x).powi(2) + (ty - m_y).powi(2)).sqrt();
        let raw_dist = (dist_index + dist_middle) / 2.0;
        let normalized = raw_dist / palm_scale;

        let alpha_high = 1.0 - self.config.pinch_smooth_factor;
        let alpha_low = self.config.pinch_smooth_factor;
        self.pinch_dist_smooth = self.pinch_dist_smooth * alpha_low + normalized * alpha_high;

        let mut event = HandEvent::None;

        if !self.is_pinched {
            if self.pinch_dist_smooth < self.config.pinch_threshold_on {
                self.is_pinched = true;
                event = HandEvent::PinchStart;
            }
        } else {
            if self.pinch_dist_smooth > self.config.pinch_threshold_off {
                self.is_pinched = false;
                event = HandEvent::PinchEnd;
            }
        }

        if self.is_pinched {
            return event;
        }

        // ============================================================
        // 旋转检测 (Rotation) - 使用 0, 9, 10, 11, 12 向量平均法
        // ============================================================

        let (wrist_x, wrist_y) = p(0);

        // 计算中指 4 个关键点相对于手腕的向量和
        // 这种方法比线性回归更适合这种短线段，且计算极快
        let finger_indices = [9, 10, 11, 12];
        let mut sum_dx = 0.0;
        let mut sum_dy = 0.0;

        for &idx in &finger_indices {
            let (px, py) = p(idx);
            sum_dx += px - wrist_x;
            sum_dy += py - wrist_y;
        }

        // 计算合成角度
        let current_angle = sum_dy.atan2(sum_dx);

        // 【关键】如果是新出现的手（base_angle 为 None），立即锁定当前角度为基准
        let base = *self.base_angle.get_or_insert(current_angle);

        // 计算差值
        let mut diff = current_angle - base;
        while diff > std::f32::consts::PI {
            diff -= 2.0 * std::f32::consts::PI;
        }
        while diff < -std::f32::consts::PI {
            diff += 2.0 * std::f32::consts::PI;
        }

        if self.last_rotation_time.elapsed().as_millis() > self.config.rotation_cooldown_ms {
            let threshold = self.config.rotation_threshold_degrees.to_radians();

            if diff > threshold {
                self.last_rotation_time = Instant::now();
                return HandEvent::RotateCW;
            } else if diff < -threshold {
                self.last_rotation_time = Instant::now();
                return HandEvent::RotateCCW;
            }
        }

        event
    }
}

// ==========================================
// 头部姿态求解器
// ==========================================
pub struct HeadPoseSolver;

impl HeadPoseSolver {
    pub fn new(_w: i32, _h: i32) -> Result<Self> {
        Ok(Self)
    }

    /// 获取实际使用的刚性特征点索引
    pub fn get_rigid_landmark_indices() -> &'static [usize] {
        const RIGID_INDICES: &[usize] = &[
            // === 1. 中轴线 (从发际线到鼻尖) ===
            10, 151, 9, 8, 168, 6, 197, 195, 5, 4, 1,
            // === 2. 整个额头区域 (高密度网格) ===
            // 左额头
            109, 67, 103, 54, 21, 162, 127, 234, 93, // 右额头
            338, 297, 332, 284, 251, 389, 356, 454, 323,
            // === 3. 眉骨 (眉毛下方的骨头) ===
            // 左眉
            46, 53, 52, 65, 55, 70, 63, 105, 66, 107, // 右眉
            276, 283, 282, 295, 285, 336, 296, 334, 293, 300,
            // === 4. 眼眶骨 (红框下边缘划过的地方) ===
            // 左眼眶下沿 (避开会动的眼皮)
            117, 118, 119, 120, 121, 47, // 右眼眶下沿
            346, 347, 348, 349, 350, 277,
            // === 5. 鼻梁两侧与脸颊连接处 ===
            // 左侧
            123, 50, 114, 192, // 右侧
            352, 280, 343, 416,
        ];
        RIGID_INDICES
    }

    pub fn solve_centroid(&self, landmarks: &Vec<[f32; 3]>) -> Option<(f64, f64)> {
        let rigid_indices = Self::get_rigid_landmark_indices();

        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let count = rigid_indices.len() as f64;

        for &idx in rigid_indices {
            if let Some(p) = landmarks.get(idx) {
                sum_x += p[0] as f64;
                sum_y += p[1] as f64;
            }
        }

        Some((sum_x / count, sum_y / count))
    }
}

// ==========================================
// 鼠标控制器
// ==========================================
#[derive(Clone, Copy)]
struct Point2D {
    x: f64,
    y: f64,
}

pub struct HeadMouseController {
    pub enigo: Enigo,
    config: MouseConfig,
    screen_width: f64,
    screen_height: f64,

    pub enabled: bool,
    anchor: Option<Point2D>,

    filtered_pos: Point2D,
    cursor_pos: Point2D,
}

impl HeadMouseController {
    pub fn new(config: MouseConfig) -> Result<Self> {
        let (sw, sh) = if config.auto_screen_size {
            let (w, h) =
                display_size().unwrap_or((config.manual_width as u64, config.manual_height as u64));
            (w as f64, h as f64)
        } else {
            (config.manual_width, config.manual_height)
        };

        let enigo = Enigo::new(&Settings::default())
            .map_err(|e| anyhow::anyhow!("Enigo 初始化失败: {:?}", e))?;

        Ok(Self {
            enigo,
            config,
            screen_width: sw,
            screen_height: sh,
            enabled: true,
            anchor: None,
            filtered_pos: Point2D {
                x: sw / 2.0,
                y: sh / 2.0,
            },
            cursor_pos: Point2D {
                x: sw / 2.0,
                y: sh / 2.0,
            },
        })
    }

    pub fn reset_anchor(&mut self, cx: f64, cy: f64) {
        self.anchor = Some(Point2D { x: cx, y: cy });

        let center_x = self.screen_width / 2.0;
        let center_y = self.screen_height / 2.0;

        self.filtered_pos = Point2D {
            x: center_x,
            y: center_y,
        };
        self.cursor_pos = Point2D {
            x: center_x,
            y: center_y,
        };

        let _ = self
            .enigo
            .move_mouse(center_x as i32, center_y as i32, Coordinate::Abs);
        println!(">> [Mouse] Reset Center: ({:.1}, {:.1})", cx, cy);
    }

    pub fn toggle(&mut self) {
        self.enabled = !self.enabled;
        println!(">> [Mouse] Enabled: {}", self.enabled);
    }

    pub fn update(&mut self, cx: f64, cy: f64) -> Result<()> {
        if !self.enabled || !self.config.enabled {
            return Ok(());
        }

        if self.anchor.is_none() {
            self.reset_anchor(cx, cy);
            return Ok(());
        }
        let anchor = self.anchor.unwrap();

        let mut dx = cx - anchor.x;
        let mut dy = cy - anchor.y;

        if dx.abs() < self.config.dead_zone as f64 {
            dx = 0.0;
        }
        if dy.abs() < self.config.dead_zone as f64 {
            dy = 0.0;
        }

        if self.config.invert_x {
            dx = -dx;
        }
        if self.config.invert_y {
            dy = -dy;
        }

        let target_x = (self.screen_width / 2.0) + dx * self.config.sensitivity_x as f64;
        let target_y = (self.screen_height / 2.0) + dy * self.config.sensitivity_y as f64;

        let ema_alpha = self.config.smoothing.clamp(0.01, 1.0) as f64;
        self.filtered_pos.x = self.filtered_pos.x * (1.0 - ema_alpha) + target_x * ema_alpha;
        self.filtered_pos.y = self.filtered_pos.y * (1.0 - ema_alpha) + target_y * ema_alpha;

        let interp = 0.4;
        self.cursor_pos.x += (self.filtered_pos.x - self.cursor_pos.x) * interp;
        self.cursor_pos.y += (self.filtered_pos.y - self.cursor_pos.y) * interp;

        let final_x = self.cursor_pos.x.clamp(0.0, self.screen_width);
        let final_y = self.cursor_pos.y.clamp(0.0, self.screen_height);

        let _ = self
            .enigo
            .move_mouse(final_x as i32, final_y as i32, Coordinate::Abs);

        Ok(())
    }

    pub fn click_down(&mut self) -> Result<()> {
        self.enigo
            .button(Button::Left, Direction::Press)
            .map_err(|e| anyhow::anyhow!("{:?}", e))
    }

    pub fn click_up(&mut self) -> Result<()> {
        self.enigo
            .button(Button::Left, Direction::Release)
            .map_err(|e| anyhow::anyhow!("{:?}", e))
    }

    pub fn scroll(&mut self, y: i32) -> Result<()> {
        self.enigo
            .scroll(y, Axis::Vertical)
            .map_err(|e| anyhow::anyhow!("{:?}", e))
    }
}
