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
pub struct HandGestureController {
    is_pinched: bool,
    pinch_dist_smooth: f32,
    last_rotation_time: Instant,
    base_angle: Option<f32>, // 【新增】基准角度
    // 配置参数
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

    pub fn is_pinched(&self) -> bool {
        self.is_pinched
    }

    pub fn process(&mut self, landmarks: &Vec<[f32; 3]>) -> HandEvent {
        if landmarks.len() < 21 {
            return HandEvent::None;
        }

        // 辅助函数
        let p = |i: usize| -> (f32, f32) { (landmarks[i][0], landmarks[i][1]) };

        // 1. 计算手掌尺度 (Palm Scale)
        let (wx, wy) = p(0);
        let roots = [5, 9, 13, 17];
        let mut sum_dist = 0.0f32;

        for &r in &roots {
            let (rx, ry) = p(r);
            sum_dist += ((rx - wx).powi(2) + (ry - wy).powi(2)).sqrt();
        }
        let palm_scale = sum_dist / 4.0;

        if palm_scale < 0.001 {
            return HandEvent::None;
        }

        // 2. 计算捏合距离 (Pinch Distance)
        let (tx, ty) = p(4); // 拇指尖
        let (i_x, i_y) = p(8); // 食指尖
        let (m_x, m_y) = p(12); // 中指尖

        let dist_index = ((tx - i_x).powi(2) + (ty - i_y).powi(2)).sqrt();
        let dist_middle = ((tx - m_x).powi(2) + (ty - m_y).powi(2)).sqrt();

        // 取两者中较小的一个，或者平均值，这里建议取平均值以获得更好的稳定性
        let raw_dist = (dist_index + dist_middle) / 2.0;
        let normalized = raw_dist / palm_scale;

        // 3. 更快的滤波 - 提高响应速度
        // 根据配置的平滑因子动态调整：0.3 = 快速响应，0.5+ = 平滑
        let alpha_high = 1.0 - self.config.pinch_smooth_factor; // 新数据权重
        let alpha_low = self.config.pinch_smooth_factor; // 旧数据权重
        self.pinch_dist_smooth = self.pinch_dist_smooth * alpha_low + normalized * alpha_high;

        // 4. 状态机 - 使用配置的阈值
        let mut event = HandEvent::None;

        if !self.is_pinched {
            // 只要一过线，立即触发！支持极速点击
            if self.pinch_dist_smooth < self.config.pinch_threshold_on {
                self.is_pinched = true;
                event = HandEvent::PinchStart;
            }
        } else {
            // 松开检测
            if self.pinch_dist_smooth > self.config.pinch_threshold_off {
                self.is_pinched = false;
                event = HandEvent::PinchEnd;
            }
        }

        if self.is_pinched {
            return event;
        }

        // 5. 改进的相对旋转检测 - 使用中指和手掌底部中轴线
        // 中指尖端: 索引12 (Middle finger tip)
        // 手掌底部中轴: 食指根部(5)和小指根部(17)的中点
        let (m_tip_x, m_tip_y) = p(12); // 中指尖端
        
        // 计算手掌底部中轴的中点
        let (index_root_x, index_root_y) = p(5);  // 食指根部
        let (pinky_root_x, pinky_root_y) = p(17); // 小指根部
        let palm_center_x = (index_root_x + pinky_root_x) / 2.0;
        let palm_center_y = (index_root_y + pinky_root_y) / 2.0;

        // 从手掌底部中点到中指尖的角度
        let current_angle = (m_tip_y - palm_center_y).atan2(m_tip_x - palm_center_x);

        // 如果是新检测到的手，记录基准角度
        let base = *self.base_angle.get_or_insert(current_angle);

        // 计算角度差值并归一化到 [-PI, PI]
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

    /// 这里的逻辑是：物理平均。。
    pub fn solve_centroid(&self, landmarks: &Vec<[f32; 3]>) -> Option<(f64, f64)> {
        // 刚性点集合 (Rigid Landmarks) - 仅包含骨骼点
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

        // 返回极其稳定的几何重心
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

    // 复刻 HTML 的双重状态
    filtered_pos: Point2D, // 滤波后的位置
    cursor_pos: Point2D,   // 实际光标位置
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

        // 重置所有状态
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

        // 1. 计算位移
        let mut dx = cx - anchor.x;
        let mut dy = cy - anchor.y;

        // 简单死区
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

        // 2. 目标坐标 (Raw Target)
        let target_x = (self.screen_width / 2.0) + dx * self.config.sensitivity_x as f64;
        let target_y = (self.screen_height / 2.0) + dy * self.config.sensitivity_y as f64;

        // ============================================================
        // 【复刻 HTML 双重滤波算法】
        // ============================================================

        let ema_alpha = self.config.smoothing.clamp(0.01, 1.0) as f64;

        // 第一层：指数移动平均
        self.filtered_pos.x = self.filtered_pos.x * (1.0 - ema_alpha) + target_x * ema_alpha;
        self.filtered_pos.y = self.filtered_pos.y * (1.0 - ema_alpha) + target_y * ema_alpha;

        // 第二层：插值跟随
        let interp = 0.4;
        self.cursor_pos.x += (self.filtered_pos.x - self.cursor_pos.x) * interp;
        self.cursor_pos.y += (self.filtered_pos.y - self.cursor_pos.y) * interp;

        // 3. 边界限制
        let final_x = self.cursor_pos.x.clamp(0.0, self.screen_width);
        let final_y = self.cursor_pos.y.clamp(0.0, self.screen_height);

        let _ = self
            .enigo
            .move_mouse(final_x as i32, final_y as i32, Coordinate::Abs);

        Ok(())
    }

    // ==========================================
    // 手势操作接口
    // ==========================================

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
