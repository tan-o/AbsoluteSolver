// src/interaction.rs
// ==========================================
// 交互模块 - 整合鼠标控制、手势检测、头部姿态
// ==========================================

use crate::config::{GestureConfig, MouseConfig};
use anyhow::Result;
use enigo::{Axis, Button, Coordinate, Direction, Enigo, Mouse, Settings};
use rdev::display_size;
// 【Fix 1: Added missing imports】
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::Arc;
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
    last_rotation_time: Instant,
    base_angle: Option<f32>,
    config: GestureConfig,
    pub rotation_state: i32,
}

impl HandGestureController {
    pub fn new(config: GestureConfig) -> Self {
        Self {
            is_pinched: false,
            last_rotation_time: Instant::now(),
            base_angle: None,
            config,
            rotation_state: 0,
        }
    }

    pub fn reset_state(&mut self) -> HandEvent {
        self.base_angle = None;
        self.rotation_state = 0;
        if self.is_pinched {
            self.is_pinched = false;
            return HandEvent::PinchEnd;
        }

        self.is_pinched = false;
        HandEvent::None
    }

    pub fn is_pinched(&self) -> bool {
        self.is_pinched
    }

    pub fn solve_palm_center(&self, landmarks: &Vec<[f32; 3]>) -> Option<(f64, f64)> {
        if landmarks.len() < 21 {
            return None;
        }

        let track_indices = [0, 1, 17];
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;

        for &idx in &track_indices {
            sum_x += landmarks[idx][0] as f64;
            sum_y += landmarks[idx][1] as f64;
        }

        let count = track_indices.len() as f64;
        Some((sum_x / count, sum_y / count))
    }

    pub fn process(&mut self, landmarks: &Vec<[f32; 3]>) -> HandEvent {
        if landmarks.len() < 21 {
            self.reset_state();
            return HandEvent::None;
        }

        let p = |i: usize| -> (f32, f32) { (landmarks[i][0], landmarks[i][1]) };

        let (wx, wy) = p(0);
        let (px, py) = p(17);
        let scale_ref = ((wx - px).powi(2) + (wy - py).powi(2)).sqrt();

        if scale_ref < 1.0 {
            self.reset_state();
            return HandEvent::None;
        }

        let fingers_indices = [
            6, 7, 8, // Index
            10, 11, 12, // Middle
            14, 15, 16, // Ring
            18, 19, 20, // Pinky
        ];

        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        for &idx in &fingers_indices {
            let (x, y) = p(idx);
            sum_x += x;
            sum_y += y;
        }

        // 【优化】用倒数乘法替代除法
        let inv_12 = 1.0 / 12.0;
        let center_x = sum_x * inv_12;
        let center_y = sum_y * inv_12;

        let (thumb_x, thumb_y) = p(4);
        let dist_thumb_to_center =
            ((thumb_x - center_x).powi(2) + (thumb_y - center_y).powi(2)).sqrt();

        let ratio = dist_thumb_to_center / scale_ref;

        let mut event = HandEvent::None;

        if !self.is_pinched {
            if ratio < self.config.pinch_threshold_on {
                self.is_pinched = true;
                event = HandEvent::PinchStart;
            }
        } else {
            if ratio > self.config.pinch_threshold_off {
                self.is_pinched = false;
                event = HandEvent::PinchEnd;
            }
        }

        if event != HandEvent::None {
            return event;
        }

        if self.is_pinched {
            return HandEvent::None;
        }

        let (wrist_x, wrist_y) = p(0);
        let finger_indices = [0, 9, 10, 11, 12];
        let mut sum_dx = 0.0;
        let mut sum_dy = 0.0;

        for &idx in &finger_indices {
            let (px, py) = p(idx);
            sum_dx += px - wrist_x;
            sum_dy += py - wrist_y;
        }

        let current_angle = sum_dy.atan2(sum_dx);
        let base = *self.base_angle.get_or_insert(current_angle);
        let mut diff = current_angle - base;

        while diff > std::f32::consts::PI {
            diff -= 2.0 * std::f32::consts::PI;
        }
        while diff < -std::f32::consts::PI {
            diff += 2.0 * std::f32::consts::PI;
        }

        let threshold = self.config.rotation_threshold_degrees.to_radians();

        if diff > threshold {
            self.rotation_state = 1;
        } else if diff < -threshold {
            self.rotation_state = -1;
        } else {
            self.rotation_state = 0;
        }

        if self.rotation_state != 0 {
            if self.last_rotation_time.elapsed().as_millis() > self.config.rotation_cooldown_ms {
                self.last_rotation_time = Instant::now();

                if self.rotation_state == 1 {
                    return HandEvent::RotateCW;
                } else {
                    return HandEvent::RotateCCW;
                }
            }
        }

        HandEvent::None
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

    pub fn get_rigid_landmark_indices() -> &'static [usize] {
        const RIGID_INDICES: &[usize] = &[
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33, 35,
            44, 45, 47, 48, 49, 51, 56, 59, 60, 64, 67, 68, 69, 71, 75, 77, 79, 89, 90, 94, 96, 97,
            98, 99, 100, 102, 103, 104, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
            120, 121, 122, 124, 125, 126, 128, 129, 130, 131, 133, 134, 139, 141, 142, 143, 144,
            145, 151, 153, 154, 155, 156, 157, 158, 159, 160, 161, 163, 166, 168, 173, 174, 188,
            189, 190, 193, 195, 196, 197, 198, 209, 217, 218, 219, 220, 226, 228, 229, 230, 231,
            232, 233, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249,
            250, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 263, 265, 266, 274, 275, 277,
            278, 279, 281, 286, 289, 290, 294, 297, 298, 299, 301, 305, 307, 309, 319, 320, 325,
            326, 327, 328, 329, 330, 331, 332, 333, 337, 338, 339, 340, 341, 342, 343, 344, 346,
            347, 348, 349, 350, 351, 353, 354, 355, 357, 358, 359, 360, 362, 363, 368, 370, 371,
            372, 373, 374, 380, 381, 382, 383, 384, 385, 386, 387, 388, 390, 392, 398, 399, 412,
            413, 414, 417, 419, 420, 423, 429, 437, 438, 439, 440, 446, 448, 449, 450, 451, 452,
            453, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470,
            471, 472, 473, 474, 475, 476, 477, 478,
        ];
        RIGID_INDICES
    }

    pub fn solve_centroid(&self, landmarks: &Vec<[f32; 3]>) -> Option<(f64, f64)> {
        let rigid_indices = Self::get_rigid_landmark_indices();
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let count = rigid_indices.len() as f64;
        let inv_count = 1.0 / count;

        for &idx in rigid_indices {
            if let Some(p) = landmarks.get(idx) {
                sum_x += p[0] as f64;
                sum_y += p[1] as f64;
            }
        }
        Some((sum_x * inv_count, sum_y * inv_count))
    }
}

// ==========================================
// 鼠标控制器
// ==========================================

// 【Fix 2: Added SharedCursorCoords struct】
pub struct SharedCursorCoords {
    pub x: AtomicI32,
    pub y: AtomicI32,
}

impl SharedCursorCoords {
    pub fn new(x: i32, y: i32) -> Self {
        Self {
            x: AtomicI32::new(x),
            y: AtomicI32::new(y),
        }
    }
}

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
    // 【Fix 3: Added shared_coords field】
    pub shared_coords: Arc<SharedCursorCoords>,
}

impl HeadMouseController {
    // 【Fix 4: Updated new() to accept shared_coords】
    pub fn new(config: MouseConfig, shared_coords: Arc<SharedCursorCoords>) -> Result<Self> {
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
            shared_coords,
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
        // Resetting to center also updates the shared coords
        self.shared_coords
            .x
            .store(center_x as i32, Ordering::Relaxed);
        self.shared_coords
            .y
            .store(center_y as i32, Ordering::Relaxed);

        if !self.config.virtual_cursor_mode {
            let _ = self
                .enigo
                .move_mouse(center_x as i32, center_y as i32, Coordinate::Abs);
        }

        println!(">> [Mouse] Reset Center: ({:.1}, {:.1})", cx, cy);
    }

    pub fn toggle(&mut self) {
        self.enabled = !self.enabled;
        println!(">> [Mouse] Enabled: {}", self.enabled);
    }

    // 【Fix 5: Updated update logic】
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

        // Update shared atomic coordinates
        self.shared_coords
            .x
            .store(final_x as i32, Ordering::Relaxed);
        self.shared_coords
            .y
            .store(final_y as i32, Ordering::Relaxed);

        // Only move system mouse if NOT in virtual mode
        if !self.config.virtual_cursor_mode {
            let _ = self
                .enigo
                .move_mouse(final_x as i32, final_y as i32, Coordinate::Abs);
        }
        Ok(())
    }

    // 【Fix 6: Added sync helper】
    fn sync_system_mouse(&mut self) -> Result<()> {
        let x = self.shared_coords.x.load(Ordering::Relaxed);
        let y = self.shared_coords.y.load(Ordering::Relaxed);
        self.enigo
            .move_mouse(x, y, Coordinate::Abs)
            .map_err(|e| anyhow::anyhow!("{:?}", e))
    }

    // 【Fix 7: Updated click/scroll methods (Removed duplicates)】
    pub fn click_down(&mut self) -> Result<()> {
        if self.config.virtual_cursor_mode {
            self.sync_system_mouse()?;
        }
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
        if self.config.virtual_cursor_mode {
            self.sync_system_mouse()?;
        }
        self.enigo
            .scroll(y, Axis::Vertical)
            .map_err(|e| anyhow::anyhow!("{:?}", e))
    }
}
