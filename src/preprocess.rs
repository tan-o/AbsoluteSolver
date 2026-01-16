use anyhow::{Context, Result};
use opencv::{
    // 这里依赖 Cargo.toml 中的 imgcodecs feature
    core::{self, Mat, Point, Scalar, Size, Vec3b, Vec4b},
    imgcodecs,
    imgproc,
    prelude::*,
};

pub struct PreprocessResult {
    pub img: Mat,
    pub scale: f32,
    pub pad_x: i32,
    pub pad_y: i32,
}

pub fn letterbox(src: &Mat, target_width: i32, target_height: i32) -> Result<PreprocessResult> {
    let src_w = src.cols();
    let src_h = src.rows();
    let scale = (target_width as f32 / src_w as f32).min(target_height as f32 / src_h as f32);
    let new_w = (src_w as f32 * scale).round() as i32;
    let new_h = (src_h as f32 * scale).round() as i32;

    let mut resized = Mat::default();
    imgproc::resize(
        src,
        &mut resized,
        Size::new(new_w, new_h),
        0.0,
        0.0,
        imgproc::INTER_LINEAR,
    )?;

    let dw = target_width - new_w;
    let dh = target_height - new_h;
    let top = dh / 2;
    let bottom = dh - top;
    let left = dw / 2;
    let right = dw - left;

    let mut dst = Mat::default();
    core::copy_make_border(
        &resized,
        &mut dst,
        top,
        bottom,
        left,
        right,
        core::BorderTypes::BORDER_CONSTANT as i32,
        Scalar::new(0.0, 0.0, 0.0, 0.0),
    )?;

    Ok(PreprocessResult {
        img: dst,
        scale,
        pad_x: left,
        pad_y: top,
    })
}

pub fn auto_correct_exposure(src: &Mat) -> Result<Mat> {
    // 【优化】用小尺寸图计算亮度，避免处理全分辨率
    let mut small_src = Mat::default();
    let small_size = Size::new(320, 240);
    imgproc::resize(
        src,
        &mut small_src,
        small_size,
        0.0,
        0.0,
        imgproc::INTER_AREA,
    )?;

    let mut gray = Mat::default();
    if small_src.channels() == 3 {
        imgproc::cvt_color(
            &small_src,
            &mut gray,
            imgproc::COLOR_BGR2GRAY,
            0,
            core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;
    } else {
        small_src.copy_to(&mut gray)?;
    }

    let mean_scalar = core::mean(&gray, &core::no_array())?;
    let mean_brightness = mean_scalar[0] as f32;

    if mean_brightness > 100.0 && mean_brightness < 160.0 {
        return Ok(src.clone());
    }

    let target_brightness: f32 = 130.0;
    let safe_mean = mean_brightness.max(1.0);
    let gamma = ((target_brightness / 255.0).ln() / (safe_mean / 255.0).ln()).clamp(0.4, 3.0);

    let mut lut_data = Vec::with_capacity(256);
    for i in 0..256 {
        let normalized = i as f32 / 255.0;
        let corrected = normalized.powf(gamma) * 255.0;
        lut_data.push(corrected.clamp(0.0, 255.0) as u8);
    }

    let lut_mat = Mat::from_slice(&lut_data)?;
    let lut_final = lut_mat.reshape(1, 256)?;
    let mut dst = Mat::default();
    core::lut(src, &lut_final, &mut dst)?;
    Ok(dst)
}

/// 加载并转换遮罩图片 (只加载原图 + 转 BGRA，不缩放)
pub fn load_and_prepare_mask(path: &str) -> Result<Mat> {
    // 读取原始图片
    let raw_mask = imgcodecs::imread(path, imgcodecs::IMREAD_UNCHANGED)
        .with_context(|| format!("无法加载图片: {}", path))?;

    // 确保格式为 BGRA (4通道)
    let mut final_mask = Mat::default();
    if raw_mask.channels() == 3 {
        // 如果是 JPG/BMP (3通道)，转 BGRA 并加上全不透明 Alpha
        imgproc::cvt_color(
            &raw_mask,
            &mut final_mask,
            imgproc::COLOR_BGR2BGRA,
            0,
            core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )?;
    } else if raw_mask.channels() == 4 {
        final_mask = raw_mask;
    } else {
        anyhow::bail!("不支持的通道数: {}", raw_mask.channels());
    }

    Ok(final_mask)
}

pub fn overlay_image(background: &mut Mat, foreground: &Mat, top_left: Point) -> Result<()> {
    let bg_w = background.cols();
    let bg_h = background.rows();
    let fg_w = foreground.cols();
    let fg_h = foreground.rows();

    let x = top_left.x;
    let y = top_left.y;

    if x + fg_w <= 0 || x >= bg_w || y + fg_h <= 0 || y >= bg_h {
        return Ok(());
    }

    let start_x = x.max(0);
    let start_y = y.max(0);
    let end_x = (x + fg_w).min(bg_w);
    let end_y = (y + fg_h).min(bg_h);
    let w = end_x - start_x;
    let h = end_y - start_y;

    if w <= 0 || h <= 0 {
        return Ok(());
    }

    let fg_off_x = start_x - x;
    let fg_off_y = start_y - y;

    for row in 0..h {
        for col in 0..w {
            let bg_y = start_y + row;
            let bg_x = start_x + col;
            let fg_y = fg_off_y + row;
            let fg_x = fg_off_x + col;

            let fg_pixel: Vec4b = *foreground.at_2d(fg_y, fg_x)?;
            let alpha = fg_pixel[3] as f32 / 255.0;

            if alpha < 0.01 {
                continue;
            }

            let bg_pixel_ptr = background.at_2d_mut::<Vec3b>(bg_y, bg_x)?;

            if alpha > 0.99 {
                bg_pixel_ptr[0] = fg_pixel[0];
                bg_pixel_ptr[1] = fg_pixel[1];
                bg_pixel_ptr[2] = fg_pixel[2];
            } else {
                let inv_alpha = 1.0 - alpha;
                bg_pixel_ptr[0] =
                    (fg_pixel[0] as f32 * alpha + bg_pixel_ptr[0] as f32 * inv_alpha) as u8;
                bg_pixel_ptr[1] =
                    (fg_pixel[1] as f32 * alpha + bg_pixel_ptr[1] as f32 * inv_alpha) as u8;
                bg_pixel_ptr[2] =
                    (fg_pixel[2] as f32 * alpha + bg_pixel_ptr[2] as f32 * inv_alpha) as u8;
            }
        }
    }
    Ok(())
}
