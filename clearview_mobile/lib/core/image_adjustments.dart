class ImageAdjustments {
  double brightness;
  double saturation;
  double contrast;
  double sharpen;
  bool denoise;
  bool upscale;
  bool detectObjects;
  bool runOcr;

  ImageAdjustments({
    this.brightness = 1.0,
    this.saturation = 1.0,
    this.contrast = 1.0,
    this.sharpen = 0.0,
    this.denoise = true,
    this.upscale = true,
    this.detectObjects = true,
    this.runOcr = true,
  });

  Map<String, dynamic> toJson() => {
        'brightness': brightness,
        'saturation': saturation,
        'contrast': contrast,
        'sharpen': sharpen,
        'denoise': denoise,
        'upscale': upscale,
        'detect_objects': detectObjects,
        'run_ocr': runOcr,
      };
}